"""Tests for sweep spec parsing and expansion (cross vs. zip, deterministic ids)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cga.core.errors import ConfigError
from cga.core.sweep import SweepSpec, expand_sweep


def _write_spec(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_from_file_parses_pipeline_dataset_and_axes(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "pipeline": "training_data",
            "dataset": "attack_cwe",
            "sweep": {"training_data.sampler.k": [5, 10, 20]},
        },
    )
    spec = SweepSpec.from_file(path)
    assert spec.pipeline == "training_data"
    assert spec.dataset == "attack_cwe"
    assert spec.axes == {"training_data.sampler.k": [5, 10, 20]}
    assert spec.zipped_keys == ()


def test_from_file_missing_required_key_raises(tmp_path):
    path = _write_spec(tmp_path, {"pipeline": "training_data", "sweep": {}})
    with pytest.raises(ConfigError):
        SweepSpec.from_file(path)


def test_from_file_zip_key_removed_from_axes(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "pipeline": "training_data",
            "dataset": "attack_cwe",
            "sweep": {
                "zip": ["training_data.sampler.top_pct", "training_data.sampler.bottom_pct"],
                "training_data.sampler.top_pct": [0.02, 0.04],
                "training_data.sampler.bottom_pct": [0.02, 0.04],
            },
        },
    )
    spec = SweepSpec.from_file(path)
    assert "zip" not in spec.axes
    assert spec.zipped_keys == ("training_data.sampler.top_pct", "training_data.sampler.bottom_pct")


def test_from_file_zip_length_mismatch_raises(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "pipeline": "training_data",
            "dataset": "attack_cwe",
            "sweep": {
                "zip": ["a.x", "a.y"],
                "a.x": [1, 2, 3],
                "a.y": [1, 2],
            },
        },
    )
    with pytest.raises(ConfigError):
        SweepSpec.from_file(path)


def test_from_file_zip_key_not_in_sweep_raises(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "pipeline": "training_data",
            "dataset": "attack_cwe",
            "sweep": {"zip": ["a.x", "a.missing"], "a.x": [1, 2]},
        },
    )
    with pytest.raises(ConfigError):
        SweepSpec.from_file(path)


def test_expand_sweep_cross_product_single_axis():
    spec = SweepSpec(
        pipeline="training_data",
        dataset="attack_cwe",
        axes={"training_data.sampler.k": [5, 10, 20, 30]},
    )
    points = expand_sweep(spec)
    assert [p.overrides["training_data.sampler.k"] for p in points] == [5, 10, 20, 30]
    assert [p.sweep_point_id for p in points] == ["k5", "k10", "k20", "k30"]


def test_expand_sweep_crosses_two_independent_axes():
    spec = SweepSpec(
        pipeline="gcl_train",
        dataset="attack_cwe",
        axes={"gcl_train.model.encoder": ["gcn", "gat"], "gcl_train.loss.name": ["triplet", "infonce"]},
    )
    points = expand_sweep(spec)
    combos = {(p.overrides["gcl_train.model.encoder"], p.overrides["gcl_train.loss.name"]) for p in points}
    assert combos == {("gcn", "triplet"), ("gcn", "infonce"), ("gat", "triplet"), ("gat", "infonce")}
    assert len(points) == 4


def test_expand_sweep_zip_pairs_by_index_not_cross_product():
    spec = SweepSpec(
        pipeline="training_data",
        dataset="attack_cwe",
        axes={
            "training_data.sampler.top_pct": [0.02, 0.04, 0.08],
            "training_data.sampler.bottom_pct": [0.02, 0.04, 0.08],
        },
        zipped_keys=("training_data.sampler.top_pct", "training_data.sampler.bottom_pct"),
    )
    points = expand_sweep(spec)
    assert len(points) == 3  # paired, not 3x3=9
    for point in points:
        assert point.overrides["training_data.sampler.top_pct"] == point.overrides["training_data.sampler.bottom_pct"]


def test_expand_sweep_zip_axis_crossed_against_other_keys():
    spec = SweepSpec(
        pipeline="training_data",
        dataset="attack_cwe",
        axes={
            "training_data.sampler.name": ["percentile"],
            "training_data.sampler.top_pct": [0.02, 0.04],
            "training_data.sampler.bottom_pct": [0.02, 0.04],
        },
        zipped_keys=("training_data.sampler.top_pct", "training_data.sampler.bottom_pct"),
    )
    points = expand_sweep(spec)
    # name has 1 value, zipped pair has 2 -> 2 total combinations.
    assert len(points) == 2


def test_expand_sweep_sweep_point_id_is_deterministic():
    spec = SweepSpec(pipeline="training_data", dataset="attack_cwe", axes={"training_data.sampler.k": [20]})
    ids_first = [p.sweep_point_id for p in expand_sweep(spec)]
    ids_second = [p.sweep_point_id for p in expand_sweep(spec)]
    assert ids_first == ids_second == ["k20"]


def test_expand_sweep_duplicate_sweep_point_id_raises():
    # "10" (int) and "10" (str) are different combinations but render to the
    # same id string ("k10") -- the guard must catch this rather than let one
    # combination's run silently overwrite the other's.
    spec = SweepSpec(
        pipeline="training_data",
        dataset="attack_cwe",
        axes={"a.k": [10, "10"]},
    )
    with pytest.raises(ConfigError):
        expand_sweep(spec)
