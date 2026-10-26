"""Tests for config layering, override precedence, interpolation, and hashing."""

from __future__ import annotations

import pytest

from cga.core.config import build_config, config_hash, deep_merge, interpolate
from cga.core.errors import ConfigError


def test_deep_merge_overrides_scalars_and_merges_nested_dicts():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"a": 2, "nested": {"y": 3, "z": 4}}
    result = deep_merge(base, override)
    assert result == {"a": 2, "nested": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_replaces_lists_wholesale():
    base = {"items": [1, 2, 3]}
    override = {"items": [4]}
    assert deep_merge(base, override) == {"items": [4]}


def test_build_config_layers_in_order_later_wins(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\nb: 1\n", encoding="utf-8")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("b: 2\nc: 2\n", encoding="utf-8")
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("c: 3\n", encoding="utf-8")

    resolved = build_config(base_path=base, dataset_path=dataset, pipeline_path=pipeline)
    assert resolved == {"a": 1, "b": 2, "c": 3}


def test_build_config_set_overrides_win_over_all_files(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("train:\n  epochs: 10\n", encoding="utf-8")

    resolved = build_config(base_path=base, overrides={"train.epochs": 5})
    assert resolved["train"]["epochs"] == 5


def test_build_config_set_creates_intermediate_keys(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\n", encoding="utf-8")

    resolved = build_config(base_path=base, overrides={"new.nested.key": "value"})
    assert resolved["new"]["nested"]["key"] == "value"


def test_interpolate_paths_reference():
    resolved = {"paths": {"raw": "data/raw"}, "dataset": {"root": "${paths.raw}/attack_cwe"}}
    out = interpolate(resolved, resolved)
    assert out["dataset"]["root"] == "data/raw/attack_cwe"


def test_interpolate_env_reference(monkeypatch):
    monkeypatch.setenv("MY_VAR", "hello")
    resolved = {"value": "${env:MY_VAR}"}
    out = interpolate(resolved, resolved)
    assert out["value"] == "hello"


def test_interpolate_missing_path_raises():
    resolved = {"value": "${paths.nonexistent}"}
    with pytest.raises(ConfigError):
        interpolate(resolved, resolved)


def test_interpolate_missing_env_raises(monkeypatch):
    monkeypatch.delenv("UNSET_VAR", raising=False)
    resolved = {"value": "${env:UNSET_VAR}"}
    with pytest.raises(ConfigError):
        interpolate(resolved, resolved)


def test_config_hash_stable_regardless_of_key_order():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_content():
    assert config_hash({"x": 1}) != config_hash({"x": 2})
