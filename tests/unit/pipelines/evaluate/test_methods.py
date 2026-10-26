"""Unit tests for ``MethodSpec`` and ``load_method``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cga.core.artifacts import write_latest
from cga.core.errors import ConfigError, MissingInputError
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.methods import MethodSpec, load_method


def _config(tmp_path: Path) -> dict:
    return {
        "paths": {"artifacts": str(tmp_path / "artifacts"), "models": str(tmp_path / "models")},
        "seed": 42,
        "device": "cpu",
        "logging": {"level": "INFO", "file": False},
    }


def _ctx(config: dict, dataset_id: str = "attack_cwe") -> RunContext:
    return RunContext.create(
        pipeline_name="evaluate",
        pipeline_version="0.0.0-test",
        dataset_id=dataset_id,
        config=config,
        run_root=Path(config["paths"]["artifacts"]) / "evaluations" / dataset_id,
    )


def test_method_spec_from_config_embedding_store():
    spec = MethodSpec.from_config(
        {"method_id": "dw", "label": "DeepWalk", "kind": "embedding_store", "embedder_id": "deepwalk-x", "ref": "latest"}
    )
    assert spec.method_id == "dw"
    assert spec.kind == "embedding_store"
    assert spec.embedder_id == "deepwalk-x"


def test_method_spec_from_config_gcl_run():
    spec = MethodSpec.from_config({"method_id": "gcn", "kind": "gcl_run", "model_id": "gcn", "ref": "latest"})
    assert spec.kind == "gcl_run"
    assert spec.model_id == "gcn"
    assert spec.label == "gcn"  # defaults to method_id


def test_method_spec_unknown_kind_raises():
    with pytest.raises(ConfigError):
        MethodSpec.from_config({"method_id": "x", "kind": "bogus"})


def test_method_spec_embedding_store_missing_embedder_id_raises():
    with pytest.raises(ConfigError):
        MethodSpec.from_config({"method_id": "x", "kind": "embedding_store"})


def test_method_spec_gcl_run_missing_model_id_raises():
    with pytest.raises(ConfigError):
        MethodSpec.from_config({"method_id": "x", "kind": "gcl_run"})


def test_load_method_embedding_store_resolves_and_records_manifest_input(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    store_dir = Path(config["paths"]["artifacts"]) / "embeddings" / "attack_cwe" / "fake-dw" / "run-1"
    AlignedMatrix(vectors=np.zeros((2, 3), dtype=np.float32), node_ids=["a", "b"]).save(store_dir)
    write_latest(store_dir.parent, store_dir.name)

    spec = MethodSpec.from_config(
        {"method_id": "dw", "kind": "embedding_store", "embedder_id": "fake-dw", "ref": "latest"}
    )
    matrix, run_id = load_method(ctx, spec)

    assert run_id == "run-1"
    assert matrix.node_ids == ["a", "b"]
    assert any(inp["name"] == "method:dw" and inp["run_id"] == "run-1" for inp in ctx.manifest.inputs)


def test_load_method_gcl_run_resolves_node_embeddings_subdir(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    model_lineage = Path(config["paths"]["models"]) / "gcl" / "attack_cwe" / "gcn"
    run_dir = model_lineage / "run-1"
    AlignedMatrix(vectors=np.zeros((2, 4), dtype=np.float32), node_ids=["a", "b"]).save(
        run_dir / "node_embeddings"
    )
    write_latest(model_lineage, run_dir.name)

    spec = MethodSpec.from_config({"method_id": "gcn", "kind": "gcl_run", "model_id": "gcn", "ref": "latest"})
    matrix, run_id = load_method(ctx, spec)

    assert run_id == "run-1"
    assert matrix.node_ids == ["a", "b"]


def test_load_method_missing_latest_raises(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    spec = MethodSpec.from_config(
        {"method_id": "dw", "kind": "embedding_store", "embedder_id": "never-created", "ref": "latest"}
    )
    with pytest.raises(MissingInputError):
        load_method(ctx, spec)


# --- fusion (Sprint 6.1) ---------------------------------------------------


def _save_run(config: dict, model_id: str, vectors: np.ndarray, ids: list[str], run: str = "run-1") -> None:
    lineage = Path(config["paths"]["models"]) / "gcl" / "attack_cwe" / model_id
    AlignedMatrix(vectors=vectors.astype(np.float32), node_ids=ids).save(lineage / run / "node_embeddings")
    write_latest(lineage, run)


def _fusion_entry(weights: dict[str, float]) -> dict:
    return {
        "method_id": "fused",
        "kind": "fusion",
        "members": [{"kind": "gcl_run", "model_id": m, "ref": "latest", "weight": w} for m, w in weights.items()],
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fusion_cosine_is_weighted_mean_of_member_cosines(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    rng = np.random.default_rng(0)
    ids = ["a", "b", "c"]
    mats = {f"m{i}": rng.normal(size=(3, 4 + i)) * (i + 1) for i in range(3)}  # different dims and scales
    for name, v in mats.items():
        _save_run(config, name, v, ids)
    weights = {"m0": 1.0, "m1": 2.0, "m2": 0.5}

    matrix, _ = load_method(ctx, MethodSpec.from_config(_fusion_entry(weights)))

    z = matrix.vectors
    got = _cos(z[0], z[2])
    want = sum(w * _cos(mats[m][0], mats[m][2]) for m, w in weights.items()) / sum(weights.values())
    assert got == pytest.approx(want, abs=1e-5)
    assert float(z[0] @ z[0]) == pytest.approx(sum(weights.values()), abs=1e-5)


def test_fusion_single_member_equals_itself(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    v = np.random.default_rng(1).normal(size=(3, 5))
    _save_run(config, "m0", v, ["a", "b", "c"])
    matrix, _ = load_method(ctx, MethodSpec.from_config(_fusion_entry({"m0": 3.0})))
    assert _cos(matrix.vectors[0], matrix.vectors[1]) == pytest.approx(_cos(v[0], v[1]), abs=1e-5)


def test_fusion_zero_weight_member_is_dropped(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    ids = ["a", "b"]
    _save_run(config, "m0", np.eye(2), ids)
    # "never-created" would raise MissingInputError if it were resolved.
    spec = MethodSpec.from_config(_fusion_entry({"m0": 1.0, "never-created": 0.0}))
    assert len(spec.members) == 1
    _, run_id = load_method(ctx, spec)
    assert run_id == "run-1"


def test_fusion_bad_weights_raise():
    with pytest.raises(ConfigError):
        MethodSpec.from_config(_fusion_entry({"m0": 0.0, "m1": 0.0}))
    with pytest.raises(ConfigError):
        MethodSpec.from_config(_fusion_entry({"m0": 1.0, "m1": -0.5}))
    with pytest.raises(ConfigError):
        MethodSpec.from_config({"method_id": "f", "kind": "fusion"})
    with pytest.raises(ConfigError):
        MethodSpec.from_config({"method_id": "f", "kind": "fusion", "members": [{"kind": "fusion", "weight": 1}]})


def test_fusion_uses_node_intersection_and_empty_raises(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    _save_run(config, "m0", np.eye(3), ["a", "b", "c"])
    _save_run(config, "m1", np.eye(3), ["b", "c", "d"])
    matrix, _ = load_method(ctx, MethodSpec.from_config(_fusion_entry({"m0": 1.0, "m1": 1.0})))
    assert matrix.node_ids == ["b", "c"]

    _save_run(config, "m2", np.eye(2), ["x", "y"])
    with pytest.raises(ConfigError):
        load_method(ctx, MethodSpec.from_config(_fusion_entry({"m0": 1.0, "m2": 1.0})))


def test_fusion_run_id_joins_members_and_records_inputs(tmp_path):
    config = _config(tmp_path)
    ctx = _ctx(config)
    _save_run(config, "m0", np.eye(2), ["a", "b"], run="run-A")
    _save_run(config, "m1", np.eye(2), ["a", "b"], run="run-B")
    _, run_id = load_method(ctx, MethodSpec.from_config(_fusion_entry({"m0": 1.0, "m1": 1.0})))
    assert run_id == "run-A+run-B"
    names = {inp["name"] for inp in ctx.manifest.inputs}
    assert {"method:fused:m0", "method:fused:m1"} <= names
