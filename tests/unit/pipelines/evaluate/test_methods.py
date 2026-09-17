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
