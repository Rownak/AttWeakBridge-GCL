"""End-to-end tests for the ``evaluate`` pipeline against a fabricated ``gcl_train`` run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cga.pipelines  # noqa: F401  (registers pipelines)
from cga.core.artifacts import write_latest
from cga.core.io import read_json
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix
from cga.data.schemas import NODES_SCHEMA

G1_IDS = ["attack:1", "attack:2", "attack:3"]
G2_IDS = ["cwe:1", "cwe:2", "cwe:3"]
ALL_IDS = G1_IDS + G2_IDS


def _write_embedding_store(store_dir: Path, node_ids: list[str], *, dim: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(node_ids), dim)).astype(np.float32)
    AlignedMatrix(vectors=vectors, node_ids=node_ids).save(store_dir)
    write_latest(store_dir.parent, store_dir.name)
    return store_dir.name


def _write_nodes(ingest_dir: Path, node_ids: list[str]) -> None:
    rows = [
        {
            "node_id": nid,
            "graph_key": nid.split(":")[0],
            "graph_role": "graph_1" if nid.startswith("attack") else "graph_2",
            "source_id": nid.split(":")[1],
            "external_id": nid,
            "name": nid,
            "node_type": None,
            "metadata": "{}",
            "has_text": True,
            "source_version": "test",
            "raw_sha256": "0" * 64,
        }
        for nid in node_ids
    ]
    table = pa.Table.from_pylist(rows, schema=NODES_SCHEMA)
    pq.write_table(table, ingest_dir / "nodes.parquet")


def _build_config(tmp_path: Path, *, ks: list[int] | None = None, visualize: bool = False) -> dict:
    return {
        "project": "cga",
        "seed": 42,
        "device": "cpu",
        "paths": {
            "raw": str(tmp_path / "raw"),
            "processed": str(tmp_path / "processed"),
            "artifacts": str(tmp_path / "artifacts"),
            "models": str(tmp_path / "models"),
        },
        "logging": {"level": "INFO", "file": False},
        "run": {"name": None},
        "dataset": {"id": "attack_cwe"},
        "evaluate": {
            "inputs": {"gcl_train_run": "latest"},
            "neighborhood": {"ks": ks or [1, 2], "views": ["text", "structural"]},
            "visualize": {"enabled": visualize, "perplexity": 5.0},
        },
    }


def _setup_gcl_train_run(config: dict, *, node_ids: list[str] | None = None) -> Path:
    node_ids = node_ids or ALL_IDS
    dataset_id = config["dataset"]["id"]

    ingest_dir = Path(config["paths"]["processed"]) / dataset_id / "20260101T000000Z-aaaaaaaa"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    _write_nodes(ingest_dir, node_ids)
    write_latest(ingest_dir.parent, ingest_dir.name)

    text_dir = Path(config["paths"]["artifacts"]) / "embeddings" / dataset_id / "hf_mean_pool-fake" / "run-text"
    text_run_id = _write_embedding_store(text_dir, node_ids, dim=6, seed=1)

    graph_dir = Path(config["paths"]["artifacts"]) / "embeddings" / dataset_id / "deepwalk-fake" / "run-graph"
    graph_run_id = _write_embedding_store(graph_dir, node_ids, dim=4, seed=2)

    training_data_dir = Path(config["paths"]["artifacts"]) / "training_data" / dataset_id / "run-td"
    training_data_dir.mkdir(parents=True, exist_ok=True)
    td_manifest = {
        "manifest_version": 1,
        "pipeline": "training_data",
        "run_id": "run-td",
        "status": "success",
        "inputs": [
            {"name": "ingest", "pipeline": "ingest", "run_id": ingest_dir.name, "path": str(ingest_dir)},
            {"name": "text_embeddings", "pipeline": "hf_mean_pool-fake", "run_id": text_run_id, "path": str(text_dir)},
            {"name": "graph_embeddings", "pipeline": "deepwalk-fake", "run_id": graph_run_id, "path": str(graph_dir)},
        ],
        "outputs": [],
        "summary": {},
    }
    (training_data_dir / "manifest.json").write_text(json.dumps(td_manifest), encoding="utf-8")
    write_latest(training_data_dir.parent, training_data_dir.name)

    gcl_dir = Path(config["paths"]["models"]) / "gcl" / dataset_id / "run-gcl"
    gcl_dir.mkdir(parents=True, exist_ok=True)
    learned_run_id = _write_embedding_store(gcl_dir / "node_embeddings", node_ids, dim=8, seed=3)
    gcl_manifest = {
        "manifest_version": 1,
        "pipeline": "gcl_train",
        "run_id": "run-gcl",
        "status": "success",
        "inputs": [
            {
                "name": "training_data",
                "pipeline": "training_data",
                "run_id": training_data_dir.name,
                "path": str(training_data_dir),
            },
        ],
        "outputs": [],
        "summary": {},
    }
    (gcl_dir / "manifest.json").write_text(json.dumps(gcl_manifest), encoding="utf-8")
    write_latest(gcl_dir.parent, gcl_dir.name)
    assert learned_run_id  # unused beyond documenting intent
    return gcl_dir


def _run_evaluate(config: dict) -> RunContext:
    pipeline_cls = pipelines.get("evaluate")
    pipeline = pipeline_cls()
    run_root = Path(config["paths"]["artifacts"]) / "evaluations" / config["dataset"]["id"]
    ctx = RunContext.create(
        pipeline_name="evaluate",
        pipeline_version="0.0.0-test",
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=run_root,
    )
    pipeline.run(ctx)
    return ctx


def test_evaluate_end_to_end_writes_metrics_for_both_views(tmp_path):
    config = _build_config(tmp_path)
    _setup_gcl_train_run(config)

    ctx = _run_evaluate(config)

    metrics = read_json(ctx.run_dir / "metrics.json")
    assert set(metrics["neighborhood"].keys()) == {"text", "structural"}
    for view in ("text", "structural"):
        for role in ("graph_1", "graph_2"):
            per_k = metrics["neighborhood"][view][role]
            assert set(per_k.keys()) == {"1", "2"}
            for k_metrics in per_k.values():
                assert k_metrics["n_queries"] == 3
                assert 0.0 <= k_metrics["precision_at_k"] <= 1.0
                assert 0.0 <= k_metrics["hit_at_k"] <= 1.0
                assert 0.0 <= k_metrics["mrr"] <= 1.0

    manifest = read_json(ctx.run_dir / "manifest.json")
    assert manifest["status"] == "success"
    assert manifest["summary"]["gcl_train_run_id"] == "run-gcl"
    assert not (ctx.run_dir / "plots").exists()


def test_evaluate_visualize_enabled_writes_plot(tmp_path):
    config = _build_config(tmp_path, visualize=True)
    _setup_gcl_train_run(config)

    ctx = _run_evaluate(config)

    plot_path = ctx.run_dir / "plots" / "learned_embeddings_tsne.png"
    assert plot_path.is_file()
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert any(o["name"] == "tsne_plot" for o in manifest["outputs"])


def test_evaluate_unknown_view_raises(tmp_path):
    from cga.core.errors import ConfigError

    config = _build_config(tmp_path)
    config["evaluate"]["neighborhood"]["views"] = ["text", "bogus"]
    _setup_gcl_train_run(config)

    with pytest.raises(ConfigError):
        _run_evaluate(config)


def test_evaluate_is_deterministic_across_runs(tmp_path):
    config_a = _build_config(tmp_path / "a")
    (tmp_path / "a").mkdir()
    _setup_gcl_train_run(config_a)
    ctx_a = _run_evaluate(config_a)
    metrics_a = read_json(ctx_a.run_dir / "metrics.json")

    config_b = _build_config(tmp_path / "b")
    (tmp_path / "b").mkdir()
    _setup_gcl_train_run(config_b)
    ctx_b = _run_evaluate(config_b)
    metrics_b = read_json(ctx_b.run_dir / "metrics.json")

    assert metrics_a == metrics_b
