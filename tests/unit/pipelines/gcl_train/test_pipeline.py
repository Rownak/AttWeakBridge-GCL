"""End-to-end tests for the ``gcl_train`` pipeline against a fabricated ``training_data`` run."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cga.pipelines  # noqa: F401  (registers pipelines)
from cga.core.artifacts import write_latest
from cga.core.errors import ArtifactValidationError
from cga.core.io import read_json
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix, NodeUniverse
from cga.data.schemas import EDGES_SCHEMA, EXAMPLES_SCHEMA

G1_IDS = ["attack:1", "attack:2", "attack:3", "attack:4"]
G2_IDS = ["cwe:1", "cwe:2", "cwe:3", "cwe:4"]
ALL_IDS = G1_IDS + G2_IDS


def _write_embedding_store(store_dir: Path, node_ids: list[str], *, dim: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(node_ids), dim)).astype(np.float32)
    matrix = AlignedMatrix(vectors=vectors, node_ids=node_ids)
    matrix.save(store_dir)
    write_latest(store_dir.parent, store_dir.name)
    return store_dir.name


def _write_edges(ingest_dir: Path, pairs: list[tuple[str, str]]) -> None:
    rows = [
        {
            "edge_id": hashlib.sha1(f"{src}|{dst}|rel".encode()).hexdigest(),
            "src_id": src,
            "dst_id": dst,
            "graph_role": "graph_1" if src.startswith("attack") else "graph_2",
            "relation": "rel",
            "directed": False,
            "weight": 1.0,
        }
        for src, dst in pairs
    ]
    table = pa.Table.from_pylist(rows, schema=EDGES_SCHEMA)
    pq.write_table(table, ingest_dir / "edges.parquet")


def _write_examples(
    training_data_dir: Path, *, example_tuples: list[tuple[str, str, str, str, str]], split: str = "train"
) -> None:
    rows = [
        {
            "example_id": hashlib.sha1("|".join(tup).encode()).hexdigest(),
            "anchor_id": tup[0],
            "text_pos_id": tup[1],
            "text_neg_id": tup[2],
            "graph_pos_id": tup[3],
            "graph_neg_id": tup[4],
            "sampler": "rank_topk",
            "split": split,
        }
        for tup in example_tuples
    ]
    table = pa.Table.from_pylist(rows, schema=EXAMPLES_SCHEMA)
    pq.write_table(table, training_data_dir / "examples.parquet")


def _build_config(tmp_path: Path, *, epochs: int = 2) -> dict:
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
        "gcl_train": {
            "inputs": {"training_data_run": "latest", "training_data_sampler_id": "rank_topk-k30"},
            "model": {"encoder": "gcn", "hidden_dim": 8, "out_dim": 8, "gat_heads": [2, 1], "dropout": 0.0},
            "loss": {"name": "triplet", "margin": 1.0, "alpha": 0.5, "temperature": 0.1},
            "optim": {"name": "adam", "lr": 1.0e-3, "weight_decay": 0.0},
            "train": {"epochs": epochs, "checkpoint_every": 1, "log_every": 1},
            "monitor": {"enabled": False, "every": 50, "ks": [1, 5, 10]},
        },
    }


def _setup_training_data_run(
    config: dict, *, example_tuples: list[tuple[str, str, str, str, str]], node_ids: list[str] | None = None
) -> Path:
    node_ids = node_ids or ALL_IDS
    dataset_id = config["dataset"]["id"]

    ingest_dir = Path(config["paths"]["processed"]) / dataset_id / "20260101T000000Z-aaaaaaaa"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    _write_edges(ingest_dir, [("attack:1", "attack:2"), ("cwe:1", "cwe:2")])
    write_latest(ingest_dir.parent, ingest_dir.name)

    text_dir = Path(config["paths"]["artifacts"]) / "embeddings" / dataset_id / "hf_mean_pool-fake" / "run-text"
    text_run_id = _write_embedding_store(text_dir, node_ids, dim=6, seed=1)

    graph_dir = Path(config["paths"]["artifacts"]) / "embeddings" / dataset_id / "deepwalk-fake" / "run-graph"
    graph_run_id = _write_embedding_store(graph_dir, node_ids, dim=4, seed=2)

    training_data_dir = (
        Path(config["paths"]["artifacts"]) / "training_data" / dataset_id / "rank_topk-k30" / "run-td"
    )
    training_data_dir.mkdir(parents=True, exist_ok=True)

    universe = NodeUniverse(
        node_ids=node_ids,
        graph_roles=["graph_1" if nid.startswith("attack") else "graph_2" for nid in node_ids],
        split=["train"] * len(node_ids),
        held_out=[False] * len(node_ids),
    )
    universe.save(training_data_dir / "universe.parquet")

    _write_examples(training_data_dir, example_tuples=example_tuples)

    manifest = {
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
    import json

    (training_data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_latest(training_data_dir.parent, training_data_dir.name)
    return training_data_dir


def _run_gcl_train(config: dict) -> RunContext:
    from cga.pipelines.gcl_train.pipeline import build_model_id

    pipeline_cls = pipelines.get("gcl_train")
    pipeline = pipeline_cls()
    model_id = build_model_id(config["gcl_train"]["model"])
    run_root = Path(config["paths"]["models"]) / "gcl" / config["dataset"]["id"] / model_id
    ctx = RunContext.create(
        pipeline_name="gcl_train",
        pipeline_version="0.0.0-test",
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=run_root,
    )
    pipeline.run(ctx)
    return ctx


def test_gcl_train_end_to_end_produces_embeddings_and_metrics(tmp_path):
    config = _build_config(tmp_path, epochs=3)
    _setup_training_data_run(
        config,
        example_tuples=[
            ("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3"),
            ("attack:2", "cwe:2", "cwe:3", "attack:1", "attack:4"),
        ],
    )

    ctx = _run_gcl_train(config)

    matrix = AlignedMatrix.load(ctx.run_dir / "node_embeddings")
    assert matrix.vectors.shape == (len(ALL_IDS), 8)
    assert set(matrix.node_ids) == set(ALL_IDS)

    metrics_lines = (ctx.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 3

    manifest = read_json(ctx.run_dir / "manifest.json")
    assert manifest["status"] == "success"
    assert manifest["summary"]["n_nodes"] == len(ALL_IDS)
    assert manifest["summary"]["n_examples_train"] == 2


def test_load_raises_when_example_id_outside_universe(tmp_path):
    config = _build_config(tmp_path)
    training_data_dir = _setup_training_data_run(
        config,
        example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3")],
        node_ids=ALL_IDS,
    )
    # Corrupt examples.parquet with an id that is not in universe.parquet.
    _write_examples(
        training_data_dir,
        example_tuples=[("attack:1", "cwe:1", "cwe:999", "attack:2", "attack:3")],
    )

    with pytest.raises(ArtifactValidationError):
        _run_gcl_train(config)


def test_gcl_train_raises_when_no_train_split_examples(tmp_path):
    config = _build_config(tmp_path)
    training_data_dir = _setup_training_data_run(
        config,
        example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3")],
    )
    _write_examples(
        training_data_dir,
        example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3")],
        split="val",
    )

    with pytest.raises(ArtifactValidationError):
        _run_gcl_train(config)


def test_residual_adapter_run_skips_training_and_exports_z(tmp_path):
    config = _build_config(tmp_path, epochs=50)
    config["gcl_train"]["model"] = {
        "encoder": "residual_adapter",
        "adapter": "neighbor_mean",
        "lambda": 0.25,
        "directional": True,
        "model_id": "srga-nb-l0.25-dir",
    }
    _setup_training_data_run(
        config, example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3")]
    )

    ctx = _run_gcl_train(config)

    matrix = AlignedMatrix.load(ctx.run_dir / "node_embeddings")
    assert matrix.vectors.shape == (len(ALL_IDS), 6 + 12)  # text_dim 6, directional h = 2 * 6
    assert (ctx.run_dir / "metrics.jsonl").read_text(encoding="utf-8") == ""  # epochs ignored
    assert not (ctx.run_dir / "checkpoints").exists()
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert manifest["summary"]["n_trainable_params"] == 0
    assert manifest["summary"]["model_id"] == "srga-nb-l0.25-dir"


def test_gcn_run_records_nonzero_trainable_params(tmp_path):
    config = _build_config(tmp_path, epochs=1)
    _setup_training_data_run(
        config, example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3")]
    )
    ctx = _run_gcl_train(config)
    assert read_json(ctx.run_dir / "manifest.json")["summary"]["n_trainable_params"] > 0


def test_infonce_full_mlp_run_end_to_end(tmp_path):
    from cga.data.schemas import ALIGNMENTS_SCHEMA

    config = _build_config(tmp_path, epochs=5)
    config["gcl_train"]["model"] = {
        "encoder": "residual_adapter",
        "adapter": "mlp",
        "lambda": 0.25,
        "adapter_dim": 8,
        "adapter_dropout": 0.2,
        "model_id": "srga-mlp-test",
    }
    config["gcl_train"]["loss"] = {"name": "infonce_full", "temperature": 0.05, "reverse_weight": 0.5}
    td_dir = _setup_training_data_run(
        config, example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3"), ("attack:2", "cwe:2", "cwe:3", "attack:1", "attack:4")]
    )
    ingest_dir = next((Path(config["paths"]["processed"]) / "attack_cwe").glob("2026*"))
    rows = [
        {"alignment_id": f"a{i}", "node_1_id": n1, "node_2_id": n2, "relation": "equivalence", "score": 1.0,
         "split": "train", "candidates": None}
        for i, (n1, n2) in enumerate([("attack:1", "cwe:1"), ("attack:2", "cwe:2"), ("attack:1", "cwe:4")])
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=ALIGNMENTS_SCHEMA), ingest_dir / "alignments.parquet")

    ctx = _run_gcl_train(config)

    matrix = AlignedMatrix.load(ctx.run_dir / "node_embeddings")
    assert matrix.vectors.shape == (len(ALL_IDS), 6 + 8)
    assert len((ctx.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 5
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert manifest["summary"]["n_trainable_params"] > 0
