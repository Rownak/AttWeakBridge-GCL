"""End-to-end tests for the ``training_data`` pipeline, including leakage prevention."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import cga.pipelines  # noqa: F401  (registers pipelines)
from cga.core.artifacts import write_latest
from cga.core.io import read_json
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix
from cga.data.aligned import NodeUniverse
from cga.data.example_store import read_examples
from cga.data.schemas import ALIGNMENTS_SCHEMA, NODES_SCHEMA, SPLITS_SCHEMA
from cga.pipelines.training_data.pipeline import build_sampler_id

G1_IDS = ["ncit:1", "ncit:2", "ncit:3", "ncit:4"]
G2_IDS = ["doid:1", "doid:2", "doid:3", "doid:4"]
ALL_IDS = G1_IDS + G2_IDS


def _write_ingest_run(
    ingest_dir: Path, *, alignment_rows: list[dict], node_splits: dict[str, str] | None = None
) -> None:
    """Write a fake ingest run.

    ``node_splits`` overrides the per-node ``splits.parquet`` split (defaults
    to ``"train"`` for every node) — independent of ``alignment_rows``'
    ``split`` column, since Phase 1.6 makes these two splits genuinely
    independent sources that ``build_universe`` unions.
    """
    ingest_dir.mkdir(parents=True, exist_ok=True)
    node_splits = node_splits or {}
    nodes = pa.Table.from_pylist(
        [
            {
                "node_id": nid,
                "graph_key": nid.split(":")[0],
                "graph_role": "graph_1" if nid.startswith("ncit") else "graph_2",
                "source_id": nid.split(":")[1],
                "external_id": nid,
                "name": nid,
                "node_type": None,
                "metadata": "{}",
                "has_text": True,
                "source_version": "test",
                "raw_sha256": "0" * 64,
            }
            for nid in ALL_IDS
        ],
        schema=NODES_SCHEMA,
    )
    pq.write_table(nodes, ingest_dir / "nodes.parquet")

    alignments = pa.Table.from_pylist(
        [
            {
                "alignment_id": hashlib.sha1(f"{r['node_1_id']}|{r['node_2_id']}".encode()).hexdigest(),
                "node_1_id": r["node_1_id"],
                "node_2_id": r["node_2_id"],
                "relation": "equivalence",
                "score": 1.0,
                "split": r["split"],
                "candidates": None,
            }
            for r in alignment_rows
        ],
        schema=ALIGNMENTS_SCHEMA,
    )
    pq.write_table(alignments, ingest_dir / "alignments.parquet")

    splits = pa.Table.from_pylist(
        [
            {
                "node_id": nid,
                "graph_role": "graph_1" if nid.startswith("ncit") else "graph_2",
                "split": node_splits.get(nid, "train"),
            }
            for nid in ALL_IDS
        ],
        schema=SPLITS_SCHEMA,
    )
    pq.write_table(splits, ingest_dir / "splits.parquet")

    write_latest(ingest_dir.parent, ingest_dir.name)


def _write_embedding_store(store_dir: Path, node_ids: list[str], *, dim: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(node_ids), dim)).astype(np.float32)
    matrix = AlignedMatrix(vectors=vectors, node_ids=node_ids)
    matrix.save(store_dir)
    write_latest(store_dir.parent, store_dir.name)


def _build_config(tmp_path: Path, *, sampler_name: str = "rank_topk", sampler_extra: dict | None = None) -> dict:
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
        "dataset": {"id": "ncit_doid"},
        "training_data": {
            "inputs": {
                "ingest_run": "latest",
                "text_embeddings": {"embedder_id": "hf_mean_pool-fake", "run": "latest"},
                "graph_embeddings": {"embedder_id": "deepwalk-fake", "run": "latest"},
            },
            "universe_policy": "intersection",
            "sampler": {"name": sampler_name, "k": 2, **(sampler_extra or {})},
        },
    }


def _run_training_data(config: dict, tmp_path: Path) -> RunContext:
    from cga.pipelines.training_data.pipeline import build_sampler_id

    pipeline_cls = pipelines.get("training_data")
    pipeline = pipeline_cls()
    sampler_id = build_sampler_id(config["training_data"]["sampler"])
    run_root = Path(config["paths"]["artifacts"]) / "training_data" / config["dataset"]["id"] / sampler_id
    ctx = RunContext.create(
        pipeline_name="training_data",
        pipeline_version="0.0.0-test",
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=run_root,
    )
    pipeline.run(ctx)
    return ctx


def _setup_stores(
    config: dict,
    tmp_path: Path,
    *,
    alignment_rows: list[dict],
    node_ids=None,
    node_splits: dict[str, str] | None = None,
):
    node_ids = node_ids or ALL_IDS
    ingest_dir = Path(config["paths"]["processed"]) / "ncit_doid" / "20260101T000000Z-aaaaaaaa"
    _write_ingest_run(ingest_dir, alignment_rows=alignment_rows, node_splits=node_splits)

    text_dir = (
        Path(config["paths"]["artifacts"]) / "embeddings" / "ncit_doid" / "hf_mean_pool-fake" / "20260101T000000Z-bbbbbbbb"
    )
    _write_embedding_store(text_dir, node_ids, seed=1)

    graph_dir = (
        Path(config["paths"]["artifacts"]) / "embeddings" / "ncit_doid" / "deepwalk-fake" / "20260101T000000Z-cccccccc"
    )
    _write_embedding_store(graph_dir, node_ids, seed=2)


def test_build_universe_flags_held_out_from_test_alignments(tmp_path):
    config = _build_config(tmp_path)
    _setup_stores(
        config,
        tmp_path,
        alignment_rows=[
            {"node_1_id": "ncit:1", "node_2_id": "doid:1", "split": "test"},
            {"node_1_id": "ncit:2", "node_2_id": "doid:2", "split": "train"},
        ],
    )

    ctx = _run_training_data(config, tmp_path)

    universe = NodeUniverse.from_parquet(ctx.run_dir / "universe.parquet")
    held_out_by_node = dict(zip(universe.node_ids, universe.held_out))
    assert held_out_by_node["ncit:1"] is True
    assert held_out_by_node["doid:1"] is True
    assert held_out_by_node["ncit:2"] is False
    assert held_out_by_node["doid:2"] is False


def test_build_universe_flags_held_out_from_node_split(tmp_path):
    # No alignments at all (attack_cwe-shaped) -- held_out must still follow
    # the ingest run's own splits.parquet val/test assignment.
    config = _build_config(tmp_path)
    _setup_stores(
        config,
        tmp_path,
        alignment_rows=[],
        node_splits={"ncit:1": "val", "doid:1": "test"},
    )

    ctx = _run_training_data(config, tmp_path)

    universe = NodeUniverse.from_parquet(ctx.run_dir / "universe.parquet")
    held_out_by_node = dict(zip(universe.node_ids, universe.held_out))
    split_by_node = dict(zip(universe.node_ids, universe.split))
    assert held_out_by_node["ncit:1"] is True
    assert split_by_node["ncit:1"] == "val"
    assert held_out_by_node["doid:1"] is True
    assert split_by_node["doid:1"] == "test"
    assert held_out_by_node["ncit:2"] is False
    assert split_by_node["ncit:2"] == "train"


def test_build_universe_unions_node_split_and_alignment_held_out(tmp_path):
    # ncit:2/doid:2 is a train-split alignment pair, but ncit:2 is in our own
    # val split -- it must still be held out (union, not either source alone).
    config = _build_config(tmp_path)
    _setup_stores(
        config,
        tmp_path,
        alignment_rows=[
            {"node_1_id": "ncit:2", "node_2_id": "doid:2", "split": "train"},
        ],
        node_splits={"ncit:2": "val"},
    )

    ctx = _run_training_data(config, tmp_path)

    universe = NodeUniverse.from_parquet(ctx.run_dir / "universe.parquet")
    held_out_by_node = dict(zip(universe.node_ids, universe.held_out))
    assert held_out_by_node["ncit:2"] is True  # held out via node split, despite train-split alignment
    assert held_out_by_node["doid:2"] is False  # doid:2 itself is train-split both ways


def test_build_universe_excludes_nodes_missing_from_either_store(tmp_path):
    config = _build_config(tmp_path)
    # graph store is missing doid:4 -> intersection policy should exclude it.
    _setup_stores(config, tmp_path, alignment_rows=[], node_ids=G1_IDS + G2_IDS)
    graph_dir = (
        Path(config["paths"]["artifacts"]) / "embeddings" / "ncit_doid" / "deepwalk-fake" / "20260101T000000Z-cccccccc"
    )
    # overwrite graph store without doid:4
    _write_embedding_store(graph_dir, G1_IDS + G2_IDS[:-1], seed=2)

    ctx = _run_training_data(config, tmp_path)

    universe = NodeUniverse.from_parquet(ctx.run_dir / "universe.parquet")
    assert "doid:4" not in universe.node_ids


def test_rank_topk_examples_never_reference_held_out_test_nodes(tmp_path):
    config = _build_config(tmp_path, sampler_name="rank_topk")
    _setup_stores(
        config,
        tmp_path,
        alignment_rows=[
            {"node_1_id": "ncit:1", "node_2_id": "doid:1", "split": "test"},
        ],
    )

    ctx = _run_training_data(config, tmp_path)

    examples = read_examples(ctx.run_dir / "examples.parquet")
    id_columns = ["anchor_id", "text_pos_id", "text_neg_id", "graph_pos_id", "graph_neg_id"]
    referenced = set()
    for col in id_columns:
        referenced.update(examples.column(col).to_pylist())

    assert "ncit:1" not in referenced
    assert "doid:1" not in referenced


def test_supervised_sampler_noops_when_alignments_empty(tmp_path):
    config = _build_config(tmp_path, sampler_name="supervised")
    _setup_stores(config, tmp_path, alignment_rows=[])

    ctx = _run_training_data(config, tmp_path)

    examples = read_examples(ctx.run_dir / "examples.parquet")
    assert examples.num_rows == 0

    stats = read_json(ctx.run_dir / "stats.json")
    assert stats["n_examples"] == 0


def test_supervised_split_keeps_test_alignment_out_of_train_examples(tmp_path):
    config = _build_config(tmp_path, sampler_name="supervised")
    _setup_stores(
        config,
        tmp_path,
        alignment_rows=[
            {"node_1_id": "ncit:1", "node_2_id": "doid:1", "split": "test"},
            {"node_1_id": "ncit:2", "node_2_id": "doid:2", "split": "train"},
        ],
    )

    ctx = _run_training_data(config, tmp_path)

    examples = read_examples(ctx.run_dir / "examples.parquet")
    anchors = examples.column("anchor_id").to_pylist()
    splits = examples.column("split").to_pylist()
    assert "ncit:1" not in anchors  # test-split alignment never became an example at all
    # ncit:2 has 2 eligible text negatives (doid:3, doid:4) and 2 other graph_1 nodes.
    assert anchors == ["ncit:2", "ncit:2"]
    assert set(examples.column("text_pos_id").to_pylist()) == {"doid:2"}
    # ncit:2's own node split is "train" (default), so its examples inherit "train".
    assert splits == ["train", "train"]


def test_supervised_examples_avoid_test_endpoints_and_gold_negatives(tmp_path):
    config = _build_config(tmp_path, sampler_name="supervised", sampler_extra={"negatives_per_positive": 5})
    alignment_rows = [
        {"node_1_id": "ncit:1", "node_2_id": "doid:1", "split": "test"},
        {"node_1_id": "ncit:2", "node_2_id": "doid:2", "split": "train"},
        {"node_1_id": "ncit:2", "node_2_id": "doid:3", "split": "train"},
        # Shares a test endpoint (doid:1): must be skipped.
        {"node_1_id": "ncit:3", "node_2_id": "doid:1", "split": "train"},
    ]
    _setup_stores(config, tmp_path, alignment_rows=alignment_rows)

    ctx = _run_training_data(config, tmp_path)

    examples = read_examples(ctx.run_dir / "examples.parquet")
    test_endpoints = {"ncit:1", "doid:1"}
    for column in ("anchor_id", "text_pos_id", "text_neg_id", "graph_pos_id", "graph_neg_id"):
        assert not test_endpoints & set(examples.column(column).to_pylist())
    gold = {(r["node_1_id"], r["node_2_id"]) for r in alignment_rows}
    for anchor_id, text_neg_id in zip(
        examples.column("anchor_id").to_pylist(), examples.column("text_neg_id").to_pylist()
    ):
        assert text_neg_id.startswith("doid:")
        assert (anchor_id, text_neg_id) not in gold
    # ncit:2's only non-gold, non-held-out DOID node is doid:4: one row per positive.
    assert sorted(examples.column("text_pos_id").to_pylist()) == ["doid:2", "doid:3"]
    assert set(examples.column("text_neg_id").to_pylist()) == {"doid:4"}

    stats = read_json(ctx.run_dir / "stats.json")
    assert stats["n_gold_train_pairs"] == 3
    assert stats["n_gold_train_pairs_used"] == 2
    assert stats["n_gold_train_pairs_skipped"] == 1


def test_supervised_sampler_excludes_pairs_with_a_val_split_endpoint(tmp_path):
    # ncit:2 is a train-split alignment pair, but ncit:2 itself sits in our
    # own val node split -- held_out (the union rule) excludes it, so the
    # supervised sampler must not use it as an anchor even though the
    # alignment's own split says "train".
    config = _build_config(tmp_path, sampler_name="supervised")
    _setup_stores(
        config,
        tmp_path,
        alignment_rows=[
            {"node_1_id": "ncit:2", "node_2_id": "doid:2", "split": "train"},
        ],
        node_splits={"ncit:2": "val"},
    )

    ctx = _run_training_data(config, tmp_path)

    examples = read_examples(ctx.run_dir / "examples.parquet")
    assert examples.num_rows == 0


def test_build_sampler_id_rank_topk_includes_k():
    assert build_sampler_id({"name": "rank_topk", "k": 30}) == "rank_topk-k30"


def test_build_sampler_id_percentile_includes_both_pcts():
    assert (
        build_sampler_id({"name": "percentile", "top_pct": 0.04, "bottom_pct": 0.08})
        == "percentile-top_pct0.04-bottom_pct0.08"
    )


def test_build_sampler_id_supervised_includes_negatives_per_positive():
    assert build_sampler_id({"name": "supervised", "negatives_per_positive": 5}) == "supervised-negatives_per_positive5"


def test_build_sampler_id_supervised_random_negatives_gets_its_own_id():
    cfg = {"name": "supervised", "negatives_per_positive": 5, "text_negatives": "random"}
    assert build_sampler_id(cfg) == "supervised-negatives_per_positive5-text_negativesrandom"
    # Default (bottomk) keeps the original id when the key is absent.
    assert build_sampler_id({"name": "supervised", "negatives_per_positive": 5}) == "supervised-negatives_per_positive5"


def test_build_sampler_id_respects_explicit_override():
    assert build_sampler_id({"name": "rank_topk", "k": 30, "sampler_id": "custom"}) == "custom"


def test_training_data_pipeline_writes_latest(tmp_path):
    config = _build_config(tmp_path)
    _setup_stores(config, tmp_path, alignment_rows=[])

    ctx = _run_training_data(config, tmp_path)

    lineage_dir = Path(config["paths"]["artifacts"]) / "training_data" / "ncit_doid" / "rank_topk-k2"
    latest_run_id = (lineage_dir / "LATEST").read_text(encoding="utf-8").strip()
    assert latest_run_id == ctx.run_id


def test_training_data_pipeline_partitions_latest_by_sampler_id(tmp_path):
    # Two different k values for the same sampler must keep separate LATEST
    # pointers -- otherwise a sweep over rank_topk.k would have its later
    # points silently clobber earlier ones.
    config_k2 = _build_config(tmp_path, sampler_extra={"k": 2})
    _setup_stores(config_k2, tmp_path, alignment_rows=[])
    ctx_k2 = _run_training_data(config_k2, tmp_path)

    config_k3 = _build_config(tmp_path, sampler_extra={"k": 3})
    ctx_k3 = _run_training_data(config_k3, tmp_path)

    lineage_root = Path(config_k2["paths"]["artifacts"]) / "training_data" / "ncit_doid"
    assert (lineage_root / "rank_topk-k2" / "LATEST").read_text(encoding="utf-8").strip() == ctx_k2.run_id
    assert (lineage_root / "rank_topk-k3" / "LATEST").read_text(encoding="utf-8").strip() == ctx_k3.run_id


def test_val_alignment_endpoints_are_held_out_and_absent_from_examples(tmp_path):
    config = _build_config(tmp_path, sampler_name="supervised")
    _setup_stores(
        config,
        tmp_path,
        alignment_rows=[
            {"node_1_id": "ncit:1", "node_2_id": "doid:1", "split": "val"},
            {"node_1_id": "ncit:2", "node_2_id": "doid:2", "split": "train"},
        ],
    )

    ctx = _run_training_data(config, tmp_path)

    universe = NodeUniverse.from_parquet(ctx.run_dir / "universe.parquet")
    held_out_by_node = dict(zip(universe.node_ids, universe.held_out))
    assert held_out_by_node["ncit:1"] is True
    assert held_out_by_node["doid:1"] is True
    assert held_out_by_node["ncit:2"] is False
    examples = read_examples(ctx.run_dir / "examples.parquet")
    used = set(examples.column("anchor_id").to_pylist()) | set(examples.column("text_pos_id").to_pylist())
    used |= set(examples.column("text_neg_id").to_pylist())
    assert {"ncit:1", "doid:1"}.isdisjoint(used)
    assert {"ncit:2", "doid:2"} <= used
