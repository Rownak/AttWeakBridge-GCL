"""Tests for the ``ingest`` pipeline against the committed attack_cwe adapter fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

import cga.adapters  # noqa: F401  (registers adapters)
import cga.pipelines  # noqa: F401  (registers pipelines)
from cga.core.errors import ArtifactValidationError
from cga.core.io import read_json
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.graph_store import read_edges, read_nodes, read_splits, read_texts

FIXTURES = Path(__file__).parents[2] / "adapters" / "fixtures"


def _build_config(tmp_path: Path, *, cwe_edge_relations=None) -> dict:
    return {
        "project": "cga",
        "seed": 42,
        "device": "cpu",
        "paths": {
            "raw": str(FIXTURES),
            "processed": str(tmp_path / "processed"),
            "artifacts": str(tmp_path / "artifacts"),
            "models": str(tmp_path / "models"),
        },
        "logging": {"level": "INFO", "file": False},
        "run": {"name": None},
        "dataset": {
            "id": "attack_cwe",
            "adapter": "attack_cwe",
            "root": str(FIXTURES),
            "adapter_config": {
                "attack": {"file": "stix_sample.json"},
                "cwe": {
                    "file": "cwe_sample.xml",
                    "version": "4.12",
                    "edge_relations": cwe_edge_relations or ["ChildOf"],
                },
            },
            "text_variants": {
                "description": {
                    "attack": ["description"],
                    "cwe": ["name", "description", "extended_description"],
                }
            },
            "alignments": None,
        },
        "ingest": {"text_variant_selection": None},
    }


def _run_ingest(config: dict, tmp_path: Path) -> RunContext:
    pipeline_cls = pipelines.get("ingest")
    pipeline = pipeline_cls()
    ctx = RunContext.create(
        pipeline_name="ingest",
        pipeline_version="0.0.0-test",
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=tmp_path / "runs" / "attack_cwe",
    )
    pipeline.run(ctx)
    return ctx


def test_ingest_writes_five_tables_and_stats(tmp_path):
    config = _build_config(tmp_path)
    ctx = _run_ingest(config, tmp_path)

    assert ctx.manifest.status == "success"
    for filename in (
        "nodes.parquet",
        "edges.parquet",
        "texts.parquet",
        "alignments.parquet",
        "splits.parquet",
        "stats.json",
    ):
        assert (ctx.run_dir / filename).is_file()

    nodes = read_nodes(ctx.run_dir / "nodes.parquet")
    assert nodes.num_rows == 6  # 3 STIX objects + 3 CWE weaknesses

    stats = read_json(ctx.run_dir / "stats.json")
    assert stats["n_nodes"] == 6
    assert stats["n_alignments"] == 0


def test_ingest_splits_cover_every_node_exactly_once(tmp_path):
    config = _build_config(tmp_path)
    ctx = _run_ingest(config, tmp_path)

    nodes = read_nodes(ctx.run_dir / "nodes.parquet")
    splits = read_splits(ctx.run_dir / "splits.parquet")

    node_ids = set(nodes.column("node_id").to_pylist())
    split_node_ids = splits.column("node_id").to_pylist()
    assert set(split_node_ids) == node_ids
    assert len(split_node_ids) == len(set(split_node_ids))
    assert all(s in ("train", "val", "test") for s in splits.column("split").to_pylist())


def test_ingest_splits_are_reproducible_under_fixed_seed(tmp_path):
    # split.seed pins the partition explicitly; run into two distinct run_dirs
    # (different --run-name) so this doesn't depend on config_hash/timestamp
    # producing distinct run_ids within the same wall-clock second.
    config1 = _build_config(tmp_path)
    config1["ingest"]["split"] = {"fractions": [0.8, 0.1, 0.1], "seed": 123}
    ctx1 = _run_ingest(config1, tmp_path)

    config2 = _build_config(tmp_path)
    config2["ingest"]["split"] = {"fractions": [0.8, 0.1, 0.1], "seed": 123}
    config2["run"]["name"] = "second-run"
    ctx2 = _run_ingest(config2, tmp_path)

    assert ctx1.run_dir != ctx2.run_dir
    splits1 = read_splits(ctx1.run_dir / "splits.parquet")
    splits2 = read_splits(ctx2.run_dir / "splits.parquet")
    map1 = dict(zip(splits1.column("node_id").to_pylist(), splits1.column("split").to_pylist()))
    map2 = dict(zip(splits2.column("node_id").to_pylist(), splits2.column("split").to_pylist()))
    assert map1 == map2


def test_ingest_split_fractions_configurable(tmp_path):
    config = _build_config(tmp_path)
    config["ingest"]["split"] = {"fractions": [0.5, 0.25, 0.25], "seed": 7}
    ctx = _run_ingest(config, tmp_path)

    stats = read_json(ctx.run_dir / "stats.json")
    assert "n_nodes_by_split_by_graph_role" in stats
    for role_counts in stats["n_nodes_by_split_by_graph_role"].values():
        assert set(role_counts.keys()) <= {"train", "val", "test"}


def test_ingest_writes_latest_file(tmp_path):
    config = _build_config(tmp_path)
    ctx = _run_ingest(config, tmp_path)

    lineage_dir = Path(config["paths"]["processed"]) / "attack_cwe"
    latest_run_id = (lineage_dir / "LATEST").read_text(encoding="utf-8").strip()
    assert latest_run_id == ctx.run_id


def test_ingest_dedupes_repeated_edge_ids(tmp_path):
    # cwe_sample.xml's CWE 79 declares both a ChildOf (kept) and a PeerOf
    # relation; requesting both relations should not produce duplicate rows
    # for the same (src, dst, relation) triple beyond what the source has.
    config = _build_config(tmp_path, cwe_edge_relations=["ChildOf", "PeerOf"])
    ctx = _run_ingest(config, tmp_path)
    edges = read_edges(ctx.run_dir / "edges.parquet")
    edge_ids = edges.column("edge_id").to_pylist()
    assert len(edge_ids) == len(set(edge_ids))


def test_ingest_cross_graph_edge_from_adapter_raises(tmp_path, monkeypatch):
    """A misbehaving adapter yielding a cross-graph edge must abort the run, not silently pass."""
    import cga.adapters.attack_cwe.stix_parser as stix_parser

    original = stix_parser.iter_stix_edges

    def _poisoned(result, *, graph_key, graph_role):
        from cga.data.ids import make_node_id
        from cga.data.schemas import EdgeRecord

        yield EdgeRecord(
            edge_id="poisoned-cross-graph-edge",
            src_id=make_node_id("attack", "attack-pattern--aaaa"),
            dst_id=make_node_id("cwe", "79"),
            graph_role="graph_1",
            relation="uses",
            directed=True,
        )
        yield from original(result, graph_key=graph_key, graph_role=graph_role)

    monkeypatch.setattr(stix_parser, "iter_stix_edges", _poisoned)
    # adapter.py imported the name directly, so patch it there too.
    import cga.adapters.attack_cwe.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "iter_stix_edges", _poisoned)

    config = _build_config(tmp_path)
    with pytest.raises(ArtifactValidationError):
        _run_ingest(config, tmp_path)


def test_ingest_texts_reference_only_known_nodes(tmp_path):
    config = _build_config(tmp_path)
    ctx = _run_ingest(config, tmp_path)
    texts = read_texts(ctx.run_dir / "texts.parquet")
    nodes = read_nodes(ctx.run_dir / "nodes.parquet")
    node_ids = set(nodes.column("node_id").to_pylist())
    assert set(texts.column("node_id").to_pylist()) <= node_ids
