"""Ingest wiring for CSV-backed gold: source selection, sidecar, manifest summary."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import cga.adapters  # noqa: F401
import cga.pipelines  # noqa: F401
from cga.core.errors import ConfigError
from cga.core.io import read_parquet, sha256_file
from cga.data.graph_store import read_alignments
from cga.data.schemas import ALIGNMENTS_SCHEMA
from .test_pipeline import FIXTURES, _build_config, _run_ingest

HEADER = "attack_id_cited,attack_id_current,capec_id,cwe_id,edge_source\n"
FILES = {
    "stix": ["T0800,T0800,CAPEC-1,CWE-79,v12.1", "T0889,T0889,CAPEC-2,CWE-1,v12.1"],
    "union": [
        "T0800,T0800,CAPEC-1,CWE-79,capec-3.9",
        "T0800,T0800,CAPEC-9,CWE-79,v12.1",
        "T0889,T0889,CAPEC-2,CWE-1,capec-3.9",
        "T0889,T0889,CAPEC-3,CWE-999999,capec-3.9",
        "T4242,T4242,CAPEC-4,CWE-79,capec-3.9",
    ],
}


def _config(tmp_path: Path, source: str, fractions=None) -> dict:
    root = tmp_path / "raw"
    if not root.exists():
        shutil.copytree(FIXTURES, root, ignore=shutil.ignore_patterns("__pycache__"))
        (root / "capec").mkdir()
        for key, rows in FILES.items():
            (root / "capec" / f"{key}.csv").write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    cfg = _build_config(tmp_path)
    cfg["dataset"]["root"] = str(root)
    split = {"seed": 1, "group_regex": r"^(T\d+)"}
    if fractions is not None:
        split["fractions"] = fractions
    cfg["dataset"]["alignments"] = {
        "equivalence": {
            "source": source,
            "sources": {k: f"capec/{k}.csv" for k in FILES},
            "columns": {"source": "attack_id_current", "target": "cwe_id"},
            "split": split,
        }
    }
    cfg["run"]["name"] = source
    return cfg


def test_source_selection_reads_own_file(tmp_path):
    for source in ("stix", "union"):
        ctx = _run_ingest(_config(tmp_path, source), tmp_path)
        table = read_alignments(ctx.run_dir / "alignments.parquet")
        assert table.num_rows == 2
        prov = read_parquet(ctx.run_dir / "alignment_provenance.parquet").to_pylist()
        assert {p["gold_source"] for p in prov} == {source}
        assert {p["source_sha256"] for p in prov} == {sha256_file(tmp_path / "raw" / "capec" / f"{source}.csv")}
        assert [p["alignment_id"] for p in prov] == table.column("alignment_id").to_pylist()
    assert table.schema.equals(ALIGNMENTS_SCHEMA, check_metadata=False)


def test_default_split_marks_every_pair_test(tmp_path):
    ctx = _run_ingest(_config(tmp_path, "union"), tmp_path)
    table = read_alignments(ctx.run_dir / "alignments.parquet")
    assert set(table.column("split").to_pylist()) == {"test"}


def test_yaml_fractions_control_split(tmp_path):
    ctx = _run_ingest(_config(tmp_path, "union", fractions=[1.0, 0.0, 0.0]), tmp_path)
    table = read_alignments(ctx.run_dir / "alignments.parquet")
    assert set(table.column("split").to_pylist()) == {"train"}


def test_summary_reports_drops(tmp_path):
    ctx = _run_ingest(_config(tmp_path, "union"), tmp_path)
    s = ctx.manifest.summary
    assert s["gold_source"] == "union"
    assert s["dropped"] == {"source_not_in_graph": 1, "target_not_in_graph": 1, "duplicate": 1}
    assert any(o["name"] == "alignment_provenance" for o in ctx.manifest.outputs)


def test_unknown_source_raises(tmp_path):
    with pytest.raises(ConfigError):
        _run_ingest(_config(tmp_path, "nope"), tmp_path)


def test_no_alignments_writes_no_sidecar(tmp_path):
    ctx = _run_ingest(_build_config(tmp_path), tmp_path)
    assert not (ctx.run_dir / "alignment_provenance.parquet").exists()


def test_gold_ignores_non_technique_nodes_sharing_an_external_id(tmp_path):
    """A mitigation reusing a technique id must not capture the gold pair."""
    from cga.adapters.attack_cwe.adapter import AttackCweAdapter, AttackCweAdapterConfig
    from cga.data.schemas import NodeRecord

    cfg = _config(tmp_path, "stix")
    adapter = AttackCweAdapter(
        AttackCweAdapterConfig(**cfg["dataset"]["adapter_config"]),
        root=cfg["dataset"]["root"],
        text_variants={},
        alignments=cfg["dataset"]["alignments"]["equivalence"],
    )

    def node(node_id, key, role, ext, ntype):
        return NodeRecord(
            node_id=node_id, graph_key=key, graph_role=role, source_id=node_id, external_id=ext,
            name="", node_type=ntype, has_text=False, source_version="", raw_sha256="",
        )

    nodes = [
        node("attack:coa", "attack", "graph_1", "T0800", "course-of-action"),
        node("attack:tech", "attack", "graph_1", "T0800", "attack-pattern"),
        node("attack:amb1", "attack", "graph_1", "T0889", "attack-pattern"),
        node("attack:amb2", "attack", "graph_1", "T0889", "attack-pattern"),
        node("cwe:79", "cwe", "graph_2", "CWE-79", "Weakness"),
        node("cwe:1", "cwe", "graph_2", "CWE-1", "Weakness"),
    ]
    adapter.iter_nodes = lambda: iter(nodes)
    records = list(adapter.iter_alignments())
    assert [(r.node_1_id, r.node_2_id) for r in records] == [("attack:tech", "cwe:79")]
    assert adapter.alignment_report()["dropped"]["source_not_in_graph"] == 1  # ambiguous T0889
