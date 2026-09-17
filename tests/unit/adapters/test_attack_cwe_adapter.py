"""Tests for AttackCweAdapter end-to-end against the committed fixtures."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from cga.adapters.attack_cwe.adapter import AttackCweAdapter, AttackCweAdapterConfig
from cga.core.errors import ArtifactValidationError, MissingInputError
from cga.data.graph_store import write_edges, write_nodes
from cga.data.schemas import EDGES_SCHEMA, NODES_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"

TEXT_VARIANTS = {
    "description": {
        "attack": ["description"],
        "cwe": ["name", "description", "extended_description"],
    },
}


def _build_adapter() -> AttackCweAdapter:
    config = AttackCweAdapterConfig(
        attack={"file": "stix_sample.json"},
        cwe={"file": "cwe_sample.xml", "version": "4.12", "edge_relations": ["ChildOf"]},
    )
    return AttackCweAdapter(config, root=FIXTURES, text_variants=TEXT_VARIANTS)


def test_validate_sources_passes_when_files_exist():
    adapter = _build_adapter()
    adapter.validate_sources()  # does not raise


def test_validate_sources_raises_for_missing_file(tmp_path):
    config = AttackCweAdapterConfig(
        attack={"file": "missing.json"},
        cwe={"file": "cwe_sample.xml", "version": "4.12"},
    )
    adapter = AttackCweAdapter(config, root=tmp_path, text_variants=TEXT_VARIANTS)
    with pytest.raises(MissingInputError):
        adapter.validate_sources()


def test_iter_nodes_combines_both_graphs():
    adapter = _build_adapter()
    nodes = list(adapter.iter_nodes())
    node_ids = {n.node_id for n in nodes}
    assert "attack:attack-pattern--aaaa" in node_ids
    assert "cwe:79" in node_ids
    roles = {n.node_id: n.graph_role for n in nodes}
    assert roles["attack:attack-pattern--aaaa"] == "graph_1"
    assert roles["cwe:79"] == "graph_2"


def test_iter_edges_never_crosses_graph_role():
    adapter = _build_adapter()
    edges = list(adapter.iter_edges())
    assert len(edges) > 0
    for edge in edges:
        src_graph = edge.src_id.split(":", 1)[0]
        dst_graph = edge.dst_id.split(":", 1)[0]
        assert src_graph == dst_graph


def test_iter_alignments_is_empty():
    adapter = _build_adapter()
    assert list(adapter.iter_alignments()) == []


def test_source_fingerprint_hashes_both_files():
    adapter = _build_adapter()
    fingerprint = adapter.source_fingerprint()
    assert set(fingerprint) == {"stix_sample.json", "cwe_sample.xml"}
    assert all(len(v) == 64 for v in fingerprint.values())


def _nodes_table(adapter: AttackCweAdapter) -> pa.Table:
    rows = [n.model_dump() for n in adapter.iter_nodes()]
    return pa.Table.from_pylist(rows, schema=NODES_SCHEMA)


def test_duplicate_node_id_rejected_by_graph_store(tmp_path):
    adapter = _build_adapter()
    rows = [n.model_dump() for n in adapter.iter_nodes()]
    rows.append(rows[0])  # duplicate
    table = pa.Table.from_pylist(rows, schema=NODES_SCHEMA)
    with pytest.raises(ArtifactValidationError):
        write_nodes(table, tmp_path / "nodes.parquet")


def test_cross_graph_edge_rejected_by_graph_store(tmp_path):
    adapter = _build_adapter()
    nodes_table = _nodes_table(adapter)
    write_nodes(nodes_table, tmp_path / "nodes.parquet")

    bad_edge = {
        "edge_id": "bad-edge",
        "src_id": "attack:attack-pattern--aaaa",
        "dst_id": "cwe:79",
        "graph_role": "graph_1",
        "relation": "uses",
        "directed": True,
        "weight": 1.0,
    }
    table = pa.Table.from_pylist([bad_edge], schema=EDGES_SCHEMA)
    with pytest.raises(ArtifactValidationError):
        write_edges(table, tmp_path / "edges.parquet", nodes=nodes_table)
