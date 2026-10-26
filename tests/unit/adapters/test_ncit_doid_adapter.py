"""Tests for NcitDoidAdapter end-to-end against the committed fixtures."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from cga.adapters.ncit_doid.adapter import NcitDoidAdapter, NcitDoidAdapterConfig
from cga.core.errors import ArtifactValidationError, MissingInputError
from cga.data.graph_store import write_edges, write_nodes
from cga.data.schemas import EDGES_SCHEMA, NODES_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"

TEXT_VARIANTS = {
    "label": ["label"],
    "label_synonyms_definition": ["label", "synonym", "definition"],
}

ALIGNMENTS = {
    "train": "refs_equiv_train.tsv",
    "test": "refs_equiv_test.tsv",
    "candidates": "refs_equiv_test_cands.tsv",
}


def _build_adapter(*, with_alignments: bool = True) -> NcitDoidAdapter:
    config = NcitDoidAdapterConfig(
        ncit={
            "file": "ncit_sample.owl",
            "predicate_map": {"label": ["rdfs:label"], "synonym": ["P90"], "definition": ["P97", "P325"]},
        },
        doid={
            "file": "doid_sample.owl",
            "predicate_map": {
                "label": ["rdfs:label"],
                "synonym": ["oboInOwl:hasExactSynonym"],
                "definition": ["obo:IAO_0000115"],
            },
        },
    )
    return NcitDoidAdapter(
        config,
        root=FIXTURES,
        text_variants=TEXT_VARIANTS,
        alignments=ALIGNMENTS if with_alignments else None,
    )


def test_validate_sources_passes_when_files_exist():
    adapter = _build_adapter()
    adapter.validate_sources()  # does not raise


def test_validate_sources_raises_for_missing_ontology_file(tmp_path):
    config = NcitDoidAdapterConfig(
        ncit={"file": "missing.owl", "predicate_map": {"label": ["rdfs:label"]}},
        doid={"file": "doid_sample.owl", "predicate_map": {"label": ["rdfs:label"]}},
    )
    adapter = NcitDoidAdapter(config, root=tmp_path, text_variants=TEXT_VARIANTS)
    with pytest.raises(MissingInputError):
        adapter.validate_sources()


def test_iter_nodes_combines_both_ontologies():
    adapter = _build_adapter()
    nodes = list(adapter.iter_nodes())
    node_ids = {n.node_id for n in nodes}
    assert "ncit:C3224" in node_ids
    assert "doid:DOID_4362" in node_ids
    roles = {n.node_id: n.graph_role for n in nodes}
    assert roles["ncit:C3224"] == "graph_1"
    assert roles["doid:DOID_4362"] == "graph_2"


def test_iter_edges_never_crosses_graph_role():
    adapter = _build_adapter()
    edges = list(adapter.iter_edges())
    assert len(edges) > 0
    for edge in edges:
        src_graph = edge.src_id.split(":", 1)[0]
        dst_graph = edge.dst_id.split(":", 1)[0]
        assert src_graph == dst_graph


def test_iter_alignments_populated_with_train_test_splits():
    adapter = _build_adapter()
    alignments = list(adapter.iter_alignments())
    splits = {a.split for a in alignments}
    assert splits == {"train", "test"}
    for a in alignments:
        assert a.node_1_id.startswith("ncit:")
        assert a.node_2_id.startswith("doid:")


def test_iter_alignments_empty_without_config():
    adapter = _build_adapter(with_alignments=False)
    assert list(adapter.iter_alignments()) == []


def test_source_fingerprint_includes_ontologies_and_alignment_files():
    adapter = _build_adapter()
    fingerprint = adapter.source_fingerprint()
    assert set(fingerprint) == {
        "ncit_sample.owl",
        "doid_sample.owl",
        "refs_equiv_train.tsv",
        "refs_equiv_test.tsv",
        "refs_equiv_test_cands.tsv",
    }


def _nodes_table(adapter: NcitDoidAdapter) -> pa.Table:
    rows = [n.model_dump() for n in adapter.iter_nodes()]
    return pa.Table.from_pylist(rows, schema=NODES_SCHEMA)


def test_duplicate_node_id_rejected_by_graph_store(tmp_path):
    adapter = _build_adapter()
    rows = [n.model_dump() for n in adapter.iter_nodes()]
    rows.append(rows[0])
    table = pa.Table.from_pylist(rows, schema=NODES_SCHEMA)
    with pytest.raises(ArtifactValidationError):
        write_nodes(table, tmp_path / "nodes.parquet")


def test_cross_graph_edge_rejected_by_graph_store(tmp_path):
    adapter = _build_adapter()
    nodes_table = _nodes_table(adapter)
    write_nodes(nodes_table, tmp_path / "nodes.parquet")

    bad_edge = {
        "edge_id": "bad-edge",
        "src_id": "ncit:C3224",
        "dst_id": "doid:DOID_4362",
        "graph_role": "graph_1",
        "relation": "subClassOf",
        "directed": True,
        "weight": 1.0,
    }
    table = pa.Table.from_pylist([bad_edge], schema=EDGES_SCHEMA)
    with pytest.raises(ArtifactValidationError):
        write_edges(table, tmp_path / "edges.parquet", nodes=nodes_table)
