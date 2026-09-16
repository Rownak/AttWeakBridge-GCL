"""Tests for graph_store schema enforcement and FK/pk checks."""

from __future__ import annotations

import pyarrow as pa
import pytest

from cga.core.errors import ArtifactValidationError
from cga.data.graph_store import (
    read_alignments,
    read_edges,
    read_nodes,
    read_texts,
    write_alignments,
    write_edges,
    write_nodes,
    write_texts,
)
from cga.data.schemas import ALIGNMENTS_SCHEMA, EDGES_SCHEMA, NODES_SCHEMA, TEXTS_SCHEMA


def _nodes_table(rows: list[dict]) -> pa.Table:
    columns = {field.name: [] for field in NODES_SCHEMA}
    for row in rows:
        for field in NODES_SCHEMA:
            columns[field.name].append(row.get(field.name))
    return pa.table(columns, schema=NODES_SCHEMA)


def _node_row(node_id: str, graph_role: str) -> dict:
    graph_key, source_id = node_id.split(":", 1)
    return {
        "node_id": node_id,
        "graph_key": graph_key,
        "graph_role": graph_role,
        "source_id": source_id,
        "external_id": source_id,
        "name": source_id,
        "node_type": None,
        "metadata": "{}",
        "has_text": True,
        "source_version": "v1",
        "raw_sha256": "0" * 64,
    }


def test_write_and_read_nodes_roundtrip(tmp_path):
    table = _nodes_table([_node_row("attack:T1", "graph_1"), _node_row("cwe:79", "graph_2")])
    write_nodes(table, tmp_path / "nodes.parquet")
    read_back = read_nodes(tmp_path / "nodes.parquet")
    assert read_back.num_rows == 2


def test_write_nodes_rejects_duplicate_pk(tmp_path):
    table = _nodes_table([_node_row("attack:T1", "graph_1"), _node_row("attack:T1", "graph_1")])
    with pytest.raises(ArtifactValidationError):
        write_nodes(table, tmp_path / "nodes.parquet")


def test_write_nodes_schema_violation_raises(tmp_path):
    bad_table = pa.table({"node_id": ["attack:T1"], "unexpected_column": [123]})
    with pytest.raises(ArtifactValidationError):
        write_nodes(bad_table, tmp_path / "nodes.parquet")


def test_write_edges_success(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1"), _node_row("attack:T2", "graph_1")])
    edges = pa.table(
        {
            "edge_id": ["e1"],
            "src_id": ["attack:T1"],
            "dst_id": ["attack:T2"],
            "graph_role": ["graph_1"],
            "relation": ["uses"],
            "directed": [True],
            "weight": pa.array([1.0], type=pa.float32()),
        },
        schema=EDGES_SCHEMA,
    )
    write_edges(edges, tmp_path / "edges.parquet", nodes=nodes)
    assert read_edges(tmp_path / "edges.parquet").num_rows == 1


def test_write_edges_rejects_unknown_endpoint(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1")])
    edges = pa.table(
        {
            "edge_id": ["e1"],
            "src_id": ["attack:T1"],
            "dst_id": ["attack:UNKNOWN"],
            "graph_role": ["graph_1"],
            "relation": ["uses"],
            "directed": [True],
            "weight": pa.array([1.0], type=pa.float32()),
        },
        schema=EDGES_SCHEMA,
    )
    with pytest.raises(ArtifactValidationError):
        write_edges(edges, tmp_path / "edges.parquet", nodes=nodes)


def test_write_edges_rejects_cross_graph_role(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1"), _node_row("cwe:79", "graph_2")])
    edges = pa.table(
        {
            "edge_id": ["e1"],
            "src_id": ["attack:T1"],
            "dst_id": ["cwe:79"],
            "graph_role": ["graph_1"],
            "relation": ["uses"],
            "directed": [True],
            "weight": pa.array([1.0], type=pa.float32()),
        },
        schema=EDGES_SCHEMA,
    )
    with pytest.raises(ArtifactValidationError):
        write_edges(edges, tmp_path / "edges.parquet", nodes=nodes)


def test_write_texts_success_and_fk(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1")])
    texts = pa.table(
        {
            "node_id": ["attack:T1"],
            "text_variant": ["description"],
            "text": ["some text"],
            "text_sha256": ["a" * 64],
            "n_chars": pa.array([9], type=pa.int32()),
        },
        schema=TEXTS_SCHEMA,
    )
    write_texts(texts, tmp_path / "texts.parquet", nodes=nodes)
    assert read_texts(tmp_path / "texts.parquet").num_rows == 1


def test_write_texts_rejects_unknown_node(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1")])
    texts = pa.table(
        {
            "node_id": ["attack:UNKNOWN"],
            "text_variant": ["description"],
            "text": ["some text"],
            "text_sha256": ["a" * 64],
            "n_chars": pa.array([9], type=pa.int32()),
        },
        schema=TEXTS_SCHEMA,
    )
    with pytest.raises(ArtifactValidationError):
        write_texts(texts, tmp_path / "texts.parquet", nodes=nodes)


def test_write_alignments_success(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1"), _node_row("cwe:79", "graph_2")])
    alignments = pa.table(
        {
            "alignment_id": ["a1"],
            "node_1_id": ["attack:T1"],
            "node_2_id": ["cwe:79"],
            "relation": ["equivalence"],
            "score": pa.array([1.0], type=pa.float32()),
            "split": ["train"],
            "candidates": pa.array([None], type=pa.list_(pa.string())),
        },
        schema=ALIGNMENTS_SCHEMA,
    )
    write_alignments(alignments, tmp_path / "alignments.parquet", nodes=nodes)
    assert read_alignments(tmp_path / "alignments.parquet").num_rows == 1


def test_write_alignments_rejects_wrong_role_order(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1"), _node_row("cwe:79", "graph_2")])
    alignments = pa.table(
        {
            "alignment_id": ["a1"],
            "node_1_id": ["cwe:79"],
            "node_2_id": ["attack:T1"],
            "relation": ["equivalence"],
            "score": pa.array([1.0], type=pa.float32()),
            "split": ["train"],
            "candidates": pa.array([None], type=pa.list_(pa.string())),
        },
        schema=ALIGNMENTS_SCHEMA,
    )
    with pytest.raises(ArtifactValidationError):
        write_alignments(alignments, tmp_path / "alignments.parquet", nodes=nodes)


def test_write_alignments_empty_table_is_valid(tmp_path):
    nodes = _nodes_table([_node_row("attack:T1", "graph_1")])
    empty = pa.table(
        {
            "alignment_id": pa.array([], type=pa.string()),
            "node_1_id": pa.array([], type=pa.string()),
            "node_2_id": pa.array([], type=pa.string()),
            "relation": pa.array([], type=pa.string()),
            "score": pa.array([], type=pa.float32()),
            "split": pa.array([], type=pa.string()),
            "candidates": pa.array([], type=pa.list_(pa.string())),
        },
        schema=ALIGNMENTS_SCHEMA,
    )
    write_alignments(empty, tmp_path / "alignments.parquet", nodes=nodes)
    assert read_alignments(tmp_path / "alignments.parquet").num_rows == 0
