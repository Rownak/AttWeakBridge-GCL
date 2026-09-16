"""Tests for example_store FK enforcement against a NodeUniverse."""

from __future__ import annotations

import pyarrow as pa
import pytest

from cga.core.errors import ArtifactValidationError
from cga.data.aligned import NodeUniverse
from cga.data.example_store import read_examples, write_examples
from cga.data.schemas import EXAMPLES_SCHEMA


def _example_row(example_id: str, anchor: str, text_pos: str, text_neg: str, graph_pos: str, graph_neg: str) -> dict:
    return {
        "example_id": example_id,
        "anchor_id": anchor,
        "text_pos_id": text_pos,
        "text_neg_id": text_neg,
        "graph_pos_id": graph_pos,
        "graph_neg_id": graph_neg,
        "sampler": "rank_topk",
        "split": "train",
    }


def _examples_table(rows: list[dict]) -> pa.Table:
    columns = {field.name: [row[field.name] for row in rows] for field in EXAMPLES_SCHEMA}
    return pa.table(columns, schema=EXAMPLES_SCHEMA)


def test_write_examples_success(tmp_path):
    universe = NodeUniverse(
        node_ids=["a:1", "a:2", "a:3", "a:4", "a:5"],
        graph_roles=["graph_1"] * 5,
    )
    table = _examples_table([_example_row("e1", "a:1", "a:2", "a:3", "a:4", "a:5")])
    write_examples(table, tmp_path / "examples.parquet", universe=universe)
    assert read_examples(tmp_path / "examples.parquet").num_rows == 1


def test_write_examples_rejects_id_outside_universe(tmp_path):
    universe = NodeUniverse(node_ids=["a:1", "a:2"], graph_roles=["graph_1", "graph_1"])
    table = _examples_table([_example_row("e1", "a:1", "a:2", "a:OUTSIDE", "a:1", "a:2")])
    with pytest.raises(ArtifactValidationError):
        write_examples(table, tmp_path / "examples.parquet", universe=universe)


def test_write_examples_rejects_duplicate_example_id(tmp_path):
    universe = NodeUniverse(node_ids=["a:1", "a:2"], graph_roles=["graph_1", "graph_1"])
    table = _examples_table(
        [
            _example_row("dup", "a:1", "a:2", "a:1", "a:2", "a:1"),
            _example_row("dup", "a:2", "a:1", "a:2", "a:1", "a:2"),
        ]
    )
    with pytest.raises(ArtifactValidationError):
        write_examples(table, tmp_path / "examples.parquet", universe=universe)
