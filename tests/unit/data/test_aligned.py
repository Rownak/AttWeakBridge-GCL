"""Tests for AlignedMatrix (load/select/save) and NodeUniverse."""

from __future__ import annotations

import numpy as np
import pytest

from cga.core.errors import ArtifactValidationError
from cga.data.aligned import AlignedMatrix, NodeUniverse


def test_save_then_load_roundtrip(tmp_path):
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    node_ids = ["a:1", "a:2", "a:3"]
    matrix = AlignedMatrix(vectors=vectors, node_ids=node_ids)
    matrix.save(tmp_path)

    loaded = AlignedMatrix.load(tmp_path)
    assert loaded.node_ids == node_ids
    assert np.array_equal(loaded.vectors, vectors)


def test_row_of_returns_correct_index():
    matrix = AlignedMatrix(vectors=np.zeros((3, 2), dtype=np.float32), node_ids=["a", "b", "c"])
    assert matrix.row_of("b") == 1


def test_row_of_missing_id_raises():
    matrix = AlignedMatrix(vectors=np.zeros((1, 2), dtype=np.float32), node_ids=["a"])
    with pytest.raises(ArtifactValidationError):
        matrix.row_of("missing")


def test_select_reorders_correctly():
    vectors = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    matrix = AlignedMatrix(vectors=vectors, node_ids=["a", "b", "c"])

    reordered = matrix.select(["c", "a"])
    assert reordered.node_ids == ["c", "a"]
    assert np.array_equal(reordered.vectors, np.array([[3.0], [1.0]], dtype=np.float32))


def test_select_raises_on_missing_id():
    matrix = AlignedMatrix(vectors=np.zeros((2, 2), dtype=np.float32), node_ids=["a", "b"])
    with pytest.raises(ArtifactValidationError):
        matrix.select(["a", "does-not-exist"])


def test_aligned_matrix_rejects_duplicate_ids():
    with pytest.raises(ArtifactValidationError):
        AlignedMatrix(vectors=np.zeros((2, 2), dtype=np.float32), node_ids=["a", "a"])


def test_aligned_matrix_rejects_row_count_mismatch():
    with pytest.raises(ArtifactValidationError):
        AlignedMatrix(vectors=np.zeros((3, 2), dtype=np.float32), node_ids=["a", "b"])


def test_node_universe_save_and_load_roundtrip(tmp_path):
    universe = NodeUniverse(node_ids=["a:1", "b:2"], graph_roles=["graph_1", "graph_2"])
    universe.save(tmp_path / "universe.parquet")

    loaded = NodeUniverse.from_parquet(tmp_path / "universe.parquet")
    assert loaded.node_ids == ["a:1", "b:2"]
    assert loaded.graph_roles == ["graph_1", "graph_2"]


def test_node_universe_rejects_duplicate_ids():
    with pytest.raises(ArtifactValidationError):
        NodeUniverse(node_ids=["a", "a"], graph_roles=["graph_1", "graph_2"])


def test_node_universe_rejects_length_mismatch():
    with pytest.raises(ArtifactValidationError):
        NodeUniverse(node_ids=["a", "b"], graph_roles=["graph_1"])
