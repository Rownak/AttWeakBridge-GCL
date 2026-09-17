"""Tests for cosine-similarity construction, including held_out exclusion."""

from __future__ import annotations

import numpy as np

from cga.data.aligned import AlignedMatrix
from cga.pipelines.training_data.similarity import compute_graph_similarity, compute_text_similarity


def _matrix(node_ids: list[str], dim: int = 3, seed: int = 0) -> AlignedMatrix:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(node_ids), dim)).astype(np.float32)
    return AlignedMatrix(vectors=vectors, node_ids=node_ids)


def test_compute_text_similarity_excludes_held_out_nodes():
    node_ids = ["a:1", "a:2", "a:3", "b:1", "b:2"]
    text_matrix = _matrix(node_ids)
    graph_role_by_node = {"a:1": "graph_1", "a:2": "graph_1", "a:3": "graph_1", "b:1": "graph_2", "b:2": "graph_2"}
    held_out_by_node = {"a:2": True, "b:2": True}

    sim = compute_text_similarity(text_matrix, graph_role_by_node=graph_role_by_node, held_out_by_node=held_out_by_node)

    assert sim.row_ids == ["a:1", "a:3"]
    assert sim.col_ids == ["b:1"]
    assert sim.values.shape == (2, 1)


def test_compute_graph_similarity_within_role_excludes_held_out():
    node_ids = ["a:1", "a:2", "a:3"]
    graph_matrix = _matrix(node_ids)
    graph_role_by_node = {"a:1": "graph_1", "a:2": "graph_1", "a:3": "graph_1"}
    held_out_by_node = {"a:3": True}

    sim = compute_graph_similarity(
        graph_matrix, role="graph_1", graph_role_by_node=graph_role_by_node, held_out_by_node=held_out_by_node
    )

    assert sim.row_ids == ["a:1", "a:2"]
    assert sim.col_ids == ["a:1", "a:2"]
    assert sim.values.shape == (2, 2)
    # diagonal cosine similarity with itself is ~1
    assert np.isclose(sim.values[0, 0], 1.0, atol=1e-5)


def test_compute_text_similarity_skips_ids_missing_from_store():
    node_ids = ["a:1", "b:1"]
    text_matrix = _matrix(node_ids)
    graph_role_by_node = {"a:1": "graph_1", "a:2": "graph_1", "b:1": "graph_2"}
    held_out_by_node = {}

    sim = compute_text_similarity(text_matrix, graph_role_by_node=graph_role_by_node, held_out_by_node=held_out_by_node)

    assert sim.row_ids == ["a:1"]  # a:2 absent from text_matrix, skipped not raised
    assert sim.col_ids == ["b:1"]
