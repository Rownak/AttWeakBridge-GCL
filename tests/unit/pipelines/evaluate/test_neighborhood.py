"""Tests for score_view: same-graph_role restriction, eligibility, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.neighborhood import score_view


def _matrix(ids: list[str], vectors: np.ndarray) -> AlignedMatrix:
    return AlignedMatrix(vectors=vectors.astype(np.float32), node_ids=ids)


def test_score_view_identical_matrices_gives_perfect_scores():
    ids = ["g1:a", "g1:b", "g1:c", "g2:x"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
    roles = {"g1:a": "graph_1", "g1:b": "graph_1", "g1:c": "graph_1", "g2:x": "graph_2"}

    learned = _matrix(ids, vectors)
    reference = _matrix(ids, vectors)

    results = score_view(
        learned_matrix=learned, reference_matrix=reference, graph_role_by_node=roles, role="graph_1", ks=[1, 2]
    )

    assert results[1].n_queries == 3
    assert results[1].precision_at_k == pytest.approx(1.0)
    assert results[1].hit_at_k == pytest.approx(1.0)
    assert results[1].mrr == pytest.approx(1.0)
    assert results[2].precision_at_k == pytest.approx(1.0)


def test_score_view_excludes_cross_role_pairs():
    # graph_2's single node must never appear as a graph_1 candidate/query.
    ids = ["g1:a", "g1:b", "g2:x"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    roles = {"g1:a": "graph_1", "g1:b": "graph_1", "g2:x": "graph_2"}

    learned = _matrix(ids, vectors)
    reference = _matrix(ids, vectors)

    results = score_view(
        learned_matrix=learned, reference_matrix=reference, graph_role_by_node=roles, role="graph_1", ks=[1]
    )
    # Only g1:a and g1:b are eligible graph_1 queries.
    assert results[1].n_queries == 2

    results_g2 = score_view(
        learned_matrix=learned, reference_matrix=reference, graph_role_by_node=roles, role="graph_2", ks=[1]
    )
    # A single-node role has no candidates to rank against; scored as empty.
    assert results_g2[1].n_queries == 0


def test_score_view_excludes_nodes_missing_from_reference():
    ids_learned = ["g1:a", "g1:b", "g1:c"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    roles = {"g1:a": "graph_1", "g1:b": "graph_1", "g1:c": "graph_1"}

    learned = _matrix(ids_learned, vectors)
    # Reference store is missing "g1:c" -- e.g. no text for that node.
    reference = _matrix(["g1:a", "g1:b"], vectors[:2])

    results = score_view(
        learned_matrix=learned, reference_matrix=reference, graph_role_by_node=roles, role="graph_1", ks=[1]
    )
    assert results[1].n_queries == 2


def test_score_view_uninverted_learned_ranking_misses_reference_neighbor():
    # Reference agrees a<->b are nearest; learned space disagrees entirely.
    ids = ["g1:a", "g1:b", "g1:c"]
    reference_vectors = np.array([[1.0, 0.0], [1.0, 0.01], [10.0, 10.0]])
    learned_vectors = np.array([[1.0, 0.0], [10.0, 10.0], [1.0, 0.01]])
    roles = {"g1:a": "graph_1", "g1:b": "graph_1", "g1:c": "graph_1"}

    learned = _matrix(ids, learned_vectors)
    reference = _matrix(ids, reference_vectors)

    results = score_view(
        learned_matrix=learned, reference_matrix=reference, graph_role_by_node=roles, role="graph_1", ks=[1]
    )
    # For query "a", reference nearest is "b", but learned top-1 for "a" is "c" -> miss.
    assert results[1].hit_at_k < 1.0
