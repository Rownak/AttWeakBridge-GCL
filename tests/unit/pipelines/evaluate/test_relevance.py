"""Unit tests for ``RelevanceProvider`` implementations."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from cga.core.errors import ConfigError
from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.relevance import CosineTopKRelevance, HopRadiusRelevance, relevance_providers


def _path_graph(*node_ids: str) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(zip(node_ids, node_ids[1:]))
    return graph


# --- HopRadiusRelevance ------------------------------------------------------


def test_hop_radius_r1_and_r2_on_path_graph():
    graph = _path_graph("a", "b", "c", "d")
    candidates = ["a", "b", "c", "d"]

    r1 = HopRadiusRelevance(graph, radius=1)
    r1.prepare(candidate_ids=candidates)
    assert r1.relevant("b") == {"a", "c"}

    r2 = HopRadiusRelevance(graph, radius=2)
    r2.prepare(candidate_ids=candidates)
    assert r2.relevant("b") == {"a", "c", "d"}


def test_hop_radius_r2_superset_of_r1():
    graph = _path_graph("a", "b", "c", "d", "e")
    candidates = ["a", "b", "c", "d", "e"]

    r1 = HopRadiusRelevance(graph, radius=1)
    r1.prepare(candidate_ids=candidates)
    r2 = HopRadiusRelevance(graph, radius=2)
    r2.prepare(candidate_ids=candidates)

    for node_id in candidates:
        assert r1.relevant(node_id) <= r2.relevant(node_id)


def test_hop_radius_isolate_excluded_from_defined_for():
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "isolated"])
    graph.add_edge("a", "b")

    provider = HopRadiusRelevance(graph, radius=1)
    provider.prepare(candidate_ids=["a", "b", "isolated"])

    assert "isolated" not in provider.defined_for()
    assert provider.relevant("isolated") == set()
    assert {"a", "b"} <= provider.defined_for()


def test_hop_radius_neighbor_outside_candidate_pool_treated_as_isolate():
    # "b" has one graph edge, to "c", which is excluded from the candidate
    # pool -- within this scope "b" must behave like an isolate (no relevant
    # set), even though it has a real edge in the full graph.
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("b", "c")  # "a" is unconnected to either

    provider = HopRadiusRelevance(graph, radius=1)
    provider.prepare(candidate_ids=["a", "b"])  # "c" excluded from the pool

    assert provider.relevant("b") == set()
    assert "b" not in provider.defined_for()


def test_hop_radius_query_never_in_its_own_relevant_set():
    graph = _path_graph("a", "b", "c")
    provider = HopRadiusRelevance(graph, radius=2)
    provider.prepare(candidate_ids=["a", "b", "c"])
    for node_id in ("a", "b", "c"):
        assert node_id not in provider.relevant(node_id)


def test_hop_radius_rejects_non_positive_radius():
    with pytest.raises(ConfigError):
        HopRadiusRelevance(nx.Graph(), radius=0)


# --- CosineTopKRelevance -----------------------------------------------------


def test_cosine_top_k_relevant_set_matches_expected_neighbors():
    ids = ["a", "b", "c", "d"]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],  # very close to "a"
            [0.0, 1.0],
            [-1.0, 0.0],  # opposite of "a"
        ],
        dtype=np.float32,
    )
    matrix = AlignedMatrix(vectors=vectors, node_ids=ids)
    provider = CosineTopKRelevance(matrix, top_k=1)
    provider.prepare(candidate_ids=ids)
    assert provider.relevant("a") == {"b"}


def test_cosine_top_k_ties_break_by_id():
    ids = ["z", "b", "a"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    matrix = AlignedMatrix(vectors=vectors, node_ids=ids)
    provider = CosineTopKRelevance(matrix, top_k=1)
    provider.prepare(candidate_ids=ids)
    # "b" and "a" are exactly equidistant from "z"; id tie-break picks "a".
    assert provider.relevant("z") == {"a"}


def test_cosine_top_k_query_excluded_from_its_own_relevant_set():
    ids = ["a", "b", "c"]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    matrix = AlignedMatrix(vectors=vectors, node_ids=ids)
    provider = CosineTopKRelevance(matrix, top_k=2)
    provider.prepare(candidate_ids=ids)
    assert "a" not in provider.relevant("a")


def test_cosine_top_k_rejects_non_positive_top_k():
    matrix = AlignedMatrix(vectors=np.zeros((2, 2), dtype=np.float32), node_ids=["a", "b"])
    with pytest.raises(ConfigError):
        CosineTopKRelevance(matrix, top_k=0)


def test_cosine_top_k_handles_candidates_missing_from_reference():
    ids = ["a", "b"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matrix = AlignedMatrix(vectors=vectors, node_ids=ids)
    provider = CosineTopKRelevance(matrix, top_k=1)
    provider.prepare(candidate_ids=["a", "b", "not_in_reference"])
    assert "not_in_reference" not in provider.defined_for()


# --- registry dispatch -------------------------------------------------------


def test_unknown_relevance_provider_raises_config_error():
    from cga.core.errors import ConfigError as CE

    with pytest.raises(CE):
        relevance_providers.get("bogus")


def test_registered_providers_include_builtins():
    assert {"hop_radius", "cosine_top_k"} <= set(relevance_providers.names())
