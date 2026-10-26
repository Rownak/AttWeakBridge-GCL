"""Tests for ``cga.pipelines.embed.graph_embedders``."""

from __future__ import annotations

import numpy as np

from cga.pipelines.embed.graph_embedders import build_graph, deepwalk_embed, node2vec_embed


def _sample_graph():
    node_ids = ["a:1", "a:2", "a:3", "a:4", "a:isolated"]
    edges = [("a:1", "a:2"), ("a:2", "a:3"), ("a:3", "a:4"), ("a:4", "a:1")]
    return build_graph(node_ids, edges)


def test_build_graph_keeps_isolated_nodes_with_zero_degree():
    graph = _sample_graph()
    assert "a:isolated" in graph.nodes()
    assert graph.degree("a:isolated") == 0


def test_deepwalk_embed_is_seeded_and_reproducible():
    graph = _sample_graph()
    kwargs = dict(dimensions=16, walk_length=10, num_walks=20, window=3, workers=1, epochs=5)

    np.random.seed(0)
    first = deepwalk_embed(graph, **kwargs)
    np.random.seed(0)
    second = deepwalk_embed(graph, **kwargs)

    assert set(first) == set(second) == set(graph.nodes())
    for node in first:
        assert first[node] == second[node]


def test_node2vec_embed_is_seeded_and_reproducible():
    graph = _sample_graph()
    kwargs = dict(dimensions=16, walk_length=10, num_walks=20, window=3, workers=1, epochs=5, p=0.5, q=2.0)

    np.random.seed(0)
    first = node2vec_embed(graph, **kwargs)
    np.random.seed(0)
    second = node2vec_embed(graph, **kwargs)

    for node in first:
        assert first[node] == second[node]


def test_deepwalk_embed_vector_dimensions_match_config():
    graph = _sample_graph()
    vectors = deepwalk_embed(graph, dimensions=32, walk_length=10, num_walks=10, window=3, workers=1, epochs=3)
    for vector in vectors.values():
        assert len(vector) == 32
