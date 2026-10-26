"""``deepwalk`` and ``node2vec`` structural embedders: random walks -> ``gensim.Word2Vec``.

Ported from ``legacy/GCL/get_graph_features/2_node2vec_deepwalk.py``, generalized
to operate on the canonical ``edges.parquet`` (string ``node_id``s instead of
integer row indices) and seeded via ``numpy``/``random`` for reproducibility
(target_architecture.md §5.2, §7.5). Walks are built once for every node in the
graph; nodes with no edges are isolated and the caller records them as
``status = skipped`` rather than this module raising.

``node2vec``'s 2nd-order biased walk follows Grover & Leskovec (2016): the
transition weight back to the previous node is scaled by ``1/p``, to an
unvisited neighbor shared with the previous node by ``1``, and to any other
neighbor by ``1/q``. Plain ``deepwalk`` is the ``p = q = 1`` uniform-walk
special case, kept as its own function for clarity and to match the legacy
two-script split.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from gensim.models import Word2Vec


def generate_random_walks(graph: nx.Graph, *, num_walks: int, walk_length: int) -> list[list[str]]:
    """Generate uniform (DeepWalk) random walks over every node in ``graph``."""
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(graph.neighbors(cur))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(walk)
    return walks


def _biased_next_node(graph: nx.Graph, prev: str | None, cur: str, p: float, q: float) -> str | None:
    neighbors = list(graph.neighbors(cur))
    if not neighbors:
        return None
    prev_neighbors = set(graph.neighbors(prev)) if prev is not None else set()
    weights = []
    for nxt in neighbors:
        if nxt == prev:
            weights.append(1.0 / p)
        elif nxt in prev_neighbors:
            weights.append(1.0)
        else:
            weights.append(1.0 / q)
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    return np.random.choice(neighbors, p=weights_arr)


def generate_biased_random_walks(
    graph: nx.Graph, *, num_walks: int, walk_length: int, p: float, q: float
) -> list[list[str]]:
    """Generate 2nd-order biased (node2vec) random walks over every node in ``graph``."""
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            prev = None
            while len(walk) < walk_length:
                cur = walk[-1]
                next_node = _biased_next_node(graph, prev, cur, p, q)
                if next_node is None:
                    break
                walk.append(next_node)
                prev = cur
            walks.append(walk)
    return walks


def deepwalk_embed(
    graph: nx.Graph,
    *,
    dimensions: int = 128,
    walk_length: int = 30,
    num_walks: int = 200,
    window: int = 5,
    min_count: int = 0,
    workers: int = 1,
    epochs: int = 10,
) -> dict[str, list[float]]:
    """Return ``{node_id: vector}`` for every node in ``graph`` via DeepWalk.

    ``workers`` defaults to 1: gensim's multi-threaded training is
    non-deterministic even with a fixed seed (target_architecture.md §7.5).
    """
    walks = generate_random_walks(graph, num_walks=num_walks, walk_length=walk_length)
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=min_count,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=1,
    )
    return {node: model.wv[node].tolist() for node in graph.nodes()}


def node2vec_embed(
    graph: nx.Graph,
    *,
    dimensions: int = 128,
    walk_length: int = 30,
    num_walks: int = 200,
    window: int = 5,
    min_count: int = 0,
    workers: int = 1,
    epochs: int = 10,
    p: float = 1.0,
    q: float = 1.0,
) -> dict[str, list[float]]:
    """Return ``{node_id: vector}`` for every node in ``graph`` via node2vec."""
    walks = generate_biased_random_walks(graph, num_walks=num_walks, walk_length=walk_length, p=p, q=q)
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=min_count,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=1,
    )
    return {node: model.wv[node].tolist() for node in graph.nodes()}


def build_graph(node_ids: list[str], edges: list[tuple[str, str]]) -> nx.Graph:
    """Build an undirected ``networkx.Graph`` with every ``node_ids`` entry present.

    Nodes are added explicitly (not only inferred from edges) so isolated
    nodes still appear in ``graph.nodes()`` with degree zero, letting the
    caller record them as ``status = skipped`` rather than dropping them.
    """
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    return graph
