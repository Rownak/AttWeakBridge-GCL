"""Cosine similarity matrices for contrastive sampling (target_architecture.md §5.3 step 2).

Two views, both restricted to non-``held_out`` universe rows so a node reserved
for evaluation can never enter a similarity ranking that unsupervised sampling
draws from:

- **Cross-graph, text space**: ``graph_1`` rows against ``graph_2`` columns,
  using the text embedding store.
- **Within-graph, structural space**: ``graph_1`` against itself and
  ``graph_2`` against itself, using the graph embedding store.

Everything here is keyed by ``node_id``, never by a raw array position, so a
caller can never accidentally index one store's matrix with another store's
row order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cga.data.aligned import AlignedMatrix


@dataclass
class SimilarityMatrix:
    """A dense cosine-similarity matrix between two ordered, disjoint id lists.

    Attributes:
        row_ids: Ordered ids indexing ``values``'s rows.
        col_ids: Ordered ids indexing ``values``'s columns.
        values: ``(len(row_ids), len(col_ids))`` cosine similarities.
    """

    row_ids: list[str]
    col_ids: list[str]
    values: np.ndarray


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the ``(len(a), len(b))`` cosine similarity matrix between two stacks of vectors."""
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def _eligible_ids_by_role(
    graph_role_by_node: dict[str, str], held_out_by_node: dict[str, bool]
) -> dict[str, list[str]]:
    """Group non-``held_out`` node ids by ``graph_role``, preserving encounter order."""
    by_role: dict[str, list[str]] = {"graph_1": [], "graph_2": []}
    for node_id, role in graph_role_by_node.items():
        if held_out_by_node.get(node_id, False):
            continue
        by_role[role].append(node_id)
    return by_role


def compute_text_similarity(
    text_matrix: AlignedMatrix,
    *,
    graph_role_by_node: dict[str, str],
    held_out_by_node: dict[str, bool],
) -> SimilarityMatrix:
    """Cross-graph cosine similarity in text-embedding space, ``graph_1`` rows x ``graph_2`` cols.

    Only node ids present in ``text_matrix`` **and** not ``held_out`` are used;
    a node absent from the text store (e.g. excluded by the universe policy) is
    silently skipped rather than raising, since ``build_universe`` is the single
    place eligibility is decided.
    """
    by_role = _eligible_ids_by_role(graph_role_by_node, held_out_by_node)
    text_ids = set(text_matrix.node_ids)
    row_ids = [nid for nid in by_role["graph_1"] if nid in text_ids]
    col_ids = [nid for nid in by_role["graph_2"] if nid in text_ids]

    row_vectors = text_matrix.select(row_ids).vectors if row_ids else np.empty((0, text_matrix.vectors.shape[1]))
    col_vectors = text_matrix.select(col_ids).vectors if col_ids else np.empty((0, text_matrix.vectors.shape[1]))
    values = _cosine_similarity(row_vectors, col_vectors)
    return SimilarityMatrix(row_ids=row_ids, col_ids=col_ids, values=values)


def compute_text_within_similarity(
    text_matrix: AlignedMatrix,
    *,
    role: str,
    graph_role_by_node: dict[str, str],
    held_out_by_node: dict[str, bool],
) -> SimilarityMatrix:
    """Within-graph cosine similarity in text-embedding space for one ``graph_role``.

    Used by the ``cross_modal_hard`` negatives to ask how text-similar two nodes of the same graph
    are. Same eligibility as :func:`compute_text_similarity`: ids present in ``text_matrix`` and not
    ``held_out``.
    """
    by_role = _eligible_ids_by_role(graph_role_by_node, held_out_by_node)
    text_ids = set(text_matrix.node_ids)
    ids = [nid for nid in by_role[role] if nid in text_ids]

    vectors = text_matrix.select(ids).vectors if ids else np.empty((0, text_matrix.vectors.shape[1]))
    values = _cosine_similarity(vectors, vectors)
    return SimilarityMatrix(row_ids=ids, col_ids=ids, values=values)


def compute_graph_similarity(
    graph_matrix: AlignedMatrix,
    *,
    role: str,
    graph_role_by_node: dict[str, str],
    held_out_by_node: dict[str, bool],
) -> SimilarityMatrix:
    """Within-graph cosine similarity in structural-embedding space for one ``graph_role``.

    Only node ids present in ``graph_matrix`` **and** not ``held_out`` are used.
    """
    by_role = _eligible_ids_by_role(graph_role_by_node, held_out_by_node)
    graph_ids = set(graph_matrix.node_ids)
    ids = [nid for nid in by_role[role] if nid in graph_ids]

    vectors = graph_matrix.select(ids).vectors if ids else np.empty((0, graph_matrix.vectors.shape[1]))
    values = _cosine_similarity(vectors, vectors)
    return SimilarityMatrix(row_ids=ids, col_ids=ids, values=values)
