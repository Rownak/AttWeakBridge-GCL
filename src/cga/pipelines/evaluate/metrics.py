"""Neighborhood-preservation metrics: Precision@k, Hit@k, MRR (target_architecture.md §5.5a).

Pure functions over ranked id lists, no I/O. Every function takes the query's
own id excluded from both the learned ranking and the reference set by the
caller — these functions never special-case self-similarity.

Retrieval direction is always *learned embedding -> reference top-k*: the
ranking comes from the learned space, "relevant" comes from the reference
(text or structural) space. Ties in either ranking are broken by node id, so
results never depend on argsort's unstable tie order.
"""

from __future__ import annotations

import numpy as np


def rank_by_similarity(similarity_row: np.ndarray, candidate_ids: list[str]) -> list[str]:
    """Return ``candidate_ids`` ordered by descending similarity, ties broken by id.

    Args:
        similarity_row: Similarity of the query against each of ``candidate_ids``,
            same order and length.
        candidate_ids: Candidate node ids, excluding the query itself.

    Returns:
        ``candidate_ids`` reordered most-similar first.
    """
    if len(similarity_row) != len(candidate_ids):
        raise ValueError("similarity_row and candidate_ids must be the same length")
    order = sorted(range(len(candidate_ids)), key=lambda i: (-similarity_row[i], candidate_ids[i]))
    return [candidate_ids[i] for i in order]


def top_k(ranked_ids: list[str], k: int) -> list[str]:
    """Return the first ``k`` ids of an already-ranked list (fewer if it is shorter)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return ranked_ids[:k]


def precision_at_k(learned_ranked_ids: list[str], reference_top_k_ids: list[str], k: int) -> float:
    """Fraction of the learned top-``k`` that also appears in the reference top-``k``.

    Args:
        learned_ranked_ids: Candidate ids ranked by the learned embedding, most
            similar first, query excluded.
        reference_top_k_ids: The reference view's top-``k`` neighbor ids for
            this query (the relevance set), query excluded.
        k: Cutoff. ``learned_ranked_ids`` is truncated to ``k`` before scoring.

    Returns:
        ``|learned_top_k ∩ reference_top_k| / k``. 0.0 if ``k`` is 0 (no
        candidates to score, e.g. an isolated query in a size-1 role).
    """
    if k == 0:
        return 0.0
    learned_top_k = set(top_k(learned_ranked_ids, k))
    relevant = set(reference_top_k_ids)
    return len(learned_top_k & relevant) / k


def hit_at_k(learned_ranked_ids: list[str], reference_top_k_ids: list[str], k: int) -> float:
    """Whether the reference's single nearest neighbor lands in the learned top-``k``.

    Args:
        learned_ranked_ids: Candidate ids ranked by the learned embedding, most
            similar first, query excluded.
        reference_top_k_ids: The reference view's top-k, ranked most similar
            first; only element ``[0]`` (the reference's nearest neighbor) is
            used as the relevant item.
        k: Cutoff.

    Returns:
        1.0 if the reference's nearest neighbor is within the learned top-``k``,
        else 0.0. 0.0 if the reference has no neighbor at all.
    """
    if not reference_top_k_ids:
        return 0.0
    nearest = reference_top_k_ids[0]
    return 1.0 if nearest in set(top_k(learned_ranked_ids, k)) else 0.0


def mrr(learned_ranked_ids: list[str], reference_top_k_ids: list[str]) -> float:
    """Reciprocal rank of the reference's nearest neighbor within the full learned ranking.

    Args:
        learned_ranked_ids: Candidate ids ranked by the learned embedding, most
            similar first, query excluded.
        reference_top_k_ids: The reference view's top-k, ranked most similar
            first; only element ``[0]`` is used as the relevant item.

    Returns:
        ``1 / rank`` (1-indexed) of the reference's nearest neighbor in
        ``learned_ranked_ids``, or 0.0 if it does not appear there (or the
        reference has no neighbor at all).
    """
    if not reference_top_k_ids:
        return 0.0
    nearest = reference_top_k_ids[0]
    try:
        rank = learned_ranked_ids.index(nearest) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank
