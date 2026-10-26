"""Set-based neighborhood-preservation metrics (target_architecture.md §5.5a).

Pure functions over ranked id lists and relevance **sets**, no I/O. Every
function takes the query's own id already excluded from both the learned
ranking and the relevant set by the caller -- these functions never
special-case self-similarity.

Retrieval direction is always *method ranking -> view's relevant set*: the
ranking comes from the method embedding, "relevant" comes from a
``RelevanceProvider`` (hop-radius or cosine top-k). Ties in a ranking are
broken by node id, so results never depend on argsort's unstable tie order.

All four metrics are pure set operations against ``relevant: set[str]`` --
deliberately no ranked/unranked distinction on this interface. A provider
that is internally ranked (``cosine_top_k``) consumes that rank to pick its
top-k and discards it before returning a set; a provider that is not
ranked (``hop_radius``) never had one. Keeping the metric functions
set-only is what lets one code path score both.

.. note::
   These are set-based generalizations of the Phase 1.8 metrics, which used
   only the reference's single nearest neighbor as the relevant item. The
   set-based versions are strictly more lenient (any relevant item counts,
   not just the nearest) -- do not compare Phase 1.9 numbers against
   previously computed Phase 1.8 ones.
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


def top_k_by_similarity(similarity_row: np.ndarray, candidate_ids: list[str], k: int) -> list[str]:
    """Return the ``k`` highest-similarity ids, ties broken by id -- the vectorized equivalent of
    ``rank_by_similarity(...)[:k]``.

    Uses ``np.lexsort`` over ``(-similarity, id_rank)`` to get the exact same
    order as :func:`rank_by_similarity` without a Python-level tuple sort of
    ``candidate_ids``. (An id-oblivious ``np.argpartition`` on similarity
    alone is not used here: it can select the wrong member of an exactly-tied
    group, silently dropping a candidate a correct tie-break would have kept.)

    Args:
        similarity_row: Similarity of the query against each of ``candidate_ids``.
        candidate_ids: Candidate node ids, excluding the query itself.
        k: Number of ids to return (fewer if ``candidate_ids`` is shorter).

    Returns:
        The top ``min(k, len(candidate_ids))`` ids, most-similar first.
    """
    if len(similarity_row) != len(candidate_ids):
        raise ValueError("similarity_row and candidate_ids must be the same length")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n = len(candidate_ids)
    effective_k = min(k, n)
    if effective_k == 0:
        return []
    if effective_k >= n:
        return rank_by_similarity(similarity_row, candidate_ids)
    # argpartition on similarity alone can select the wrong member of a tied
    # group (any of several equal-similarity ids may land on either side of
    # the partition), silently dropping a candidate that `rank_by_similarity`
    # would have kept via its id tie-break. Partitioning on a composite key
    # that already encodes the (-sim, id) order fixes this: ids are bucketed
    # by rank first (dense, so exact float ties share a bucket), and the
    # bucket index -- not the raw similarity -- is what argpartition selects.
    id_rank = np.argsort(candidate_ids)
    id_rank_of = np.empty(n, dtype=np.int64)
    id_rank_of[id_rank] = np.arange(n)
    order = np.lexsort((id_rank_of, -similarity_row))  # exact (-sim, id) order, O(n log n)
    top_order = order[:effective_k]
    return [candidate_ids[i] for i in top_order]


def top_k(ranked_ids: list[str], k: int) -> list[str]:
    """Return the first ``k`` ids of an already-ranked list (fewer if it is shorter)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return ranked_ids[:k]


def normalized_precision_at_k(learned_ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Recall-bounded precision: ``|L_k ∩ R| / min(k, |R|)``.

    Ordinary precision (``/k``) penalizes a method for ``k`` exceeding the
    relevant set's size even when every relevant item was retrieved. Dividing
    by ``min(k, |R|)`` instead means a method that retrieves every relevant
    item scores 1.0 regardless of how much larger ``k`` is than ``|R|``.

    Args:
        learned_ranked_ids: Candidate ids ranked by the method, most similar
            first, query excluded.
        relevant: The relevant set for this query (query excluded), from a
            ``RelevanceProvider``.
        k: Cutoff.

    Returns:
        0.0 if ``k == 0`` or ``relevant`` is empty.
    """
    if k == 0 or not relevant:
        return 0.0
    learned_top_k = set(top_k(learned_ranked_ids, k))
    denom = min(k, len(relevant))
    return len(learned_top_k & relevant) / denom


def recall_at_k(learned_ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Recall: ``|L_k ∩ R| / |R|``.

    Reported alongside :func:`normalized_precision_at_k` rather than folded
    into it, so a view where ``k`` routinely exceeds ``|R|`` (e.g. textual
    top-k relevance at large ``k``) is never reported under a precision label
    it doesn't earn.

    Returns:
        0.0 if ``relevant`` is empty.
    """
    if not relevant:
        return 0.0
    learned_top_k = set(top_k(learned_ranked_ids, k)) if k > 0 else set()
    return len(learned_top_k & relevant) / len(relevant)


def hit_at_k(learned_ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Whether *any* relevant item lands in the method's top-``k``.

    Args:
        learned_ranked_ids: Candidate ids ranked by the method, most similar
            first, query excluded.
        relevant: The relevant set for this query, query excluded.
        k: Cutoff.

    Returns:
        1.0 if ``top_k(learned_ranked_ids, k)`` intersects ``relevant``, else
        0.0. 0.0 if ``relevant`` is empty.
    """
    if not relevant or k == 0:
        return 0.0
    return 1.0 if set(top_k(learned_ranked_ids, k)) & relevant else 0.0


def mrr(learned_ranked_ids: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the *first* relevant item in the full ranking.

    Args:
        learned_ranked_ids: Candidate ids ranked by the method, most similar
            first, query excluded.
        relevant: The relevant set for this query, query excluded.

    Returns:
        ``1 / rank`` (1-indexed) of the first id in ``learned_ranked_ids``
        that is in ``relevant``, or 0.0 if none is (or ``relevant`` is empty).
    """
    if not relevant:
        return 0.0
    for rank, node_id in enumerate(learned_ranked_ids, start=1):
        if node_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr_via_argsort(similarity_row: np.ndarray, candidate_ids: list[str], relevant: set[str]) -> float:
    """Vectorized MRR: exact ``(-sim, node_id)`` tie order via ``np.lexsort``, no Python sort.

    MRR is the one metric that may need a deep rank (the first relevant item
    can be arbitrarily far down the ranking), so unlike the top-k metrics it
    cannot use :func:`top_k_by_similarity`'s partial ``argpartition``. This
    still avoids a full Python-level tuple sort of ``candidate_ids``.

    Args:
        similarity_row: Similarity of the query against each of ``candidate_ids``.
        candidate_ids: Candidate node ids, excluding the query itself, in any
            order -- the id-based tie-break is computed here, not assumed of
            the caller's ordering.
        relevant: The relevant set for this query, query excluded.

    Returns:
        Same value as ``mrr(rank_by_similarity(similarity_row, candidate_ids), relevant)``.
    """
    if not relevant:
        return 0.0
    n = len(candidate_ids)
    # lexsort's last key is primary: -similarity_row first (ascending ==
    # descending similarity), id_rank as the secondary tie-break -- the same
    # (-sim, node_id) order `rank_by_similarity` produces, computed without a
    # Python-level sort of `n` tuples.
    id_rank = np.argsort(candidate_ids)  # candidate_ids[id_rank] is ascending by id
    id_rank_of = np.empty(n, dtype=np.int64)
    id_rank_of[id_rank] = np.arange(n)
    order = np.lexsort((id_rank_of, -similarity_row))
    relevant_mask = np.fromiter((candidate_ids[i] in relevant for i in order), dtype=bool, count=n)
    hit_positions = np.flatnonzero(relevant_mask)
    if hit_positions.size == 0:
        return 0.0
    return 1.0 / (int(hit_positions[0]) + 1)
