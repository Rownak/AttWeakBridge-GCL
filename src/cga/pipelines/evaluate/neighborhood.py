"""Build reference neighborhoods and score the learned embedding against them (§5.5a).

Two independent reference views, each restricted to same-``graph_role`` pairs
(a node's neighbors are only ever drawn from its own graph):

- **text**: reference top-k = cosine top-k in the text-embedding space.
- **structural**: reference top-k = cosine top-k in the structural
  (DeepWalk/node2vec) embedding space.

For each view, each ``graph_role``, and each configured ``k``, Precision@k /
Hit@k / MRR are computed per query node and averaged. A node is scored in a
view only if it has a vector in *both* the learned matrix and that view's
reference matrix; nodes missing from a reference store (e.g. no text) are
silently excluded from that view rather than raising, matching
``training_data.similarity``'s existing eligibility convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.metrics import hit_at_k, mrr, precision_at_k, rank_by_similarity, top_k


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the ``(len(a), len(b))`` cosine similarity matrix between two stacks of vectors."""
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


@dataclass
class RoleViewMetrics:
    """Averaged metrics for one (view, graph_role, k) combination.

    Attributes:
        n_queries: Number of nodes actually scored (eligible in both matrices,
            role size >= 2).
        precision_at_k: Mean Precision@k over scored queries.
        hit_at_k: Mean Hit@k over scored queries.
        mrr: Mean reciprocal rank over scored queries.
    """

    n_queries: int
    precision_at_k: float
    hit_at_k: float
    mrr: float


def _role_ids(ids: list[str], graph_role_by_node: dict[str, str], role: str) -> list[str]:
    return [nid for nid in ids if graph_role_by_node.get(nid) == role]


def score_view(
    *,
    learned_matrix: AlignedMatrix,
    reference_matrix: AlignedMatrix,
    graph_role_by_node: dict[str, str],
    role: str,
    ks: list[int],
) -> dict[int, RoleViewMetrics]:
    """Score one reference view for one ``graph_role``, across every ``k``.

    Args:
        learned_matrix: The trained model's exported node embeddings.
        reference_matrix: The text or structural embedding store defining
            "ground truth" neighborhoods for this view.
        graph_role_by_node: ``node_id`` -> ``graph_role`` for every node in the universe.
        role: ``graph_1`` or ``graph_2`` — comparisons never cross roles.
        ks: Cutoffs to report, e.g. ``[5, 10, 20]``.

    Returns:
        ``{k: RoleViewMetrics}`` for every requested ``k``.
    """
    eligible_ids = set(learned_matrix.node_ids) & set(reference_matrix.node_ids)
    role_ids = sorted(_role_ids(list(eligible_ids), graph_role_by_node, role))

    max_k = max(ks) if ks else 0
    results: dict[int, RoleViewMetrics] = {k: RoleViewMetrics(0, 0.0, 0.0, 0.0) for k in ks}
    if len(role_ids) < 2:
        return results

    learned_sub = learned_matrix.select(role_ids)
    reference_sub = reference_matrix.select(role_ids)
    learned_sim = _cosine_similarity(learned_sub.vectors, learned_sub.vectors)
    reference_sim = _cosine_similarity(reference_sub.vectors, reference_sub.vectors)

    per_k_precision: dict[int, list[float]] = {k: [] for k in ks}
    per_k_hit: dict[int, list[float]] = {k: [] for k in ks}
    per_query_mrr: list[float] = []

    for i, query_id in enumerate(role_ids):
        candidate_ids = role_ids[:i] + role_ids[i + 1 :]
        learned_row = np.concatenate([learned_sim[i, :i], learned_sim[i, i + 1 :]])
        reference_row = np.concatenate([reference_sim[i, :i], reference_sim[i, i + 1 :]])

        learned_ranked = rank_by_similarity(learned_row, candidate_ids)
        reference_ranked = rank_by_similarity(reference_row, candidate_ids)
        reference_top = top_k(reference_ranked, min(max_k, len(reference_ranked)))

        per_query_mrr.append(mrr(learned_ranked, reference_top))
        for k in ks:
            effective_k = min(k, len(candidate_ids))
            reference_top_k_for_k = reference_top[:effective_k]
            per_k_precision[k].append(precision_at_k(learned_ranked, reference_top_k_for_k, effective_k))
            per_k_hit[k].append(hit_at_k(learned_ranked, reference_top_k_for_k, effective_k))

    n_queries = len(role_ids)
    for k in ks:
        results[k] = RoleViewMetrics(
            n_queries=n_queries,
            precision_at_k=float(np.mean(per_k_precision[k])),
            hit_at_k=float(np.mean(per_k_hit[k])),
            mrr=float(np.mean(per_query_mrr)),
        )
    return results
