"""``gold_alignment_prf`` evaluator: BioGITOM-style precision / recall / F1 of a predicted pair set (Sprint 8).

``gold_alignment`` ranks, per gold pair, the gold target among all candidates (MRR, Hits@k).
BioGITOM instead reports P / R / F1 of a *set* of predicted ``(source, target)`` pairs: top-k per
source, optional filters, then a greedy one-to-one assignment (``ncit2doid.py``,
``evaluate_predictions``). This evaluator turns any method's embeddings into that pair set and
scores it against the gold pairs of one split, so our variants can sit next to BioGITOM's numbers.
It reads only the aligned matrix, so every method kind (embedding store, ``gcl_run``, ``fusion``)
works and no model is trained.

Protocol (see ``dev_docs/sprints/sprint_8.md``):

1. Cosine of every source of the split against every ``graph_2`` node (global search; the
   ``candidates`` pool is not used).
2. Top-``k`` targets per source, ties broken by node id.
3. Optional ``biogitom_filters``: drop a row whose source or target is a train-only node (in the
   train pairs but not in the scored split's gold pairs), and keep only sources of the split.
4. ``greedy_one_to_one`` (rows sorted by score, each endpoint used once, ``score >= threshold``) or
   ``top1`` (best remaining row per source, no assignment; BioGITOM's ``evaluate_topk``).
5. ``precision = correct / |predicted|``, ``recall = correct / |gold|``, ``f1`` their harmonic mean.

The helpers are module-level so analysis scripts (threshold sweep) reuse the exact same steps.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from cga.core.errors import ConfigError
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.evaluators.gold_alignment import _alignment_rows, _normalize
from cga.pipelines.evaluate.rows import MetricRow

_ASSIGNMENTS = ("greedy_one_to_one", "top1")
_SOURCE_ROLE = "graph_1"
_TARGET_ROLE = "graph_2"

Candidate = tuple[float, str, str]  # (score, source_id, target_id)


def top_k_candidates(
    src_vecs: np.ndarray, pool_vecs: np.ndarray, sources: list[str], pool_ids: list[str], top_k: int
) -> list[Candidate]:
    """Return the top-``top_k`` cosine rows per source.

    Args:
        src_vecs: ``(n_sources, d)`` source vectors.
        pool_vecs: ``(n_pool, d)`` target vectors.
        sources: Source ids aligned with ``src_vecs``.
        pool_ids: Target ids aligned with ``pool_vecs``; must be sorted so the stable sort breaks
            score ties by node id.
        top_k: Rows kept per source.

    Returns:
        ``(score, source, target)`` rows, best first within each source.
    """
    sim = _normalize(src_vecs) @ _normalize(pool_vecs).T
    k = min(top_k, sim.shape[1])
    order = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return [(float(sim[i, j]), sources[i], pool_ids[j]) for i in range(len(sources)) for j in order[i]]


def apply_filters(
    rows: list[Candidate], excluded_nodes: set[str], split_sources: set[str]
) -> list[Candidate]:
    """BioGITOM's pre-filters: drop train-only endpoints, keep only sources of the scored split."""
    return [
        r for r in rows if r[1] in split_sources and r[1] not in excluded_nodes and r[2] not in excluded_nodes
    ]


def assign_pairs(rows: list[Candidate], assignment: str, threshold: float) -> list[tuple[str, str]]:
    """Turn candidate rows into the predicted pair set.

    Args:
        rows: ``(score, source, target)`` rows (any order).
        assignment: ``greedy_one_to_one`` or ``top1``.
        threshold: Minimum score; a row below it is never predicted.

    Returns:
        Predicted ``(source, target)`` pairs. Ordering is by ``(-score, source, target)``, so the
        greedy result does not depend on input order.
    """
    ordered = sorted(rows, key=lambda r: (-r[0], r[1], r[2]))
    pairs: list[tuple[str, str]] = []
    used_s: set[str] = set()
    used_t: set[str] = set()
    for score, s, t in ordered:
        if score < threshold:
            break
        if s in used_s or (assignment == "greedy_one_to_one" and t in used_t):
            continue
        used_s.add(s)
        used_t.add(t)
        pairs.append((s, t))
    return pairs


def score_pairs(predicted: list[tuple[str, str]], gold: set[tuple[str, str]]) -> dict[str, float]:
    """Return precision, recall, F1 and the counts of a predicted pair set against ``gold``."""
    n_correct = sum(1 for p in predicted if p in gold)
    precision = n_correct / len(predicted) if predicted else 0.0
    recall = n_correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": float(len(predicted)),
        "n_correct": float(n_correct),
    }


def _gold_sets(
    alignment_rows: list[dict[str, Any]], split: str
) -> tuple[set[tuple[str, str]], set[str]]:
    """Return the split's de-duplicated gold pairs and the nodes the train pairs touch."""
    gold = {(r["node_1_id"], r["node_2_id"]) for r in alignment_rows if r["split"] == split}
    train_nodes = {n for r in alignment_rows if r["split"] == "train" for n in (r["node_1_id"], r["node_2_id"])}
    return gold, train_nodes


@evaluators.register("gold_alignment_prf")
class GoldAlignmentPRFEvaluator:
    """Scores every configured method by P / R / F1 of a BioGITOM-style predicted pair set.

    Params:
        split: Gold alignment split to score (``"test"`` by default, or ``"val"``).
        top_k: Candidates kept per source before filtering (default 3).
        threshold: Minimum cosine for a prediction (default 0.0).
        biogitom_filters: Apply the train-only-endpoint and split-source filters (default false).
        assignment: ``greedy_one_to_one`` (default) or ``top1``.
    """

    def evaluate(self, res: Any, evaluation_id: str, params: dict[str, Any]) -> list[MetricRow]:
        """Emit ``precision``, ``recall``, ``f1`` and count rows per method.

        Args:
            res: ``EvalResources``; uses ``methods``, ``method_specs``, ``graph_role_by_node``,
                ``alignments`` and ``digest``.
            evaluation_id: Config-supplied instance name.
            params: See the class docstring.

        Returns:
            One row per metric per method, with ``view_id`` = the assignment. Besides P/R/F1:
            ``n_predicted``, ``n_correct``, ``n_gold`` (all split pairs, including those a method
            cannot score), ``n_unique_sources`` and ``n_unique_targets`` (the recall ceiling of a
            one-to-one assignment).

        Raises:
            ConfigError: On an invalid option or when the split has no gold pairs.
        """
        split: str = params.get("split", "test")
        top_k = int(params.get("top_k", 3))
        threshold = float(params.get("threshold", 0.0))
        filters = bool(params.get("biogitom_filters", False))
        assignment: str = params.get("assignment", "greedy_one_to_one")
        if top_k < 1:
            raise ConfigError(f"gold_alignment_prf evaluation '{evaluation_id}' needs top_k >= 1")
        if assignment not in _ASSIGNMENTS:
            raise ConfigError(
                f"gold_alignment_prf evaluation '{evaluation_id}' has invalid assignment {assignment!r}; "
                f"known: {list(_ASSIGNMENTS)}"
            )

        alignment_rows = _alignment_rows(getattr(res, "alignments", None))
        gold, train_nodes = _gold_sets(alignment_rows, split)
        if not gold:
            raise ConfigError(
                f"gold_alignment_prf evaluation '{evaluation_id}': no '{split}' alignment pairs "
                "(does the dataset config enable `alignments`?)"
            )
        split_sources = {s for s, _ in gold}
        split_nodes = {n for pair in gold for n in pair}
        excluded = train_nodes - split_nodes
        role_of = res.graph_role_by_node
        digest_hash = hashlib.sha256("\n".join(f"{s}|{t}" for s, t in sorted(gold)).encode("utf-8")).hexdigest()

        rows: list[MetricRow] = []
        for method_id in sorted(res.methods):
            matrix, method_run_id = res.methods[method_id]
            spec = res.method_specs[method_id]
            have = set(matrix.node_ids)
            pool_ids = sorted(n for n in have if role_of.get(n) == _TARGET_ROLE)
            sources = sorted(s for s in split_sources if s in have and role_of.get(s) == _SOURCE_ROLE)
            candidates = top_k_candidates(
                matrix.select(sources).vectors, matrix.select(pool_ids).vectors, sources, pool_ids, top_k
            )
            if filters:
                candidates = apply_filters(candidates, excluded, split_sources)
            predicted = assign_pairs(candidates, assignment, threshold)
            metrics = score_pairs(predicted, gold)
            metrics["n_gold"] = float(len(gold))
            metrics["n_unique_sources"] = float(len(split_sources))
            metrics["n_unique_targets"] = float(len({t for _, t in gold}))
            for name, value in metrics.items():
                rows.append(
                    MetricRow(
                        evaluation_id=evaluation_id,
                        evaluation_kind="gold_alignment_prf",
                        method_id=method_id,
                        method_label=spec.label,
                        method_kind=spec.kind,
                        method_run_id=method_run_id,
                        graph_role=_SOURCE_ROLE,
                        view_id=assignment,
                        metric_name=name,
                        metric_value=float(value),
                        n_queries=len(gold),
                    )
                )

        res.digest.add_scope_provenance(
            evaluation_id=evaluation_id,
            view_id=assignment,
            graph_role=_SOURCE_ROLE,
            gold_set_sha256=digest_hash,
            split=split,
            top_k=top_k,
            threshold=threshold,
            biogitom_filters=filters,
            n_excluded_train_only_nodes=len(excluded),
        )
        return rows
