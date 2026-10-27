"""``gold_set_retrieval`` evaluator: set-based top-k retrieval of every gold target of a source (Sprint 11.7).

The gold alignment is many-to-many and unordered: one ATT&CK technique has several correct CWEs
and nothing ranks them. ``gold_alignment`` scores one query per gold *pair* (the other correct
targets are removed from the candidates); this evaluator scores one query per *source* and treats
all of its gold targets as one relevant set. Only order-free top-k metrics are reported:

``hits_at_k``
    1 if any relevant target is in the top ``k``.
``recall_at_k``
    ``|top-k ∩ R| / |R|``.
``precision_at_k``
    ``|top-k ∩ R| / k``.
``precision_ceiling_at_k``
    ``min(|R|, k) / k``, the best ``precision_at_k`` the query allows. A source with one gold
    target can score at most ``1 / k``, so read precision next to this ceiling.

Each metric is a macro-average over sources. Candidates are every ``graph_2`` node shared by all
methods (optionally restricted to a ``target_pool``); there is no sibling filter because all gold
targets are relevant together. A source is a query only when it is present in every method and has
at least one gold target among the candidates, so all methods share one query set.

``gold_alignment`` and ``gold_alignment_prf`` are unchanged, so earlier runs stay comparable.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np

from cga.core.errors import ConfigError
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.evaluators.gold_alignment import _alignment_rows, _normalize
from cga.pipelines.evaluate.evaluators.target_pool import check_targets_in_pool, load_target_pool
from cga.pipelines.evaluate.metrics import (
    hit_at_k,
    precision_at_k,
    precision_ceiling_at_k,
    recall_at_k,
    top_k_by_similarity,
)
from cga.pipelines.evaluate.rows import MetricRow

_SOURCE_ROLE = "graph_1"
_TARGET_ROLE = "graph_2"


@evaluators.register("gold_set_retrieval")
class GoldSetRetrievalEvaluator:
    """Scores every configured method by Hits / Recall / Precision at k per gold source.

    Params:
        ks: Cutoffs, e.g. ``[1, 5, 10, 15, 20, 50]`` (required, each >= 1).
        split: Gold alignment split to query (default ``"test"``).
        target_pool: Path of a pool file (``scripts/make_target_pool.py``); when set, candidates
            are restricted to the pool (default: off). A gold target outside the pool raises
            ``ConfigError``.
    """

    def evaluate(self, res: Any, evaluation_id: str, params: dict[str, Any]) -> list[MetricRow]:
        """Emit ``hits_at_k``, ``recall_at_k``, ``precision_at_k`` and ``precision_ceiling_at_k``.

        Args:
            res: ``EvalResources``; uses ``methods``, ``method_specs``, ``graph_role_by_node``,
                ``alignments`` and ``digest``.
            evaluation_id: Config-supplied instance name.
            params: See the class docstring.

        Returns:
            One row per metric and ``k`` per method, ``view_id = "global"``, ``param_name = "k"``
            and ``n_queries`` = number of source nodes scored.

        Raises:
            ConfigError: On an empty or invalid ``ks``, when the split has no gold pairs, or when
                no source is scorable.
        """
        ks: list[int] = list(params.get("ks") or [])
        split: str = params.get("split", "test")
        if not ks or any((not isinstance(k, int)) or k < 1 for k in ks):
            raise ConfigError(f"gold_set_retrieval evaluation '{evaluation_id}' needs a non-empty ks of ints >= 1")

        split_rows = [r for r in _alignment_rows(getattr(res, "alignments", None)) if r["split"] == split]
        if not split_rows:
            raise ConfigError(
                f"gold_set_retrieval evaluation '{evaluation_id}': no '{split}' alignment pairs "
                "(does the dataset config enable `alignments`?)"
            )

        method_ids = sorted(res.methods)
        shared = set.intersection(*(set(res.methods[m][0].node_ids) for m in method_ids))
        role_of = res.graph_role_by_node
        pool_ids = sorted(n for n in shared if role_of.get(n) == _TARGET_ROLE)  # sorted: ties break by id
        target_pool_sha = None
        if params.get("target_pool"):
            pool_set, target_pool_sha = load_target_pool(params["target_pool"], evaluation_id=evaluation_id)
            check_targets_in_pool({r["node_2_id"] for r in split_rows}, pool_set, evaluation_id=evaluation_id)
            pool_ids = [n for n in pool_ids if n in pool_set]
        pool_index = set(pool_ids)

        gold_of: dict[str, set[str]] = defaultdict(set)
        for r in split_rows:
            gold_of[r["node_1_id"]].add(r["node_2_id"])
        queries: dict[str, set[str]] = {}
        n_dropped = 0
        for source, targets in gold_of.items():
            relevant = targets & pool_index
            if source not in shared or role_of.get(source) != _SOURCE_ROLE or not relevant:
                n_dropped += 1
                continue
            queries[source] = relevant
        if not queries:
            raise ConfigError(f"gold_set_retrieval evaluation '{evaluation_id}': no scorable source in '{split}'")
        sources = sorted(queries)
        sizes = np.array([len(queries[s]) for s in sources])

        rows: list[MetricRow] = []
        for method_id in method_ids:
            matrix, method_run_id = res.methods[method_id]
            spec = res.method_specs[method_id]
            sim = _normalize(matrix.select(sources).vectors) @ _normalize(matrix.select(pool_ids).vectors).T
            max_k = max(ks)
            sums: dict[str, dict[int, float]] = {
                m: dict.fromkeys(ks, 0.0) for m in ("hits_at_k", "recall_at_k", "precision_at_k")
            }
            for i, source in enumerate(sources):
                ranked = top_k_by_similarity(sim[i], pool_ids, max_k)
                for k in ks:
                    sums["hits_at_k"][k] += hit_at_k(ranked[:k], queries[source], k)
                    sums["recall_at_k"][k] += recall_at_k(ranked, queries[source], k)
                    sums["precision_at_k"][k] += precision_at_k(ranked, queries[source], k)
            values = {m: {k: v / len(sources) for k, v in per_k.items()} for m, per_k in sums.items()}
            values["precision_ceiling_at_k"] = {
                k: float(np.mean([precision_ceiling_at_k(int(n), k) for n in sizes])) for k in ks
            }
            for metric_name, per_k in values.items():
                for k in ks:
                    rows.append(
                        MetricRow(
                            evaluation_id=evaluation_id,
                            evaluation_kind="gold_set_retrieval",
                            method_id=method_id,
                            method_label=spec.label,
                            method_kind=spec.kind,
                            method_run_id=method_run_id,
                            graph_role=_SOURCE_ROLE,
                            view_id="global",
                            param_name="k",
                            param_value=k,
                            metric_name=metric_name,
                            metric_value=float(per_k[k]),
                            n_queries=len(sources),
                        )
                    )

        res.digest.add_scope_provenance(
            evaluation_id=evaluation_id,
            view_id="global",
            graph_role=_SOURCE_ROLE,
            query_set_sha256=hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest(),
            split=split,
            n_eligible_queries=len(sources),
            n_dropped_missing_in_a_method=n_dropped,
            n_candidates=len(pool_ids),
            n_relevant_mean=float(sizes.mean()),
            n_relevant_median=float(np.median(sizes)),
            n_relevant_max=int(sizes.max()),
            target_pool_sha256=target_pool_sha,
            n_target_pool=len(pool_ids) if target_pool_sha else None,
        )
        return rows
