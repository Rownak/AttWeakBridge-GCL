"""``gold_alignment_rerank`` evaluator: ``gold_alignment`` after an anchor-based re-ranker (Sprint 6.2).

Scores the same queries, candidate lists, metrics and modes as :mod:`gold_alignment`, but for a
*re-ranked* method: a first-stage method's top-K is reordered by a fitted listwise linear model on
anchor-consistency features (:mod:`cga.pipelines.evaluate.rerank`). The first-stage method, the
methods that supply the score features, and the fitted re-ranker file are named in ``params``; the
evaluator reads but does not change ``gold_alignment``. Anchors are gold train pairs only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cga.core.errors import ConfigError
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.evaluators.gold_alignment import _query_set_sha256
from cga.pipelines.evaluate.rerank import (
    SCORE_SOURCES,
    build_anchors,
    featurize_split,
    gold_rank_after_rerank,
    load_reranker,
)
from cga.pipelines.evaluate.rows import MetricRow

_MODES = ("global", "candidates")


@evaluators.register("gold_alignment_rerank")
class GoldAlignmentRerankEvaluator:
    """Hits@K / MRR of a first stage's top-K reordered by an anchor-based re-ranker.

    Params:
        name: Method id of the emitted rows (e.g. ``A21``).
        first_stage: ``method_id`` of the first-stage method (in ``evaluate.methods``).
        score_methods: ``{sapbert, a11, a12: method_id}`` supplying the cosine features.
        reranker: Path of the JSON written by ``scripts/srga_rerank_fit.py``.
        ks: Hits@K cutoffs.
        split: Gold split to query (default ``test``).
        modes: Subset of ``["global", "candidates"]`` (default both).
    """

    def evaluate(self, res: Any, evaluation_id: str, params: dict[str, Any]) -> list[MetricRow]:
        """Emit ``hits_at_k`` and ``mrr`` rows for the re-ranked method per mode.

        Raises:
            ConfigError: On missing params, unknown method ids, an unreadable re-ranker file, or
                a dataset without alignments / edges.
        """
        ks = list(params.get("ks") or [])
        split = params.get("split", "test")
        modes = list(params.get("modes", _MODES))
        name = params.get("name")
        if not ks or any(k < 1 for k in ks) or not name:
            raise ConfigError(f"gold_alignment_rerank '{evaluation_id}' needs `name` and a non-empty ks")
        if not modes or any(m not in _MODES for m in modes):
            raise ConfigError(f"gold_alignment_rerank '{evaluation_id}' has invalid modes {modes}")
        wanted = {"first": params.get("first_stage"), **(params.get("score_methods") or {})}
        if any(wanted.get(s) is None for s in SCORE_SOURCES):
            raise ConfigError(f"gold_alignment_rerank '{evaluation_id}' needs first_stage and score_methods {SCORE_SOURCES[:3]}")
        unknown = [m for m in wanted.values() if m not in res.methods]
        if unknown:
            raise ConfigError(f"gold_alignment_rerank '{evaluation_id}': methods {unknown} are not in evaluate.methods")
        alignments = getattr(res, "alignments", None)
        edges = getattr(res, "edges", None)
        if alignments is None or edges is None:
            raise ConfigError("gold_alignment_rerank needs the ingest alignments and edges")
        model = load_reranker(params["reranker"])
        matrices = {s: res.methods[wanted[s]][0] for s in SCORE_SOURCES}
        first_run_id = res.methods[wanted["first"]][1]
        align_rows = alignments.to_pylist()
        anchors = build_anchors(align_rows, res.graph_role_by_node, shuffle_seed=model.get("shuffle_seed"))

        rows: list[MetricRow] = []
        for mode in modes:
            feats = featurize_split(
                matrices, res.graph_role_by_node, align_rows, edges, split, mode, model["groups"], model["top_k"], anchors
            )
            ranks = np.array([gold_rank_after_rerank(q, model) for q in feats])
            common = dict(
                evaluation_id=evaluation_id,
                evaluation_kind="gold_alignment_rerank",
                method_id=name,
                method_label=params.get("label", name),
                method_kind="rerank",
                method_run_id=f"{first_run_id}+rerank:{name}",
                graph_role="graph_1",
                view_id=mode,
                n_queries=len(feats),
            )
            for k in ks:
                rows.append(
                    MetricRow(**common, param_name="k", param_value=k, metric_name="hits_at_k", metric_value=float(np.mean(ranks <= k)))
                )
            rows.append(MetricRow(**common, metric_name="mrr", metric_value=float(np.mean(1.0 / ranks))))
            res.digest.add_scope_provenance(
                evaluation_id=evaluation_id,
                view_id=mode,
                graph_role="graph_1",
                query_set_sha256=_query_set_sha256([(q.source, q.target) for q in feats]),
                split=split,
                n_eligible_queries=len(feats),
                n_dropped_missing_in_a_method=0,
                n_skipped_no_candidate_pool=0,
            )
        return rows
