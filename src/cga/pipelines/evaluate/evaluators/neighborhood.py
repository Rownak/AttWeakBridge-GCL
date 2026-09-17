"""``neighborhood`` evaluator: methods x views x graph_role x k (target_architecture.md §5.5a).

Asks *does the learned space preserve neighborhoods that already exist* in a
declared reference view (structural hop-radius, or textual cosine top-k),
scored for every configured method so they land in one comparable table.

Eligibility, per (view, role), in this exact order:

    candidates = sorted(role_ids ∩ (∩ over methods of method.node_ids))
    provider.prepare(candidate_ids=candidates)
    queries    = sorted(candidates ∩ provider.defined_for())

Two correctness points, each covered by a dedicated test:

1. ``prepare`` receives the **method-intersected** candidates, not raw role
   ids -- otherwise a provider (e.g. hop-radius) could count as relevant a
   neighbor that no method can even score.
2. The intersection applies to the **candidate pool**, not just the query
   set. If method B lacks node ``x``, ``x`` must leave *every* method's
   pool -- otherwise method A ranks against a larger candidate pool than
   method B, and precision becomes pool-size-sensitive rather than
   comparable across methods.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from cga.core.errors import ConfigError
from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.metrics import (
    hit_at_k,
    mrr_via_argsort,
    normalized_precision_at_k,
    recall_at_k,
    top_k_by_similarity,
)
from cga.pipelines.evaluate.relevance import relevance_providers
from cga.pipelines.evaluate.rows import MetricRow


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def _role_ids(ids: list[str], graph_role_by_node: dict[str, str], role: str) -> list[str]:
    return [nid for nid in ids if graph_role_by_node.get(nid) == role]


def _query_set_sha256(query_ids: list[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(query_ids)).encode("utf-8"))
    return digest.hexdigest()


def _build_provider(view_cfg: dict[str, Any], *, graph: Any, ctx: Any) -> Any:
    provider_kind = view_cfg["provider"]

    def load_reference(reference_cfg: dict[str, Any]) -> AlignedMatrix:
        from cga.pipelines.evaluate.methods import MethodSpec, load_method

        spec = MethodSpec.from_config({"method_id": f"ref:{view_cfg['view_id']}", **reference_cfg})
        matrix, _ = load_method(ctx, spec)
        return matrix

    factory = relevance_providers.get(provider_kind)
    return factory(view_cfg.get("params", {}), graph=graph, load_reference=load_reference)


@evaluators.register("neighborhood")
class NeighborhoodEvaluator:
    """Scores every configured method against every configured view, per graph_role and k."""

    def evaluate(self, res: Any, evaluation_id: str, params: dict[str, Any]) -> list[MetricRow]:
        ks: list[int] = params["ks"]
        roles: list[str] = params.get("roles", ["graph_1", "graph_2"])
        views: list[dict[str, Any]] = params["views"]
        if not ks:
            raise ConfigError(f"neighborhood evaluation '{evaluation_id}' has an empty ks list")
        if not views:
            raise ConfigError(f"neighborhood evaluation '{evaluation_id}' has no views configured")

        max_k = max(ks)
        rows: list[MetricRow] = []
        method_ids = sorted(res.methods.keys())
        all_method_node_ids = [set(res.methods[mid][0].node_ids) for mid in method_ids]
        method_intersection = set.intersection(*all_method_node_ids) if all_method_node_ids else set()

        for view_cfg in views:
            view_id = view_cfg["view_id"]
            provider = _build_provider(view_cfg, graph=res.graph, ctx=res.ctx)

            for role in roles:
                candidates = sorted(_role_ids(list(method_intersection), res.graph_role_by_node, role))
                provider.prepare(candidate_ids=candidates)
                queries = sorted(set(candidates) & provider.defined_for())

                n_queries = len(queries)
                query_hash = _query_set_sha256(queries)

                for method_id in method_ids:
                    matrix, method_run_id = res.methods[method_id]
                    spec = res.method_specs[method_id]

                    per_k_norm_prec: dict[int, list[float]] = {k: [] for k in ks}
                    per_k_recall: dict[int, list[float]] = {k: [] for k in ks}
                    per_k_hit: dict[int, list[float]] = {k: [] for k in ks}
                    mrr_values: list[float] = []

                    if n_queries >= 1 and len(candidates) >= 2:
                        sub = matrix.select(candidates)
                        sim = _cosine_similarity(sub.vectors, sub.vectors)
                        index_of = {nid: i for i, nid in enumerate(candidates)}

                        for query_id in queries:
                            i = index_of[query_id]
                            others = candidates[:i] + candidates[i + 1 :]
                            row = np.concatenate([sim[i, :i], sim[i, i + 1 :]])
                            relevant = provider.relevant(query_id) - {query_id}
                            relevant &= set(others)

                            learned_top = top_k_by_similarity(row, others, max_k)
                            mrr_values.append(mrr_via_argsort(row, others, relevant))
                            for k in ks:
                                effective_k = min(k, len(others))
                                top_for_k = learned_top[:effective_k]
                                per_k_norm_prec[k].append(
                                    normalized_precision_at_k(top_for_k, relevant, effective_k)
                                )
                                per_k_recall[k].append(recall_at_k(top_for_k, relevant, effective_k))
                                per_k_hit[k].append(hit_at_k(top_for_k, relevant, effective_k))

                    for k in ks:
                        common = dict(
                            evaluation_id=evaluation_id,
                            evaluation_kind="neighborhood",
                            method_id=method_id,
                            method_label=spec.label,
                            method_kind=spec.kind,
                            method_run_id=method_run_id,
                            graph_role=role,
                            view_id=view_id,
                            param_name="k",
                            param_value=k,
                            n_queries=n_queries,
                        )
                        rows.append(
                            MetricRow(
                                **common,
                                metric_name="normalized_precision_at_k",
                                metric_value=float(np.mean(per_k_norm_prec[k])) if n_queries else 0.0,
                            )
                        )
                        rows.append(
                            MetricRow(
                                **common,
                                metric_name="recall_at_k",
                                metric_value=float(np.mean(per_k_recall[k])) if n_queries else 0.0,
                            )
                        )
                        rows.append(
                            MetricRow(
                                **common,
                                metric_name="hit_at_k",
                                metric_value=float(np.mean(per_k_hit[k])) if n_queries else 0.0,
                            )
                        )
                        rows.append(
                            MetricRow(
                                **common,
                                metric_name="mrr",
                                metric_value=float(np.mean(mrr_values)) if n_queries else 0.0,
                            )
                        )

                res.digest.add_scope_provenance(
                    evaluation_id=evaluation_id,
                    view_id=view_id,
                    graph_role=role,
                    query_set_sha256=query_hash,
                    n_eligible_queries=n_queries,
                    n_excluded_undefined=len(candidates) - n_queries,
                )

        return rows
