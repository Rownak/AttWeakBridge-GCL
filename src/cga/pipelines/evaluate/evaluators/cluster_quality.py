"""``cluster_quality`` evaluator: methods x graph_role x n_clusters (target_architecture.md §5.5a, §8.4).

Silhouette score of each method's embedding space, clustered with k-means at
each configured ``n_clusters``. Closes the §8.4 diagnostic that Phase 1.8
never implemented, and -- structurally -- is the second registered
evaluator, proving ``evaluators`` is an actual extension point rather than a
one-off abstraction built around a single user.

Unlike ``neighborhood``, this evaluator has no view and no relevant set: it
scores each method's own geometry in isolation, so ``view_id`` is always
``None`` on its rows and it needs no ``RelevanceProvider``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cga.core.errors import ConfigError
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.rows import MetricRow


def _role_ids(ids: list[str], graph_role_by_node: dict[str, str], role: str) -> list[str]:
    return [nid for nid in ids if graph_role_by_node.get(nid) == role]


@evaluators.register("cluster_quality")
class ClusterQualityEvaluator:
    """Scores each method's embedding space by k-means silhouette, per graph_role."""

    def evaluate(self, res: Any, evaluation_id: str, params: dict[str, Any]) -> list[MetricRow]:
        roles: list[str] = params.get("roles", ["graph_1", "graph_2"])
        n_clusters_list: list[int] = params["n_clusters"]
        seed: int = params.get("seed", 42)
        if not n_clusters_list:
            raise ConfigError(f"cluster_quality evaluation '{evaluation_id}' has an empty n_clusters list")

        rows: list[MetricRow] = []
        for method_id, (matrix, method_run_id) in sorted(res.methods.items()):
            spec = res.method_specs[method_id]
            for role in roles:
                role_ids = sorted(set(_role_ids(matrix.node_ids, res.graph_role_by_node, role)))
                sub = matrix.select(role_ids) if role_ids else None

                for n_clusters in n_clusters_list:
                    common = dict(
                        evaluation_id=evaluation_id,
                        evaluation_kind="cluster_quality",
                        method_id=method_id,
                        method_label=spec.label,
                        method_kind=spec.kind,
                        method_run_id=method_run_id,
                        graph_role=role,
                        view_id=None,
                        param_name="n_clusters",
                        param_value=n_clusters,
                        n_queries=len(role_ids),
                    )
                    # silhouette_score requires 2 <= n_clusters < n_samples.
                    if sub is None or n_clusters < 2 or n_clusters >= len(role_ids):
                        score = 0.0
                    else:
                        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
                            sub.vectors
                        )
                        if len(set(labels)) < 2:
                            score = 0.0
                        else:
                            score = float(silhouette_score(sub.vectors, labels))
                    rows.append(MetricRow(**common, metric_name="silhouette", metric_value=score))

        return rows
