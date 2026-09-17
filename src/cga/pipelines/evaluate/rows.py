"""``MetricRow``: the unit every registered evaluator emits (target_architecture.md §6.10).

A row is one ``(evaluation, method, scope, sweep point, metric)`` value. This
is the only currency an evaluator deals in -- the pipeline never inspects an
evaluator's internal loop shape, only collects the rows it returns. Writing
``metrics.parquet`` and deriving the ``metrics.json`` digest from the *same*
rows (rather than computing each independently) is what keeps the two
artifacts from silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

from cga.core.io import write_json, write_parquet
from cga.data.schemas import METRICS_SCHEMA


@dataclass(frozen=True)
class MetricRow:
    """One scored value: identifies its evaluation, method, scope, and metric.

    Attributes:
        evaluation_id: Config-supplied instance name (unique per run) --
            distinguishes two instances of the same ``evaluation_kind`` run
            with different params (e.g. a ``top_k`` sensitivity sweep).
        evaluation_kind: The registered evaluator name (e.g. "neighborhood").
        method_id: The method (baseline or trained model) this row scores.
        method_label: Human-readable label for figures/tables.
        method_kind: ``embedding_store`` or ``gcl_run``.
        method_run_id: The resolved upstream run_id -- per-row lineage.
        graph_role: ``graph_1``/``graph_2``, or None for a cross-role evaluation.
        view_id: The evaluator's view/reference id, or None if it has none.
        param_name: The sweep axis name (e.g. "k", "n_clusters"), or None.
        param_value: The sweep axis value, or None.
        metric_name: e.g. "normalized_precision_at_k", "silhouette", "mrr".
        metric_value: The scored value.
        n_queries: Population this value was averaged over, or None.
    """

    evaluation_id: str
    evaluation_kind: str
    method_id: str
    method_label: str
    method_kind: str
    method_run_id: str
    metric_name: str
    metric_value: float
    graph_role: str | None = None
    view_id: str | None = None
    param_name: str | None = None
    param_value: int | None = None
    n_queries: int | None = None


@dataclass
class MetricsDigest:
    """Nested, human-readable view derived from a run's ``MetricRow``s.

    Shaped ``evaluation -> view -> role -> method -> param -> metric``, plus
    per-(evaluation, view, role) provenance (``query_set_sha256`` etc.) an
    evaluator may attach via :meth:`add_scope_provenance`. Never computed
    independently of the rows that produced it.
    """

    rows: list[MetricRow] = field(default_factory=list)
    _provenance: dict[tuple[str, str | None, str | None], dict[str, Any]] = field(default_factory=dict)

    def add_scope_provenance(
        self, *, evaluation_id: str, view_id: str | None, graph_role: str | None, **fields: Any
    ) -> None:
        """Attach provenance (e.g. ``query_set_sha256``) to one (evaluation, view, role) scope."""
        self._provenance[(evaluation_id, view_id, graph_role)] = dict(fields)

    def to_dict(self) -> dict[str, Any]:
        """Build the nested digest dict from ``self.rows`` and any attached provenance."""
        digest: dict[str, Any] = {"schema_version": 2, "evaluations": {}}
        evaluations = digest["evaluations"]

        for row in self.rows:
            ev = evaluations.setdefault(row.evaluation_id, {"kind": row.evaluation_kind, "scopes": {}})
            scope_key = f"{row.view_id or '_'}::{row.graph_role or '_'}"
            scope = ev["scopes"].setdefault(
                scope_key, {"view_id": row.view_id, "graph_role": row.graph_role, "methods": {}}
            )
            method = scope["methods"].setdefault(row.method_id, {"label": row.method_label, "params": {}})
            param_key = str(row.param_value) if row.param_name else "_"
            param_bucket = method["params"].setdefault(param_key, {})
            param_bucket[row.metric_name] = row.metric_value
            if row.n_queries is not None:
                scope.setdefault("n_queries", row.n_queries)

        for (evaluation_id, view_id, graph_role), fields_ in self._provenance.items():
            ev = evaluations.get(evaluation_id)
            if ev is None:
                continue
            scope_key = f"{view_id or '_'}::{graph_role or '_'}"
            scope = ev["scopes"].get(scope_key)
            if scope is not None:
                scope.update(fields_)

        return digest


def rows_to_table(dataset_id: str, run_id: str, rows: list[MetricRow]) -> pa.Table:
    """Build the ``METRICS_SCHEMA``-conformant table for this run's rows."""
    records = [
        {
            "dataset_id": dataset_id,
            "run_id": run_id,
            "evaluation_id": r.evaluation_id,
            "evaluation_kind": r.evaluation_kind,
            "method_id": r.method_id,
            "method_label": r.method_label,
            "method_kind": r.method_kind,
            "method_run_id": r.method_run_id,
            "graph_role": r.graph_role,
            "view_id": r.view_id,
            "param_name": r.param_name,
            "param_value": r.param_value,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "n_queries": r.n_queries,
        }
        for r in rows
    ]
    return pa.Table.from_pylist(records, schema=METRICS_SCHEMA)


def write_metrics(
    dataset_id: str, run_id: str, digest: MetricsDigest, *, parquet_path: Path, json_path: Path
) -> None:
    """Write ``metrics.parquet`` (the machine contract) and ``metrics.json`` (its digest).

    Both are derived from ``digest.rows`` in this one call, so they cannot
    report different numbers for the same run.
    """
    table = rows_to_table(dataset_id, run_id, digest.rows)
    write_parquet(table, parquet_path, schema=METRICS_SCHEMA)
    write_json(digest.to_dict(), json_path)
