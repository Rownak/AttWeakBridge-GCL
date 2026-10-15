"""Registered evaluators: the pluggable unit under ``evaluate.evaluations[]``.

An evaluator owns its own loop and returns ``MetricRow``s (target_architecture.md
§5.5a); the pipeline never learns what shape that loop takes. Its ``params``
block is opaque to the pipeline -- handed verbatim, validated by the
evaluator itself -- which is what makes a new evaluation a new file rather
than a pipeline change (§10a).

Importing this package registers every built-in evaluator as a side effect.
"""

from __future__ import annotations

from typing import Any, Protocol

from cga.core.registry import Registry
from cga.pipelines.evaluate.rows import MetricRow

evaluators: Registry = Registry("evaluator")


class EvalResources(Protocol):
    """Everything an evaluator may need, loaded once by the pipeline's ``load`` step."""

    dataset_id: str
    run_id: str
    methods: dict[str, Any]  # method_id -> (AlignedMatrix, resolved_run_id)
    method_specs: dict[str, Any]  # method_id -> MethodSpec
    graph_role_by_node: dict[str, str]
    graph: Any  # networkx.Graph
    alignments: Any  # pyarrow Table of the ingest run's alignments.parquet, or None if absent
    ctx: Any  # RunContext, for resolving evaluator-declared references
    digest: Any  # MetricsDigest -- evaluators may attach scope provenance to it


class Evaluator(Protocol):
    """One registered evaluation method."""

    def evaluate(self, res: EvalResources, evaluation_id: str, params: dict[str, Any]) -> list[MetricRow]:
        """Run this evaluation over ``res`` and return its scored rows."""
        ...


from cga.pipelines.evaluate.evaluators import cluster_quality, gold_alignment, neighborhood  # noqa: E402,F401
