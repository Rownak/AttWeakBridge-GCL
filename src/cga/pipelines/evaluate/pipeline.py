"""``evaluate`` pipeline: comparative evaluation over embedding methods (target_architecture.md §5.5).

Scores several embedding methods side by side under one or more registered
**evaluations**. A method is any per-node ``AlignedMatrix`` -- a baseline
embedding store or a trained model's export -- so baselines and trained
models are compared by the same code on the same node universe.

Four steps, none of which know any evaluation kind: ``load`` (validate config
eagerly, resolve ``ingest``, build the graph, load every method and
evaluator-declared reference) -> ``evaluate`` (registry lookup per
``evaluations[]`` entry, collect ``MetricRow``s -- the pipeline's only loop)
-> ``visualize`` (optional t-SNE per method) -> ``validate_and_manifest``
(``metrics.parquet`` + derived ``metrics.json``, ``LATEST``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cga.core.artifacts import ArtifactRef, write_latest
from cga.core.errors import ConfigError
from cga.core.pipeline import Pipeline, Step
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix
from cga.data.graph_store import read_alignments, read_edges, read_nodes
from cga.pipelines.embed.graph_embedders import build_graph
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.methods import MethodSpec, load_method
from cga.pipelines.evaluate.rows import MetricRow, MetricsDigest, write_metrics
from cga.pipelines.evaluate.visualize import plot_tsne_by_role


class _EvaluateState:
    """Mutable state threaded across this run's steps (one instance per run).

    Doubles as the concrete ``EvalResources`` object passed to every
    evaluator -- evaluators only read from it, never mutate it except via
    ``digest.add_scope_provenance``.
    """

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.dataset_id = ctx.dataset_id
        self.run_id = ctx.run_id
        self.methods: dict[str, tuple[AlignedMatrix, str]] = {}
        self.method_specs: dict[str, MethodSpec] = {}
        self.graph_role_by_node: dict[str, str] = {}
        self.graph: Any = None
        self.alignments: Any = None
        self.edges: list[dict[str, Any]] = []  # directed edge rows (additive; ``graph`` stays undirected)
        self.digest = MetricsDigest()
        self.rows: list[MetricRow] = []
        self.ingest_run_id: str = ""


def _validate_config(cfg: dict[str, Any]) -> None:
    """Fail before any compute on config mistakes that would otherwise surface mid-run."""
    method_ids = [m["method_id"] for m in cfg.get("methods", [])]
    if len(method_ids) != len(set(method_ids)):
        raise ConfigError(f"Duplicate method_id in evaluate.methods: {method_ids}")

    evaluation_ids = [e["evaluation_id"] for e in cfg.get("evaluations", [])]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ConfigError(f"Duplicate evaluation_id in evaluate.evaluations: {evaluation_ids}")

    for entry in cfg.get("evaluations", []):
        kind = entry["kind"]
        if kind not in evaluators.names():
            raise ConfigError(
                f"Unknown evaluation kind '{kind}' for evaluation_id '{entry['evaluation_id']}'. "
                f"Known kinds: {evaluators.names()}"
            )


class LoadStep:
    """Step 1: validate config, resolve ingest, build the graph, load every method/reference."""

    name = "load"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["evaluate"]
        _validate_config(cfg)

        ingest_lineage = Path(ctx.config["paths"]["processed"]) / ctx.dataset_id
        ingest_dir = ctx.resolve("ingest", ArtifactRef(lineage_dir=ingest_lineage, ref=cfg["inputs"]["ingest_run"]))
        state.ingest_run_id = ingest_dir.name

        nodes_table = read_nodes(ingest_dir / "nodes.parquet")
        state.graph_role_by_node = dict(
            zip(nodes_table.column("node_id").to_pylist(), nodes_table.column("graph_role").to_pylist())
        )

        edges_table = read_edges(ingest_dir / "edges.parquet")
        state.edges = edges_table.to_pylist()
        edge_pairs = list(
            zip(edges_table.column("src_id").to_pylist(), edges_table.column("dst_id").to_pylist())
        )
        state.graph = build_graph(nodes_table.column("node_id").to_pylist(), edge_pairs)

        # Exposed to evaluators that need gold pairs (gold_alignment); others ignore it.
        alignments_path = ingest_dir / "alignments.parquet"
        state.alignments = read_alignments(alignments_path) if alignments_path.exists() else None

        for entry in cfg.get("methods", []):
            spec = MethodSpec.from_config(entry)
            matrix, run_id = load_method(ctx, spec)
            state.methods[spec.method_id] = (matrix, run_id)
            state.method_specs[spec.method_id] = spec

        if not state.methods:
            raise ConfigError("evaluate.methods is empty; nothing to compare")

        ctx.logger.info(
            "load: %d method(s), graph has %d nodes / %d edges",
            len(state.methods),
            state.graph.number_of_nodes(),
            state.graph.number_of_edges(),
        )


class EvaluateStep:
    """Step 2: run every configured evaluation, collecting MetricRows. The pipeline's only loop."""

    name = "evaluate"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["evaluate"]

        for entry in cfg.get("evaluations", []):
            evaluator = evaluators.get(entry["kind"])()
            rows = evaluator.evaluate(state, entry["evaluation_id"], entry.get("params", {}))
            state.rows.extend(rows)
            ctx.logger.info(
                "evaluate: evaluation_id=%s kind=%s produced %d rows",
                entry["evaluation_id"],
                entry["kind"],
                len(rows),
            )

        state.digest.rows = state.rows


class VisualizeStep:
    """Step 3 (optional): t-SNE scatter per configured method, colored by graph_role."""

    name = "visualize"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["evaluate"].get("visualize", {})
        if not cfg.get("enabled", False):
            ctx.logger.info("visualize: disabled, skipping")
            return

        method_ids = cfg.get("methods") or list(state.methods.keys())
        for method_id in method_ids:
            if method_id not in state.methods:
                raise ConfigError(f"evaluate.visualize.methods references unknown method_id '{method_id}'")
            matrix, _ = state.methods[method_id]
            output_path = ctx.run_dir / "plots" / f"tsne_{method_id}.png"
            plot_tsne_by_role(
                matrix,
                state.graph_role_by_node,
                output_path,
                seed=ctx.seed,
                perplexity=cfg.get("perplexity", 25.0),
            )
            ctx.logger.info("visualize: wrote %s", output_path)


class ValidateAndManifestStep:
    """Step 4: write metrics.parquet + derived metrics.json, record outputs, LATEST."""

    name = "validate_and_manifest"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state

        metrics_parquet_path = ctx.run_dir / "metrics.parquet"
        metrics_json_path = ctx.run_dir / "metrics.json"
        write_metrics(
            state.dataset_id,
            state.run_id,
            state.digest,
            parquet_path=metrics_parquet_path,
            json_path=metrics_json_path,
        )
        ctx.manifest.record_output("metrics_parquet", metrics_parquet_path)
        ctx.manifest.record_output("metrics_json", metrics_json_path)

        plots_dir = ctx.run_dir / "plots"
        if plots_dir.is_dir():
            for plot_path in sorted(plots_dir.glob("*.png")):
                ctx.manifest.record_output(f"plot_{plot_path.stem}", plot_path)

        ctx.manifest.summary = {
            "ingest_run_id": state.ingest_run_id,
            "methods": sorted(state.methods.keys()),
            "evaluations": sorted({r.evaluation_id for r in state.rows}),
            "n_rows": len(state.rows),
        }


@pipelines.register("evaluate")
class EvaluatePipeline(Pipeline):
    """Runs one or more registered evaluations over several comparison methods."""

    name = "evaluate"

    def build_steps(self, ctx: RunContext) -> list[Step]:
        state = _EvaluateState(ctx)
        self._state = state
        return [
            LoadStep(state),
            EvaluateStep(state),
            VisualizeStep(state),
            ValidateAndManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "evaluations" / ctx.dataset_id
        write_latest(lineage_dir, ctx.run_id)
