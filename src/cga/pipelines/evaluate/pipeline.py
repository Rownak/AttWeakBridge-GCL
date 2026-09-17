"""``evaluate`` pipeline: neighborhood-preservation analysis (target_architecture.md §5.5a).

Phase 1.8 scope: three steps, ``load`` (resolve the ``gcl_train`` run's
manifest chain back to ``nodes.parquet`` and the text/structural embedding
stores it was trained from -- nothing is passed in by hand) -> ``neighborhood``
(Precision@k / Hit@k / MRR against both reference views, every configured
``k``) -> ``visualize`` (optional t-SNE, skipped by default) ->
``validate_and_manifest``, writing
``artifacts/evaluations/{dataset_id}/{run_id}/metrics.json``.

The alignment-ranking ``rank``/``diagnose`` steps for labeled datasets
(target_architecture.md §5.5, Phase 1.9) are a separate, not-yet-implemented
extension of this same pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cga.core.artifacts import ArtifactRef, write_latest
from cga.core.errors import ArtifactValidationError, ConfigError
from cga.core.io import read_json, write_json
from cga.core.pipeline import Pipeline, Step
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix
from cga.data.graph_store import read_nodes
from cga.pipelines.evaluate.neighborhood import RoleViewMetrics, score_view
from cga.pipelines.evaluate.visualize import plot_tsne_by_role

_ROLES = ("graph_1", "graph_2")


class _EvaluateState:
    """Mutable state threaded across this run's steps (one instance per run)."""

    def __init__(self) -> None:
        self.learned_matrix: AlignedMatrix | None = None
        self.text_matrix: AlignedMatrix | None = None
        self.structural_matrix: AlignedMatrix | None = None
        self.graph_role_by_node: dict[str, str] = {}
        self.gcl_train_run_id: str = ""
        self.results: dict[str, dict[str, dict[int, RoleViewMetrics]]] = {}


def _find_input(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in manifest.get("inputs", []):
        if entry["name"] == name:
            return entry
    raise ArtifactValidationError(f"gcl_train run manifest has no recorded input named '{name}'")


class LoadStep:
    """Step 1: resolve the gcl_train run's manifest chain to embeddings and node roles."""

    name = "load"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["evaluate"]

        gcl_lineage = Path(ctx.config["paths"]["models"]) / "gcl" / ctx.dataset_id
        gcl_ref = ArtifactRef(lineage_dir=gcl_lineage, ref=cfg["inputs"]["gcl_train_run"])
        gcl_dir = ctx.resolve("gcl_train", gcl_ref)
        state.gcl_train_run_id = gcl_dir.name

        gcl_manifest = read_json(gcl_dir / "manifest.json")
        state.learned_matrix = AlignedMatrix.load(gcl_dir / "node_embeddings")

        training_data_input = _find_input(gcl_manifest, "training_data")
        training_data_dir = ctx.resolve(
            "training_data",
            ArtifactRef(
                lineage_dir=Path(training_data_input["path"]).parent, ref=training_data_input["run_id"]
            ),
        )
        td_manifest = read_json(training_data_dir / "manifest.json")

        text_input = _find_input(td_manifest, "text_embeddings")
        graph_input = _find_input(td_manifest, "graph_embeddings")
        ingest_input = _find_input(td_manifest, "ingest")

        text_dir = ctx.resolve(
            "text_embeddings",
            ArtifactRef(lineage_dir=Path(text_input["path"]).parent, ref=text_input["run_id"]),
        )
        structural_dir = ctx.resolve(
            "structural_embeddings",
            ArtifactRef(lineage_dir=Path(graph_input["path"]).parent, ref=graph_input["run_id"]),
        )
        ingest_dir = ctx.resolve(
            "ingest", ArtifactRef(lineage_dir=Path(ingest_input["path"]).parent, ref=ingest_input["run_id"])
        )

        state.text_matrix = AlignedMatrix.load(text_dir)
        state.structural_matrix = AlignedMatrix.load(structural_dir)

        nodes_table = read_nodes(ingest_dir / "nodes.parquet")
        state.graph_role_by_node = dict(
            zip(nodes_table.column("node_id").to_pylist(), nodes_table.column("graph_role").to_pylist())
        )

        ctx.logger.info(
            "load: learned=%d nodes, text_reference=%d nodes, structural_reference=%d nodes",
            len(state.learned_matrix.node_ids),
            len(state.text_matrix.node_ids),
            len(state.structural_matrix.node_ids),
        )


class NeighborhoodStep:
    """Step 2: Precision@k / Hit@k / MRR against the text and structural reference views."""

    name = "neighborhood"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert (
            state.learned_matrix is not None
            and state.text_matrix is not None
            and state.structural_matrix is not None
        )

        cfg = ctx.config["evaluate"]["neighborhood"]
        ks: list[int] = cfg["ks"]
        views: list[str] = cfg["views"]

        reference_by_view = {"text": state.text_matrix, "structural": state.structural_matrix}
        unknown_views = set(views) - set(reference_by_view)
        if unknown_views:
            raise ConfigError(f"Unknown evaluate.neighborhood.views entries: {sorted(unknown_views)}")

        for view in views:
            reference_matrix = reference_by_view[view]
            state.results[view] = {}
            for role in _ROLES:
                state.results[view][role] = score_view(
                    learned_matrix=state.learned_matrix,
                    reference_matrix=reference_matrix,
                    graph_role_by_node=state.graph_role_by_node,
                    role=role,
                    ks=ks,
                )
                ctx.logger.info(
                    "neighborhood: view=%s role=%s n_queries=%d",
                    view,
                    role,
                    state.results[view][role][ks[0]].n_queries if ks else 0,
                )


class VisualizeStep:
    """Step 3 (optional): t-SNE scatter of the learned embeddings, colored by graph_role."""

    name = "visualize"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["evaluate"].get("visualize", {})
        if not cfg.get("enabled", False):
            ctx.logger.info("visualize: disabled, skipping")
            return

        assert state.learned_matrix is not None
        output_path = ctx.run_dir / "plots" / "learned_embeddings_tsne.png"
        plot_tsne_by_role(
            state.learned_matrix,
            state.graph_role_by_node,
            output_path,
            seed=ctx.seed,
            perplexity=cfg.get("perplexity", 25.0),
        )
        ctx.logger.info("visualize: wrote %s", output_path)


class ValidateAndManifestStep:
    """Step 4: write metrics.json, record outputs, LATEST."""

    name = "validate_and_manifest"

    def __init__(self, state: _EvaluateState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state

        metrics: dict[str, Any] = {"neighborhood": {}}
        for view, by_role in state.results.items():
            metrics["neighborhood"][view] = {
                role: {
                    str(k): {
                        "n_queries": rvm.n_queries,
                        "precision_at_k": rvm.precision_at_k,
                        "hit_at_k": rvm.hit_at_k,
                        "mrr": rvm.mrr,
                    }
                    for k, rvm in by_k.items()
                }
                for role, by_k in by_role.items()
            }

        metrics_path = ctx.run_dir / "metrics.json"
        write_json(metrics, metrics_path)
        ctx.manifest.record_output("metrics", metrics_path)

        plots_path = ctx.run_dir / "plots" / "learned_embeddings_tsne.png"
        if plots_path.is_file():
            ctx.manifest.record_output("tsne_plot", plots_path)

        ctx.manifest.summary = {
            "gcl_train_run_id": state.gcl_train_run_id,
            "views": list(state.results.keys()),
        }


@pipelines.register("evaluate")
class EvaluatePipeline(Pipeline):
    """Runs neighborhood-preservation scoring and optional visualization."""

    name = "evaluate"

    def build_steps(self, ctx: RunContext) -> list[Step]:
        state = _EvaluateState()
        self._state = state
        return [
            LoadStep(state),
            NeighborhoodStep(state),
            VisualizeStep(state),
            ValidateAndManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "evaluations" / ctx.dataset_id
        write_latest(lineage_dir, ctx.run_id)
