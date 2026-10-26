"""``plot`` pipeline: figures from an ``evaluate`` run's ``metrics.parquet`` (§5.6),
plus per-run training loss curves from ``gcl_train`` runs' ``metrics.jsonl`` (§6.8).

A thin sixth pipeline so re-plotting never re-scores: it only reads
``metrics.parquet``/``metrics.jsonl`` and renders PNGs, and one ``evaluate``
run can be plotted several ways (draft, paper, slides) without touching the
numbers. Four steps: ``load`` (resolve the ``evaluate`` run, read
``metrics.parquet``, validate every figure's ``select`` up front) ->
``render`` (registry lookup per ``figures[]`` entry) -> ``loss_curves``
(optional; one PNG per discovered ``gcl_train`` model_id) ->
``validate_and_manifest``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa

from cga.core.artifacts import ArtifactRef, write_latest
from cga.core.errors import ConfigError
from cga.core.io import read_parquet
from cga.core.pipeline import Pipeline, Step
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.schemas import METRICS_SCHEMA
from cga.pipelines.plot.loss_curves import discover_gcl_runs, read_loss_curve, render_loss_curve
from cga.pipelines.plot.renderers import FigureSpec, apply_select, renderers


class _PlotState:
    """Mutable state threaded across this run's steps (one instance per run)."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.metrics_table: pa.Table | None = None
        self.figure_specs: list[FigureSpec] = []
        self.selected_tables: dict[str, pa.Table] = {}
        self.evaluate_run_id: str = ""
        self.loss_curve_model_ids: list[str] = []


def _validate_config(cfg: dict[str, Any]) -> list[FigureSpec]:
    """Build and validate every ``FigureSpec``, failing before any rendering.

    Raises:
        ConfigError: On a duplicate ``figure_id`` or an unknown ``kind``.
    """
    entries = cfg.get("figures", [])
    figure_ids = [e["figure_id"] for e in entries]
    if len(figure_ids) != len(set(figure_ids)):
        raise ConfigError(f"Duplicate figure_id in plot.figures: {figure_ids}")

    specs = []
    for entry in entries:
        spec = FigureSpec.from_config(entry)
        if spec.kind not in renderers.names():
            raise ConfigError(
                f"Unknown renderer kind '{spec.kind}' for figure_id '{spec.figure_id}'. "
                f"Known kinds: {renderers.names()}"
            )
        specs.append(spec)
    return specs


class LoadStep:
    """Step 1: resolve the evaluate run, read metrics.parquet, validate every figure's select."""

    name = "load"

    def __init__(self, state: _PlotState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["plot"]
        state.figure_specs = _validate_config(cfg)
        loss_curves_enabled = cfg.get("loss_curves", {}).get("enabled", False)
        if not state.figure_specs and not loss_curves_enabled:
            raise ConfigError("plot.figures is empty and plot.loss_curves.enabled is false; nothing to render")

        if not state.figure_specs:
            ctx.logger.info("load: plot.figures is empty, skipping metrics.parquet (loss_curves only)")
            return

        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "evaluations" / ctx.dataset_id
        evaluate_dir = ctx.resolve("evaluate_run", ArtifactRef(lineage_dir=lineage_dir, ref=cfg["inputs"]["evaluate_run"]))
        state.evaluate_run_id = evaluate_dir.name

        state.metrics_table = read_parquet(evaluate_dir / "metrics.parquet", schema=METRICS_SCHEMA)

        for spec in state.figure_specs:
            state.selected_tables[spec.figure_id] = apply_select(state.metrics_table, spec.select)

        ctx.logger.info(
            "load: %d row(s) from evaluate run %s, %d figure(s) configured",
            state.metrics_table.num_rows,
            state.evaluate_run_id,
            len(state.figure_specs),
        )


class RenderStep:
    """Step 2: registry lookup per figures[] entry, one PNG each."""

    name = "render"

    def __init__(self, state: _PlotState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        format_cfg = ctx.config["plot"].get("format", {})
        ext = format_cfg.get("ext", "png")

        for spec in state.figure_specs:
            renderer = renderers.get(spec.kind)()
            output_path = ctx.run_dir / "figures" / f"{spec.figure_id}.{ext}"
            renderer.render(state.selected_tables[spec.figure_id], spec, output_path)
            ctx.logger.info("render: figure_id=%s kind=%s -> %s", spec.figure_id, spec.kind, output_path)


class LossCurvesStep:
    """Step 3 (optional): one loss-curve PNG per discovered ``gcl_train`` model_id.

    Not driven by ``metrics.parquet``/``FigureSpec`` -- each ``gcl_train``
    run's per-epoch losses are training diagnostics, read directly from that
    run's ``metrics.jsonl``. Gated by ``plot.loss_curves.enabled`` (default
    ``False``) so existing figure-only configs are unaffected. One figure per
    run, never a cross-run comparison (decided with the developer): every
    ``model_id`` under the dataset is discovered and plotted independently.
    """

    name = "loss_curves"

    def __init__(self, state: _PlotState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["plot"].get("loss_curves", {})
        if not cfg.get("enabled", False):
            ctx.logger.info("loss_curves: disabled, skipping")
            return

        gcl_dataset_dir = Path(ctx.config["paths"]["models"]) / "gcl" / ctx.dataset_id
        runs = discover_gcl_runs(gcl_dataset_dir)
        if not runs:
            ctx.logger.warning("loss_curves: no gcl_train runs with metrics.jsonl found under %s", gcl_dataset_dir)
            return

        format_cfg = ctx.config["plot"].get("format", {})
        ext = format_cfg.get("ext", "png")

        for model_id, run_dir in runs.items():
            ctx.resolve(f"gcl_run:{model_id}", ArtifactRef(lineage_dir=run_dir.parent, ref=run_dir.name))
            curve = read_loss_curve(run_dir)
            output_path = ctx.run_dir / "figures" / f"loss_curve_{model_id}.{ext}"
            render_loss_curve(curve, output_path, title=f"{model_id} training loss")
            state.loss_curve_model_ids.append(model_id)
            ctx.logger.info("loss_curves: model_id=%s run=%s -> %s", model_id, run_dir.name, output_path)


class ValidateAndManifestStep:
    """Step 4: record every rendered figure as an output, LATEST."""

    name = "validate_and_manifest"

    def __init__(self, state: _PlotState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        figures_dir = ctx.run_dir / "figures"
        for figure_path in sorted(figures_dir.glob("*")):
            ctx.manifest.record_output(f"figure_{figure_path.stem}", figure_path)

        ctx.manifest.summary = {
            "evaluate_run_id": state.evaluate_run_id,
            "figures": sorted(spec.figure_id for spec in state.figure_specs),
            "loss_curve_model_ids": sorted(state.loss_curve_model_ids),
        }


@pipelines.register("plot")
class PlotPipeline(Pipeline):
    """Renders one or more figures from a resolved ``evaluate`` run's metrics."""

    name = "plot"

    def build_steps(self, ctx: RunContext) -> list[Step]:
        state = _PlotState(ctx)
        self._state = state
        return [
            LoadStep(state),
            RenderStep(state),
            LossCurvesStep(state),
            ValidateAndManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "plots" / ctx.dataset_id
        write_latest(lineage_dir, ctx.run_id)
