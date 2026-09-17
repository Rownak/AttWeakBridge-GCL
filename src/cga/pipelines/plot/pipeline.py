"""``plot`` pipeline: figures from an ``evaluate`` run's ``metrics.parquet`` (§5.6).

A thin sixth pipeline so re-plotting never re-scores: it only reads
``metrics.parquet`` and renders PNGs, and one ``evaluate`` run can be plotted
several ways (draft, paper, slides) without touching the numbers. Three
steps: ``load`` (resolve the ``evaluate`` run, read ``metrics.parquet``,
validate every figure's ``select`` up front) -> ``render`` (registry lookup
per ``figures[]`` entry) -> ``validate_and_manifest``.
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
from cga.pipelines.plot.renderers import FigureSpec, apply_select, renderers


class _PlotState:
    """Mutable state threaded across this run's steps (one instance per run)."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.metrics_table: pa.Table | None = None
        self.figure_specs: list[FigureSpec] = []
        self.selected_tables: dict[str, pa.Table] = {}
        self.evaluate_run_id: str = ""


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
        if not state.figure_specs:
            raise ConfigError("plot.figures is empty; nothing to render")

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


class ValidateAndManifestStep:
    """Step 3: record every rendered figure as an output, LATEST."""

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
            ValidateAndManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "plots" / ctx.dataset_id
        write_latest(lineage_dir, ctx.run_id)
