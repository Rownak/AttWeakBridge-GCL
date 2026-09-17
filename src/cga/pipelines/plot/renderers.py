"""``renderers`` registry + ``FigureSpec``: the pluggable unit under ``plot.figures[]``.

A figure spec never names an evaluation kind (target_architecture.md §5.6) --
it selects a slice of the long-form ``metrics.parquet`` (§6.10) and maps
columns to plot roles (``x``/``hue``/``facet``). A renderer is a pure function
of that filtered table plus its own opaque ``params``, so a new evaluation is
usually plottable with zero new renderer code: only a new ``select`` filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, Field

from cga.core.errors import ConfigError
from cga.core.registry import Registry

renderers: Registry = Registry("renderer")


class FigureSpec(BaseModel):
    """One configured figure: a row selection plus its plot-role mapping.

    Attributes:
        figure_id: Unique id within this run's ``figures:`` list; also the
            output filename stem (``figures/{figure_id}.png``).
        kind: A key in the renderer registry (``line``, ``bar``, ...).
        select: Equality filter over ``metrics.parquet`` columns
            (e.g. ``{"evaluation_id": "neigh", "metric_name": "mrr"}``).
        x: Column plotted on the x-axis.
        hue: Column distinguishing series by color.
        facet: Columns to facet subplots by, in order.
        params: Opaque, renderer-specific extras.
    """

    figure_id: str
    kind: str
    select: dict[str, Any] = Field(default_factory=dict)
    x: str = "param_value"
    hue: str = "method_label"
    facet: list[str] = Field(default_factory=lambda: ["view_id", "graph_role"])
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> "FigureSpec":
        """Build a ``FigureSpec`` from one ``plot.figures[]`` config entry."""
        return cls(**entry)


def apply_select(table: pa.Table, select: dict[str, Any]) -> pa.Table:
    """Filter ``table`` to rows matching every ``column -> value`` pair in ``select``.

    Raises:
        ConfigError: If the resulting selection has zero rows -- a blank PNG
            is not an acceptable output (target_architecture.md §5.6).
    """
    filtered = table
    for column, value in select.items():
        filtered = filtered.filter(pc.equal(filtered.column(column), value))
    if filtered.num_rows == 0:
        raise ConfigError(f"Figure selection matched zero rows in metrics.parquet: {select}")
    return filtered


class Renderer(Protocol):
    """One registered figure kind."""

    def render(self, table: pa.Table, spec: FigureSpec, output_path: Path) -> None:
        """Render ``table`` (already ``select``-filtered) to ``output_path`` as a PNG."""
        ...


def _facet_groups(table: pa.Table, facet: list[str]) -> list[tuple[tuple[Any, ...], pa.Table]]:
    """Split ``table`` into ``(facet_key, sub_table)`` groups, skipping absent/null facet columns.

    A facet column that is entirely null for this selection (e.g. ``view_id``
    for ``cluster_quality`` rows) collapses to a single group rather than
    producing one all-null-key group per row, so faceting never crashes on it.
    """
    usable_facets = [
        col for col in facet if col in table.column_names and pc.any(pc.is_valid(table.column(col))).as_py()
    ]
    if not usable_facets:
        return [((), table)]

    keys = list(zip(*(table.column(col).to_pylist() for col in usable_facets)))
    order: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            order.append(key)

    groups = []
    for key in order:
        mask = [k == key for k in keys]
        groups.append((key, table.filter(pa.array(mask))))
    return groups


@renderers.register("line")
class LineRenderer:
    """One line per ``hue`` value, x=``spec.x``, one subplot per facet group."""

    def render(self, table: pa.Table, spec: FigureSpec, output_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = _facet_groups(table, spec.facet)
        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)

        for ax, (facet_key, sub_table) in zip(axes[0], groups):
            hue_values = sub_table.column(spec.hue).to_pylist()
            x_values = sub_table.column(spec.x).to_pylist()
            y_values = sub_table.column("metric_value").to_pylist()
            for hue_value in sorted(set(hue_values), key=str):
                points = sorted(
                    (x, y) for x, y, h in zip(x_values, y_values, hue_values) if h == hue_value
                )
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker="o", label=str(hue_value))
            ax.set_title(", ".join(str(k) for k in facet_key) or spec.figure_id)
            ax.set_xlabel(spec.x)
            ax.set_ylabel("metric_value")
            ax.legend(fontsize="small")

        fig.suptitle(spec.figure_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


@renderers.register("bar")
class BarRenderer:
    """One bar group per ``x`` value, bars within a group colored by ``hue``."""

    def render(self, table: pa.Table, spec: FigureSpec, output_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        groups = _facet_groups(table, spec.facet)
        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)

        for ax, (facet_key, sub_table) in zip(axes[0], groups):
            x_values = sub_table.column(spec.x).to_pylist()
            hue_values = sub_table.column(spec.hue).to_pylist()
            y_values = sub_table.column("metric_value").to_pylist()

            x_labels = sorted(set(str(v) for v in x_values))
            hue_labels = sorted(set(str(v) for v in hue_values))
            value_by = {(str(x), str(h)): y for x, h, y in zip(x_values, hue_values, y_values)}

            n_hues = len(hue_labels)
            width = 0.8 / max(n_hues, 1)
            positions = np.arange(len(x_labels))
            for i, hue_label in enumerate(hue_labels):
                heights = [value_by.get((x, hue_label), 0.0) for x in x_labels]
                ax.bar(positions + i * width, heights, width=width, label=hue_label)

            ax.set_xticks(positions + width * (n_hues - 1) / 2)
            ax.set_xticklabels(x_labels)
            ax.set_title(", ".join(str(k) for k in facet_key) or spec.figure_id)
            ax.set_xlabel(spec.x)
            ax.set_ylabel("metric_value")
            ax.legend(fontsize="small")

        fig.suptitle(spec.figure_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
