"""Per-run training loss curves from a ``gcl_train`` run's ``metrics.jsonl`` (§6.8).

Independent of ``renderers.py``'s ``FigureSpec``/``select`` machinery: a
``gcl_train`` run's per-epoch losses are training diagnostics, not evaluate
output, so there is no ``metrics.parquet`` row to select here -- this module
reads ``metrics.jsonl`` directly. One figure per run, never a cross-run
comparison (decided with the developer): every ``model_id`` under the
dataset gets its own loss-curve PNG, discovered automatically rather than
named in config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cga.core.artifacts import read_latest


@dataclass(frozen=True)
class LossCurve:
    """One run's per-epoch losses, in epoch order.

    Attributes:
        epoch: Epoch numbers, 1-indexed.
        loss: Combined loss (``alpha * loss_text + (1 - alpha) * loss_graph``).
        loss_text: Text-view loss.
        loss_graph: Graph-view loss.
    """

    epoch: list[int]
    loss: list[float]
    loss_text: list[float]
    loss_graph: list[float]


def read_loss_curve(run_dir: str | Path) -> LossCurve:
    """Load ``{run_dir}/metrics.jsonl`` into a :class:`LossCurve`.

    Raises:
        FileNotFoundError: If ``run_dir`` has no ``metrics.jsonl``.
    """
    metrics_path = Path(run_dir) / "metrics.jsonl"
    epoch, loss, loss_text, loss_graph = [], [], [], []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            epoch.append(record["epoch"])
            loss.append(record["loss"])
            loss_text.append(record["loss_text"])
            loss_graph.append(record["loss_graph"])
    return LossCurve(epoch=epoch, loss=loss, loss_text=loss_text, loss_graph=loss_graph)


def render_loss_curve(curve: LossCurve, output_path: str | Path, *, title: str) -> None:
    """Draw combined/graph/text loss vs. epoch on one log-scale axes and save a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.epoch, curve.loss, label="combined", linewidth=2)
    ax.plot(curve.epoch, curve.loss_graph, label="graph", linewidth=1, alpha=0.8)
    ax.plot(curve.epoch, curve.loss_text, label="text", linewidth=1, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(title)
    ax.legend(fontsize="small")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def discover_gcl_runs(gcl_dataset_dir: str | Path) -> dict[str, Path]:
    """Return ``{model_id: run_dir}`` for every ``model_id`` under a dataset's ``models/gcl/`` tree.

    Skips a ``model_id`` directory with no ``LATEST`` (no successful run) or
    whose ``LATEST`` run has no ``metrics.jsonl`` (e.g. checkpoint-only export
    runs); callers should log these rather than fail the whole pipeline over
    one missing run. A stray flat ``LATEST`` file directly under
    ``gcl_dataset_dir`` (pre-model_id-partition legacy, see Phase 1.9's
    execution log) is not itself a ``model_id`` directory and is ignored.
    """
    gcl_dataset_dir = Path(gcl_dataset_dir)
    if not gcl_dataset_dir.is_dir():
        return {}

    runs: dict[str, Path] = {}
    for model_dir in sorted(gcl_dataset_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        run_id = read_latest(model_dir)
        if run_id is None:
            continue
        run_dir = model_dir / run_id
        if not (run_dir / "metrics.jsonl").is_file():
            continue
        runs[model_dir.name] = run_dir
    return runs
