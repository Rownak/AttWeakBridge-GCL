"""Seed-spread table of the Sprint 12 gated models (Phase 12.6).

Reads ``metrics.parquet`` of an evaluate run whose methods are trained at several seeds (``method_id`` ending in
``-s<seed>``) and writes ``seed_table.md`` / ``seed_table.csv`` next to it with the same 13 columns as
``paper_proxy_table.py`` (MRR and normalized precision@10 per view and graph, and the mean silhouette). A model with
several seeds is reported as ``mean ± sample std (ddof=1)`` over its seeds; a method without a seed suffix (the frozen
reference) is reported as a single value. The Markdown table bolds the best mean per column. The CSV has the mean and
std as separate columns (``<column> mean`` / ``<column> std``; std is empty for single-value rows). Output is
deterministic.

Example::

    uv run python scripts/paper_proxy_seed_table.py --dataset attack_cwe_ent --run 20261028T170633Z-afc3f972
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paper_proxy_table as base  # noqa: E402

_SEED = re.compile(r"^(?P<base>.+)-s(?P<seed>\d+)$")
DIGITS = base.DIGITS


def seed_groups(metrics: pd.DataFrame) -> dict[str, list[str]]:
    """Group the methods of a run by model, in first-appearance order.

    Args:
        metrics: Long metrics frame (needs ``method_id`` and ``method_label``).

    Returns:
        ``{group label: [method_id, ...]}``. The label is the method label with its trailing `` seed <n>`` removed;
        a method id without a ``-s<seed>`` suffix is its own group.
    """
    groups: dict[str, list[str]] = {}
    for method_id, label in metrics.drop_duplicates("method_id")[["method_id", "method_label"]].itertuples(index=False):
        key = re.sub(r"\s+seed\s+\d+$", "", label) if _SEED.match(method_id) else label
        groups.setdefault(key, []).append(method_id)
    return groups


def build_seed_table(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Mean and sample standard deviation over seeds of every table column.

    Args:
        metrics: Long metrics frame of a multi-seed evaluate run.

    Returns:
        ``(mean, std, n_seeds)``: two frames indexed by group label with the 13 columns of
        :func:`paper_proxy_table.build_table` (``std`` is NaN where a group has one method) and the number of methods
        per group.
    """
    wide = base.build_table(metrics)
    # build_table is indexed by label (one row per method, first-appearance order); re-key by method id for grouping.
    wide.index = metrics.drop_duplicates("method_id")["method_id"].tolist()
    groups = seed_groups(metrics)
    mean = pd.DataFrame({g: wide.loc[ids].mean() for g, ids in groups.items()}).T
    std = pd.DataFrame({g: wide.loc[ids].std(ddof=1) if len(ids) > 1 else wide.loc[ids].iloc[0] * float("nan")
                        for g, ids in groups.items()}).T
    return mean, std, {g: len(ids) for g, ids in groups.items()}


def to_markdown(mean: pd.DataFrame, std: pd.DataFrame, n_seeds: dict[str, int]) -> str:
    """Render ``mean ± std`` cells as Markdown, bolding the best mean of each column.

    Args:
        mean: Output mean frame of :func:`build_seed_table`.
        std: Output std frame.
        n_seeds: Number of methods (seeds) per group.

    Returns:
        Markdown text ending in a newline; groups with one method show the plain value.
    """
    best = mean.round(DIGITS).max(axis=0)
    lines = ["| Method (seeds) | " + " | ".join(mean.columns) + " |", "|---|" + "---:|" * len(mean.columns)]
    for label, row in mean.round(DIGITS).iterrows():
        cells = []
        for col, value in row.items():
            text = f"{value:.{DIGITS}f}"
            if n_seeds[label] > 1:
                text += f" ± {std.loc[label, col]:.{DIGITS}f}"
            cells.append(f"**{text}**" if value == best[col] else text)
        lines.append(f"| {label} ({n_seeds[label]}) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_seed_table(run_dir: Path) -> tuple[Path, Path]:
    """Write ``seed_table.md`` and ``seed_table.csv`` into an evaluate run directory.

    Args:
        run_dir: Evaluate run directory containing ``metrics.parquet``.

    Returns:
        The ``(markdown, csv)`` paths written.

    Side effects:
        Creates or overwrites the two files in ``run_dir``.
    """
    mean, std, n_seeds = build_seed_table(pd.read_parquet(run_dir / "metrics.parquet"))
    md_path, csv_path = run_dir / "seed_table.md", run_dir / "seed_table.csv"
    md_path.write_text(to_markdown(mean, std, n_seeds), encoding="utf-8", newline="\n")
    out = pd.concat([mean.add_suffix(" mean"), std.add_suffix(" std")], axis=1)
    out.insert(0, "n_seeds", pd.Series(n_seeds))
    cols = ["n_seeds"] + [f"{c} {k}" for c in mean.columns for k in ("mean", "std")]
    out[cols].round(DIGITS).to_csv(csv_path, index_label="method", lineterminator="\n")
    return md_path, csv_path


def main() -> None:
    """Parse arguments and write the seed table for the given evaluate run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="evaluate run id")
    ap.add_argument("--root", default="artifacts/evaluations", help="evaluations root directory")
    args = ap.parse_args()
    for path in write_seed_table(Path(args.root) / args.dataset / args.run):
        print(path)


if __name__ == "__main__":
    main()
