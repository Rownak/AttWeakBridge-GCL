"""Markdown tables of a ``gold_alignment_prf`` evaluate run (Sprint 8 analysis helper).

Groups the per-seed methods of a run by their label (``"A3 MLP seed 42"`` -> ``"A3 MLP"``), and prints
mean +- sample std over seeds of precision / recall / F1 for each ``gold_alignment_prf`` evaluation,
plus the counts (means) and the one-to-one recall ceiling ``min(unique sources, unique targets) / n_gold``.
It reads ``metrics.parquet`` only and writes nothing.

Example::

    uv run python scripts/srga_prf_table.py --dataset ncit_doid_srga --run 20261025T213718Z-b6f8d965
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _fmt(values: pd.Series, digits: int = 4) -> str:
    if len(values) > 1:
        return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"
    return f"{values.iloc[0]:.{digits}f}"


def main() -> None:
    """Print one table per evaluation in the run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--run", required=True, help="evaluate run id")
    ap.add_argument("--evaluation", help="only this evaluation_id")
    args = ap.parse_args()

    path = Path("artifacts/evaluations") / args.dataset / args.run / "metrics.parquet"
    df = pd.read_parquet(path)
    df = df[df.evaluation_kind == "gold_alignment_prf"].copy()
    df["group"] = df.method_label.map(lambda s: re.sub(r" seed \d+$", "", s))
    wide = df.pivot_table(
        index=["evaluation_id", "group", "method_id"], columns="metric_name", values="metric_value"
    ).reset_index()
    wide["ceiling"] = wide[["n_unique_sources", "n_unique_targets"]].min(axis=1) / wide["n_gold"]
    order = sorted(dict.fromkeys(df.group), key=lambda g: (int(m.group(1)) if (m := re.search(r"A(\d+)", g)) else 0, g))
    for eid, block in wide.groupby("evaluation_id", sort=False):
        if args.evaluation and eid != args.evaluation:
            continue
        print(f"\n### {eid}\n")
        print("| method | seeds | P | R | F1 | n_predicted | n_correct | n_gold | recall ceiling |")
        print("|---|---|---|---|---|---|---|---|---|")
        for g in order:
            b = block[block.group == g]
            if b.empty:
                continue
            print(
                f"| {g} | {len(b)} | {_fmt(b.precision)} | {_fmt(b.recall)} | {_fmt(b.f1)} | "
                f"{b.n_predicted.mean():.1f} | {b.n_correct.mean():.1f} | {int(b.n_gold.iloc[0])} | {b.ceiling.iloc[0]:.4f} |"
            )


if __name__ == "__main__":
    main()
