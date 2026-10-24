"""Data-scaling table and log-linear fit for the Sprint 6.3 ``pair_fraction`` runs (analysis only, reads artifacts).

Takes an evaluate run of ``ncit_doid_srga_scaling`` (val, global), reads each model's number of positive pairs from its
``gcl_train`` manifest, and prints per recipe and size the mean +- sample std over seeds of val MRR and Hits@10. It then
fits ``MRR = a + b * ln(pairs)`` by least squares over all (seed, size) points and reports the slope with its standard
error and the gain that the fit predicts for the extra pairs of the full-data lineage (``--extra-ratio``, default
1383 / 1174).

Example::

    uv run python scripts/srga_scaling_curve.py --evaluate-run 20261016T000000Z-abcdef12
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from srga_paired_ranks import _latest


def main() -> None:
    """Print the scaling table and the fitted slope and predicted gain per recipe."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--evaluate-run", default=None, help="run id under artifacts/evaluations/<dataset>; default LATEST")
    ap.add_argument("--extra-ratio", type=float, default=1383 / 1174)
    args = ap.parse_args()

    root = Path.cwd()
    eval_dir = root / "artifacts" / "evaluations" / args.dataset
    run = args.evaluate_run or (eval_dir / "LATEST").read_text(encoding="utf-8").strip()
    metrics = pd.read_parquet(eval_dir / run / "metrics.parquet")

    rows = []
    for mid in sorted(metrics.method_id.unique()):
        m = re.fullmatch(r"scale-gcn(-nofloor)?-p(\d+)-s(\d+)", mid)
        if not m:
            continue
        mrr = float(metrics[(metrics.method_id == mid) & (metrics.metric_name == "mrr")].metric_value.iloc[0])
        h10 = float(
            metrics[(metrics.method_id == mid) & (metrics.metric_name == "hits_at_k") & (metrics.param_value == 10)].metric_value.iloc[0]
        )
        manifest = json.loads((_latest(root / "models" / "gcl" / args.dataset / mid) / "manifest.json").read_text(encoding="utf-8"))
        rows.append(
            {"recipe": "A12" if m.group(1) else "A11", "pct": int(m.group(2)), "seed": int(m.group(3)), "mrr": mrr, "h10": h10,
             "pairs": manifest["summary"]["n_positive_pairs"]}
        )
    df = pd.DataFrame(rows)
    print(f"evaluate run {run}; {len(df)} models")
    print("\n| recipe | size | positive pairs (per seed) | val MRR | val Hits@10 |")
    print("|---|---|---|---|---|")
    for (recipe, pct), g in df.groupby(["recipe", "pct"]):
        pairs = "/".join(str(p) for p in g.sort_values("seed").pairs)
        print(f"| {recipe} | {pct}% | {pairs} | {g.mrr.mean():.4f} +- {g.mrr.std(ddof=1):.4f} | {g.h10.mean():.4f} +- {g.h10.std(ddof=1):.4f} |")

    print(f"\nLog-linear fit MRR = a + b ln(pairs) over all (seed, size) points; extra ratio {args.extra_ratio:.3f} (+{(args.extra_ratio - 1) * 100:.1f}% pairs)")
    print("| recipe | b (MRR per ln-pair) | SE(b) | MRR gain per doubling | predicted gain for the extra pairs | 95% range of that gain |")
    print("|---|---|---|---|---|---|")
    for recipe, g in df.groupby("recipe"):
        x, y = np.log(g.pairs.to_numpy(dtype=float)), g.mrr.to_numpy()
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        se = float(np.sqrt(resid.var(ddof=2) / np.sum((x - x.mean()) ** 2)))
        step = np.log(args.extra_ratio)
        print(f"| {recipe} | {b:+.4f} | {se:.4f} | {b * np.log(2):+.4f} | {b * step:+.4f} | [{(b - 1.96 * se) * step:+.4f}, {(b + 1.96 * se) * step:+.4f}] |")


if __name__ == "__main__":
    main()
