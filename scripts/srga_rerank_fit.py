"""Fit and cross-validate the anchor re-ranker variants A21-A24 on val (Sprint 6.2). Gated: run only after the probe passes.

Variants (all fitted on the **val** split, global mode, over the first stage's top-K):

- **A21** all feature groups; **A22** without the exclusivity group; **A23** first-stage scores only (learned
  score fusion, no anchors); **A24** A21 with the anchors' DOID partners shuffled (same endpoints, correspondences destroyed).

Per variant it reports the nested 5-fold CV MRR (folds grouped by source node, ``(K, l2)`` re-selected inside each
training fold, K in {10, 20, 50}, l2 in {0.01, 0.1, 1}), paired bootstrap CIs of A21 against A22 / A23 / A24 / the
first stage, and the gate "A21 beats A23 (CI excludes 0) and A24 < A21". It then selects ``(K, l2)`` on all val by CV,
fits on all val, prints the weights and writes ``artifacts/rerankers/<dataset>/<variant>.json``. Val only; no test read.

Example::

    uv run python scripts/srga_rerank_fit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from srga_fusion_bootstrap import bootstrap_diff
from srga_rerank_common import FIRST_STAGE_IDS, load_inputs

from cga.pipelines.evaluate.rerank import (
    build_anchors,
    feature_names,
    featurize_split,
    fit_listwise,
    nested_cv_ranks,
    save_reranker,
    select_config,
)

VARIANTS = {
    "A21": {"groups": ["scores", "anchor", "granularity", "exclusivity"], "shuffle": False},
    "A22": {"groups": ["scores", "anchor", "granularity"], "shuffle": False},
    "A23": {"groups": ["scores"], "shuffle": False},
    "A24": {"groups": ["scores", "anchor", "granularity", "exclusivity"], "shuffle": True},
}
MAX_K = 50


def main() -> None:
    """Run the nested CV, print the val table, comparisons and weights, and write the re-ranker files."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    matrices, role_of, align, edges = load_inputs(args.dataset, args.seeds)
    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts") / "rerankers" / args.dataset
    true_anchors = build_anchors(align, role_of)
    shuffled = build_anchors(align, role_of, shuffle_seed=args.shuffle_seed)

    cv_rr: dict[str, np.ndarray] = {}
    first_rr = None
    for name, v in VARIANTS.items():
        anchors = shuffled if v["shuffle"] else true_anchors
        feats = featurize_split(matrices, role_of, align, edges, "val", "global", v["groups"], MAX_K, anchors)
        if first_rr is None:
            first_rr = np.array([1.0 / q.first_rank for q in feats])
        ranks, chosen = nested_cv_ranks(feats)
        cv_rr[name] = 1.0 / ranks
        k, l2, grid = select_config(feats)
        model = fit_listwise(feats, k, l2)
        meta = {
            "name": name,
            "groups": v["groups"],
            "feature_names": feature_names(v["groups"]),
            "first_stage": FIRST_STAGE_IDS,
            "shuffle_seed": args.shuffle_seed if v["shuffle"] else None,
            "fit_split": "val",
            "cv_selection": {f"K={a},l2={b}": round(s, 4) for (a, b), s in grid.items()},
            "nested_cv_mrr": float(np.mean(cv_rr[name])),
        }
        save_reranker(out_dir / f"{name}.json", model, meta)
        print(f"\n{name}: groups={v['groups']}  nested-CV choices per fold={chosen}  final K={k} l2={l2}")
        for fname, w in zip(feature_names(v["groups"]), model["weights"]):
            print(f"  {fname:22s} {w:+.3f}")

    print(f"\nval queries={len(first_rr)}")
    print("| variant | nested-CV MRR | Hits@1 | Hits@10 |")
    print("|---|---|---|---|")
    print(f"| first stage (A19) | {first_rr.mean():.4f} | {np.mean(first_rr == 1):.4f} | {np.mean(first_rr >= 0.1):.4f} |")
    for name, rr in cv_rr.items():
        print(f"| {name} | {rr.mean():.4f} | {np.mean(rr == 1):.4f} | {np.mean(rr >= 0.1):.4f} |")
    print("\n| comparison | diff | 95% CI |")
    print("|---|---|---|")
    for other in ("A22", "A23", "A24"):
        d, lo, hi = bootstrap_diff(cv_rr["A21"], cv_rr[other], args.n_boot)
        print(f"| A21 vs {other} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    d, lo, hi = bootstrap_diff(cv_rr["A21"], first_rr, args.n_boot)
    print(f"| A21 vs first stage | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    _, lo23, _ = bootstrap_diff(cv_rr["A21"], cv_rr["A23"], args.n_boot)
    ok = lo23 > 0 and cv_rr["A24"].mean() < cv_rr["A21"].mean()
    print(f"\nFit gate: A21 > A23 with CI excluding 0 (lower bound {lo23:+.4f}) and A24 < A21 -> {'GO' if ok else 'STOP'}")


if __name__ == "__main__":
    main()
