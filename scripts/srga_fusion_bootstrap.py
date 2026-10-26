"""Paired bootstrap of fused / ensembled SRGA variants against their members (Sprint 6.1 analysis helper).

Builds the evaluator's exact global-mode query set (via ``srga_paired_ranks.query_ranks``), fuses
trained exports at the score level with the same ``fuse_matrices`` the ``fusion`` method kind uses, and
prints for each comparison the MRR of both sides, the paired MRR difference with a bootstrap CI over
queries, the paired wins/losses of the reciprocal rank, and the share of SapBERT-correct queries that
the first side breaks. A group of single models (``{seed}`` pattern) is represented by the per-query
mean reciprocal rank over its seeds, so its MRR equals the mean-over-seeds MRR of the tables.
Reads artifacts only and writes nothing.

Example::

    uv run python scripts/srga_fusion_bootstrap.py --split val \
        --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}" \
        --fuse A17=A11 --fuse A18=A12 --fuse A19=A11:0.5,A12:0.5 \
        --compare A17:A11 --compare A18:A12 --compare A19:A17 --compare A19:A18 --compare A19:A0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.methods import fuse_matrices
from srga_paired_ranks import _latest, model_run_dir, query_ranks


def bootstrap_diff(rr_a: np.ndarray, rr_b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Return ``(mean diff, ci_low, ci_high)`` of ``mean(rr_a) - mean(rr_b)`` resampling queries (95% percentile CI)."""
    rng = np.random.default_rng(seed)
    d = rr_a - rr_b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main() -> None:
    """Parse groups / fusions / comparisons and print the paired bootstrap table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--split", default="val")
    ap.add_argument("--group", action="append", default=[], help="LABEL=model_id pattern with '{seed}'")
    ap.add_argument("--fuse", action="append", default=[], help="NAME=GROUP[:weight],GROUP[:weight]... (members = every seed of each group)")
    ap.add_argument("--compare", action="append", default=[], help="A:B, compares side A against side B (group, fusion or A0)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    root = Path.cwd()
    ingest_dir = _latest(root / "data" / "processed" / args.dataset)
    matrices = {"A0": AlignedMatrix.load(_latest(root / "artifacts" / "embeddings" / args.dataset / "hf_mean_pool-sapbert"))}
    seed_names: dict[str, list[str]] = {}
    for spec in args.group:
        label, pattern = spec.split("=", 1)
        seed_names[label] = []
        for s in args.seeds:
            name = f"{label}:{s}"
            path = model_run_dir(root, args.dataset, pattern, s) / "node_embeddings"
            matrices[name] = AlignedMatrix.load(path)
            seed_names[label].append(name)
    fused: dict[str, str] = {}
    for spec in args.fuse:
        name, body = spec.split("=", 1)
        members = []
        for part in body.split(","):
            label, _, w = part.partition(":")
            for n in seed_names[label]:
                members.append((matrices[n], float(w or 1.0) / len(seed_names[label])))
        matrices[name] = fuse_matrices([m for m in members if m[1] > 0])
        fused[name] = name

    _, ranks, _ = query_ranks(matrices, ingest_dir, args.split)
    rr = {k: 1.0 / v for k, v in ranks.items()}
    side_rr = {"A0": rr["A0"]}
    side_ranks = {"A0": [ranks["A0"]]}
    for label, names in seed_names.items():
        side_rr[label] = np.mean([rr[n] for n in names], axis=0)
        side_ranks[label] = [ranks[n] for n in names]
    for name in fused:
        side_rr[name] = rr[name]
        side_ranks[name] = [ranks[name]]

    print(f"split={args.split} queries={len(rr['A0'])}")
    print("| side | MRR | Hits@1 | Hits@10 | broken SapBERT-correct |")
    print("|---|---|---|---|---|")
    right = ranks["A0"] == 1
    for k in side_rr:
        h1 = np.mean([np.mean(r == 1) for r in side_ranks[k]])
        h10 = np.mean([np.mean(r <= 10) for r in side_ranks[k]])
        broken = np.mean([np.sum(right & (r > 1)) for r in side_ranks[k]])
        print(f"| {k} | {side_rr[k].mean():.4f} | {h1:.4f} | {h10:.4f} | {broken:.1f} of {int(right.sum())} |")
    print("\n| A vs B | MRR A | MRR B | diff | 95% CI | wins | losses | ties |")
    print("|---|---|---|---|---|---|---|---|")
    for spec in args.compare:
        a, b = spec.split(":")
        d, lo, hi = bootstrap_diff(side_rr[a], side_rr[b], args.n_boot)
        # wins/losses on the (seed-averaged) reciprocal rank; ties within float precision
        diff = side_rr[a] - side_rr[b]
        w, l = int(np.sum(diff > 1e-9)), int(np.sum(diff < -1e-9))
        flag = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
        print(f"| {a} vs {b} | {side_rr[a].mean():.4f} | {side_rr[b].mean():.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] ({flag}) | {w} | {l} | {len(diff) - w - l} |")


if __name__ == "__main__":
    main()
