"""Score-threshold sweep of the BioGITOM-style greedy one-to-one P/R/F1 (Sprint 8.2 analysis helper).

BioGITOM uses threshold 0.0, so the number of predictions is set by the assignment, not by confidence, and
P and R stay close. This sweeps the minimum cosine over a fixed grid on one split (use ``val``: the test
split must not choose the threshold), prints the P / R / F1 curve per method (mean +- std over seeds for a
seeded group), and the threshold with the best mean F1 (ties go to the lower threshold). It reuses the
evaluator's own helpers (``top_k_candidates``, ``apply_filters``, ``assign_pairs``, ``score_pairs``), so the
threshold-0.0 row equals the ``gold_alignment_prf`` run. It reads artifacts only and writes nothing.

Example::

    uv run python scripts/srga_prf_threshold.py --dataset ncit_doid_srga --split val \
        --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}" --fuse A19=A11:0.5,A12:0.5 \
        --report A12 --report A19
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from srga_paired_ranks import _latest, add_fusions, model_run_dir

from cga.data.aligned import AlignedMatrix
from cga.data.graph_store import read_alignments, read_nodes
from cga.pipelines.evaluate.evaluators.gold_alignment_prf import (
    _gold_sets,
    apply_filters,
    assign_pairs,
    score_pairs,
    top_k_candidates,
)

DEFAULT_GRID = [round(0.05 * i, 2) for i in range(20)]  # 0.00 .. 0.95


def sweep(
    matrix: AlignedMatrix,
    role_of: dict[str, str],
    alignment_rows: list[dict],
    split: str,
    grid: list[float],
    filters: bool,
    top_k: int = 3,
) -> list[dict[str, float]]:
    """Return the greedy one-to-one metrics of ``matrix`` at every threshold of ``grid``."""
    gold, train_nodes = _gold_sets(alignment_rows, split)
    split_sources = {s for s, _ in gold}
    excluded = train_nodes - {n for pair in gold for n in pair}
    have = set(matrix.node_ids)
    pool_ids = sorted(n for n in have if role_of.get(n) == "graph_2")
    sources = sorted(s for s in split_sources if s in have and role_of.get(s) == "graph_1")
    rows = top_k_candidates(matrix.select(sources).vectors, matrix.select(pool_ids).vectors, sources, pool_ids, top_k)
    if filters:
        rows = apply_filters(rows, excluded, split_sources)
    return [score_pairs(assign_pairs(rows, "greedy_one_to_one", t), gold) for t in grid]


def main() -> None:
    """Print the P/R/F1 curve and best-F1 threshold per reported group, filters off and on."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--split", default="val")
    ap.add_argument("--group", action="append", required=True, help="LABEL=[lineage::]model_id, '{seed}' expands per seed")
    ap.add_argument("--fuse", action="append", default=[], help="NAME=GROUP[:w],GROUP[:w] (as srga_paired_ranks.py)")
    ap.add_argument("--report", action="append", help="groups to print (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    args = ap.parse_args()

    root = Path.cwd()
    ingest_dir = _latest(root / "data" / "processed" / args.dataset)
    role_of = {r["node_id"]: r["graph_role"] for r in read_nodes(ingest_dir / "nodes.parquet").to_pylist()}
    alignment_rows = read_alignments(ingest_dir / "alignments.parquet").to_pylist()

    matrices: dict[str, AlignedMatrix] = {}
    groups: dict[str, list[str]] = {}
    for spec in args.group:
        label, pattern = spec.split("=", 1)
        names = []
        for seed in args.seeds if "{seed}" in pattern else [None]:
            name = f"{label}:{seed}" if seed is not None else label
            matrices[name] = AlignedMatrix.load(model_run_dir(root, args.dataset, pattern, seed) / "node_embeddings")
            names.append(name)
        groups[label] = names
    add_fusions(matrices, groups, args.fuse)

    for label in args.report or list(groups):
        for filters in (False, True):
            curves = np.array(
                [
                    [[m[k] for k in ("precision", "recall", "f1", "n_predicted", "n_correct")] for m in
                     sweep(matrices[n], role_of, alignment_rows, args.split, args.grid, filters)]
                    for n in groups[label]
                ]
            )  # (seeds, grid, 5)
            mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(curves[0])
            best = int(np.argmax(mean[:, 2]))  # argmax returns the first (lowest) threshold on ties
            print(f"\n### {label}  split={args.split}  filters={'on' if filters else 'off'}  seeds={len(curves)}\n")
            print("| threshold | P | R | F1 | n_predicted | n_correct |")
            print("|---|---|---|---|---|---|")
            for i, t in enumerate(args.grid):
                cells = " | ".join(f"{mean[i, j]:.4f} ± {std[i, j]:.4f}" if len(curves) > 1 else f"{mean[i, j]:.4f}" for j in range(3))
                print(f"| {t:.2f} | {cells} | {mean[i, 3]:.1f} | {mean[i, 4]:.1f} |")
            print(f"\nbest mean F1 {mean[best, 2]:.4f} at threshold {args.grid[best]:.2f} (threshold 0.00: {mean[0, 2]:.4f})")


if __name__ == "__main__":
    main()
