"""Does a model rank held-out gold targets higher just because the loss never saw them? (Sprint 5, Phase 5.5)

Training excludes every held-out node (val and test endpoints) from the loss, so it never appears as a
negative, while the ~5k trainable DOID nodes are pushed away from the training anchors on every step.
A high-capacity model could therefore score a gold target (always held-out) above the trainable
candidates for a reason that has nothing to do with the target: those candidates are memorised, the
target is not. The standard ``global`` pool is ~61% trainable nodes, so this would inflate Hits@K/MRR.

This script re-scores each query with the candidate pool cut three ways (the gold target is always kept):

- **all:** the evaluator's pool (reproduces ``metrics.parquet``);
- **held-out only:** target-graph candidates the loss never saw (val + test endpoints). Every candidate
  is on equal footing with the target, so a memorisation shortcut cannot help;
- **trainable only:** only candidates the loss did train on. This is where a shortcut would help most.

A model whose gain survives the "held-out only" pool is not exploiting it. Reads artifacts only.

Example::

    uv run python scripts/srga_pool_diagnostic.py --dataset ncit_doid_srga --split val \
        --group A3=srga-mlp-l0.25-s{seed} --group A12=gcn-full-e200-s{seed}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cga.data.aligned import AlignedMatrix, NodeUniverse
from srga_paired_ranks import _latest, model_run_dir, add_fusions, query_ranks


def main() -> None:
    """Print mean MRR (over seeds) of each group under the three candidate pools."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--split", default="val")
    ap.add_argument("--group", action="append", required=True, help="LABEL=model_id, '{seed}' expands per seed")
    ap.add_argument("--fuse", action="append", default=[], help="NAME=GROUP[:weight],... score-level fusion of groups")
    ap.add_argument("--only", nargs="*", default=None, help="report only these labels (plus A0)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--sampler", default="supervised-negatives_per_positive5")
    args = ap.parse_args()

    root = Path.cwd()
    ingest_dir = _latest(root / "data" / "processed" / args.dataset)
    universe = NodeUniverse.from_parquet(_latest(root / "artifacts" / "training_data" / args.dataset / args.sampler) / "universe.parquet")
    held = {n for n, h, r in zip(universe.node_ids, universe.held_out, universe.graph_roles) if h and r == "graph_2"}
    trainable = {n for n, h, r in zip(universe.node_ids, universe.held_out, universe.graph_roles) if not h and r == "graph_2"}

    matrices = {"A0": AlignedMatrix.load(_latest(root / "artifacts" / "embeddings" / args.dataset / "hf_mean_pool-sapbert"))}
    groups: dict[str, list[str]] = {"A0": ["A0"]}
    for spec in args.group:
        label, pattern = spec.split("=", 1)
        groups[label] = []
        for seed in args.seeds if "{seed}" in pattern else [None]:
            name = f"{label}:{seed}"
            matrices[name] = AlignedMatrix.load(
                model_run_dir(root, args.dataset, pattern, seed) / "node_embeddings"
            )
            groups[label].append(name)
    add_fusions(matrices, groups, args.fuse)
    if args.only is not None:
        groups = {k: v for k, v in groups.items() if k == "A0" or k in args.only}

    pools = {"all": None, "held-out only": held, "trainable only": trainable}
    result: dict[str, dict[str, list[float]]] = {g: {p: [] for p in pools} for g in groups}
    n_by_pool: dict[str, int] = {}
    for pool_name, keep in pools.items():
        queries, ranks, _ = query_ranks(matrices, ingest_dir, args.split, keep_candidates=keep)
        n_by_pool[pool_name] = len(queries)
        for label, names in groups.items():
            result[label][pool_name] = [float(np.mean(1.0 / ranks[n])) for n in names]

    print(f"split={args.split} queries={n_by_pool['all']}; DOID held-out={len(held)} trainable={len(trainable)}")
    print("| model | MRR, all (evaluator) | MRR, held-out only | MRR, trainable only |")
    print("|---|---|---|---|")
    for label in groups:
        cells = []
        for pool_name in pools:
            v = np.array(result[label][pool_name])
            cells.append(f"{v.mean():.4f}" + (f" +- {v.std(ddof=1):.4f}" if len(v) > 1 else ""))
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
