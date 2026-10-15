"""Stratified fix/break report of an SRGA variant against SapBERT (Sprint 5 Phase 5.4, design §7).

Each gold query (global mode) is put in one of four strata by two questions:

- **SapBERT right / wrong:** does SapBERT (A0) rank the gold target first?
- **Confuser near / far:** is SapBERT's top confuser (its best-ranked candidate that is not the gold
  target) within 2 hops of the gold target in the target graph (undirected stored edges), or farther?

Per stratum it counts, for each seed of the variant:

- **fixed:** SapBERT rank > 1 and variant rank = 1;
- **broken:** SapBERT rank = 1 and variant rank > 1;
- **unchanged** at rank 1 (the rest), plus how many ranks got better or worse.

Real structural value should show up as fixes in the "SapBERT wrong" strata (most of all where the
confuser is near the target) with few breaks in "SapBERT right". Reads artifacts only; writes nothing.

Example::

    uv run python scripts/srga_stratified.py --dataset ncit_doid_srga --split test \
        --group A3=srga-mlp-l0.25-s{seed} --group A4=srga-rel-l0.25-s{seed}
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from cga.data.aligned import AlignedMatrix
from cga.data.graph_store import read_edges
from srga_paired_ranks import _latest, query_ranks  # noqa: E402  (sibling script on sys.path)


def _within_hops(adj: dict[str, set[str]], start: str, goal: str, max_hops: int) -> bool:
    """Return whether ``goal`` is within ``max_hops`` undirected hops of ``start`` (BFS)."""
    if start == goal:
        return True
    seen, frontier = {start}, deque([(start, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth == max_hops:
            continue
        for nxt in adj.get(node, ()):
            if nxt == goal:
                return True
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, depth + 1))
    return False


def main() -> None:
    """Print the stratum sizes and per-stratum fixed/broken/unchanged counts (mean +- std over seeds)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--split", default="test")
    ap.add_argument("--group", action="append", required=True, help="LABEL=model_id with '{seed}'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-hops", type=int, default=2)
    args = ap.parse_args()

    root = Path.cwd()
    ingest_dir = _latest(root / "data" / "processed" / args.dataset)
    matrices = {"A0": AlignedMatrix.load(_latest(root / "artifacts" / "embeddings" / args.dataset / "hf_mean_pool-sapbert"))}
    groups: dict[str, list[str]] = {}
    for spec in args.group:
        label, pattern = spec.split("=", 1)
        groups[label] = []
        for seed in args.seeds:
            name = f"{label}:{seed}"
            run = _latest(root / "models" / "gcl" / args.dataset / pattern.format(seed=seed))
            matrices[name] = AlignedMatrix.load(run / "node_embeddings")
            groups[label].append(name)

    queries, ranks, confusers = query_ranks(matrices, ingest_dir, args.split)

    edges = read_edges(ingest_dir / "edges.parquet").to_pylist()
    adj: dict[str, set[str]] = {}
    for e in edges:
        if e["graph_role"] == "graph_2":
            adj.setdefault(e["src_id"], set()).add(e["dst_id"])
            adj.setdefault(e["dst_id"], set()).add(e["src_id"])

    base = ranks["A0"]
    right = base == 1
    near = np.array(
        [_within_hops(adj, target, conf, args.max_hops) for (_, target), conf in zip(queries, confusers["A0"])]
    )
    strata = {
        "SapBERT right, confuser near": right & near,
        "SapBERT right, confuser far": right & ~near,
        "SapBERT wrong, confuser near": ~right & near,
        "SapBERT wrong, confuser far": ~right & ~near,
        "all": np.ones_like(right),
    }
    print(f"split={args.split} queries={len(queries)} near = top SapBERT confuser within {args.max_hops} hops of the gold target")
    for label, names in groups.items():
        print(f"\n### {label} vs A0 (mean +- std over seeds {args.seeds})")
        print("| stratum | n | fixed | broken | unchanged | rank better | rank worse |")
        print("|---|---|---|---|---|---|---|")
        for stratum, mask in strata.items():
            per_seed = []
            for name in names:
                v, b = ranks[name][mask], base[mask]
                fixed = int(np.sum((b > 1) & (v == 1)))
                broken = int(np.sum((b == 1) & (v > 1)))
                per_seed.append((fixed, broken, len(v) - fixed - broken, int(np.sum(v < b)), int(np.sum(v > b))))
            arr = np.array(per_seed, dtype=float)
            m, s = arr.mean(axis=0), arr.std(axis=0, ddof=1)
            cells = " | ".join(f"{mi:.1f} +- {si:.1f}" for mi, si in zip(m, s))
            print(f"| {stratum} | {int(mask.sum())} | {cells} |")


if __name__ == "__main__":
    main()
