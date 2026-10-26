"""Probe before fitting the anchor re-ranker (Sprint 6.2): do anchors carry usable signal? Analysis only.

Prints (1) the share of queries whose source has a train anchor within 1 and 2 hops, for val and test
(no model involved), and (2) on **val only**, among queries whose gold target is in the first stage's
top-K, how often anchor agreement fires for the gold target against the first stage's top wrong
candidate. Applies the probe gate: the gold/wrong firing ratio >= 1.5 on at least 20% of val queries.
"Agreement fires" = parent, child or 2-hop agreement > 0. Nothing is fitted or written.

Example::

    uv run python scripts/srga_rerank_probe.py --top-k 20
"""

from __future__ import annotations

import argparse

import numpy as np
from srga_rerank_common import load_inputs

from cga.pipelines.evaluate.rerank import OntologyGraph, build_anchors, feature_names, featurize_split

ALL_GROUPS = ["scores", "anchor", "granularity", "exclusivity"]


def coverage(split: str, align, role_of, edges, anchors) -> tuple[int, float, float]:
    """Return (n queries, share with an anchor within 1 hop, within 2 hops) of the source, by split."""
    g1 = OntologyGraph(e for e in edges if e["graph_role"] == "graph_1")
    sources = sorted({(r["node_1_id"], r["node_2_id"]) for r in align if r["split"] == split})
    n1 = n2 = 0
    for s, _ in sources:
        n1 += any(a in anchors.partners for a in g1.within(s, 1))
        n2 += any(a in anchors.partners for a in g1.within(s, 2))
    return len(sources), n1 / len(sources), n2 / len(sources)


def main() -> None:
    """Print the coverage and firing tables and the probe-gate verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    matrices, role_of, align, edges = load_inputs(args.dataset, args.seeds)
    anchors = build_anchors(align, role_of)
    print(f"anchors: {len(anchors.pairs())} train pairs, {len(anchors.partners)} sources, {len(anchors.sources_of)} targets")
    print("\n| split | pairs | anchor within 1 hop | anchor within 2 hops |")
    print("|---|---|---|---|")
    for split in ("val", "test"):
        n, c1, c2 = coverage(split, align, role_of, edges, anchors)
        print(f"| {split} | {n} | {c1:.3f} | {c2:.3f} |")

    feats = featurize_split(matrices, role_of, align, edges, "val", "global", ALL_GROUPS, args.top_k, anchors)
    names = feature_names(ALL_GROUPS)
    col = {n: names.index(n) for n in names}
    in_k = [q for q in feats if q.gold_pos >= 0]
    rows = {"fires (parent, child or 2-hop)": [], "parent or child agreement": [], "granularity mismatch": [], "exclusive": []}
    for q in in_k:
        wrong = next(i for i, c in enumerate(q.top_ids) if c != q.target)
        agree = lambda i: float(max(q.X[i, col["parent_agree"]], q.X[i, col["child_agree"]], q.X[i, col["hop2_agree"]]) > 0)  # noqa: E731
        agree1 = lambda i: float(max(q.X[i, col["parent_agree"]], q.X[i, col["child_agree"]]) > 0)  # noqa: E731
        rows["fires (parent, child or 2-hop)"].append((agree(q.gold_pos), agree(wrong)))
        rows["parent or child agreement"].append((agree1(q.gold_pos), agree1(wrong)))
        rows["granularity mismatch"].append((q.X[q.gold_pos, col["granularity_mismatch"]], q.X[wrong, col["granularity_mismatch"]]))
        rows["exclusive"].append((q.X[q.gold_pos, col["exclusive"]], q.X[wrong, col["exclusive"]]))
    print(f"\nval queries {len(feats)}; gold in the first stage's top-{args.top_k}: {len(in_k)}")
    print("\n| feature | fires for gold | fires for top wrong | ratio | share of all val queries where it fires for gold |")
    print("|---|---|---|---|---|")
    for label, vals in rows.items():
        g, w = np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals])
        print(f"| {label} | {g:.3f} | {w:.3f} | {g / w if w else float('inf'):.2f} | {g * len(in_k) / len(feats):.3f} |")
    g, w = np.mean([v[0] for v in rows["fires (parent, child or 2-hop)"]]), np.mean([v[1] for v in rows["fires (parent, child or 2-hop)"]])
    share = g * len(in_k) / len(feats)
    ok = (w == 0 or g / w >= 1.5) and share >= 0.2
    print(f"\nProbe gate: ratio {g / w if w else float('inf'):.2f} (>= 1.5) and share {share:.3f} (>= 0.20) -> {'GO' if ok else 'STOP'}")


if __name__ == "__main__":
    main()
