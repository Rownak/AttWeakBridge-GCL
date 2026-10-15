"""Per-query paired rank comparison of SRGA variants against SapBERT (Sprint 5 analysis helper).

The ``gold_alignment`` evaluator reports only aggregates, but the Sprint 5 gates also need *paired*
win/loss counts. This script rebuilds the evaluator's exact query set and candidate lists (it calls
the evaluator's own ``_build_queries`` and ``mrr_via_argsort``), so its MRR matches the evaluator's
``metrics.parquet`` to float precision, and prints each variant's wins / losses / ties against
SapBERT per seed. It reads artifacts only and writes nothing.

Example::

    uv run python scripts/srga_paired_ranks.py --dataset ncit_doid_srga --split val \
        --group A1=srga-nb-l0.25-dir --group A3=srga-mlp-l0.25-s{seed} --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cga.data.aligned import AlignedMatrix
from cga.data.graph_store import read_alignments, read_nodes
from cga.pipelines.evaluate.evaluators.gold_alignment import GoldAlignmentEvaluator, _normalize
from cga.pipelines.evaluate.metrics import mrr_via_argsort, top_k_by_similarity


def _latest(lineage: Path) -> Path:
    """Return the run directory a lineage's ``LATEST`` file points to."""
    return lineage / (lineage / "LATEST").read_text(encoding="utf-8").strip()


def query_ranks(
    matrices: dict[str, AlignedMatrix],
    ingest_dir: Path,
    split: str,
    keep_candidates: set[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, np.ndarray], dict[str, list[str]]]:
    """Return the evaluator's global-mode queries, each method's target rank, and its top confuser.

    Args:
        matrices: Method name -> embeddings. All methods share one query set and candidate pool.
        ingest_dir: Ingest run directory (``nodes.parquet``, ``alignments.parquet``).
        split: Gold alignment split to query.
        keep_candidates: When given, every query's candidate list is cut to these ids (the gold
            target is always kept). ``None`` reproduces the evaluator's pool exactly.

    Returns:
        ``(queries, ranks, confusers)``: ``ranks[name][i]`` is the 1-based rank of query ``i``'s gold
        target and ``confusers[name][i]`` the highest-ranked candidate that is not the target.
    """
    nodes = read_nodes(ingest_dir / "nodes.parquet").to_pylist()
    role_of = {r["node_id"]: r["graph_role"] for r in nodes}
    rows = read_alignments(ingest_dir / "alignments.parquet").to_pylist()
    shared = set.intersection(*(set(m.node_ids) for m in matrices.values()))
    targets_of: dict[str, set[str]] = {}
    for r in rows:
        targets_of.setdefault(r["node_1_id"], set()).add(r["node_2_id"])
    pool_ids = sorted(n for n in shared if role_of.get(n) == "graph_2")
    index_of = {n: i for i, n in enumerate(pool_ids)}
    pairs = sorted(
        {
            (r["node_1_id"], r["node_2_id"])
            for r in rows
            if r["split"] == split
            and r["node_1_id"] in shared
            and r["node_2_id"] in shared
            and role_of.get(r["node_1_id"]) == "graph_1"
            and role_of.get(r["node_2_id"]) == "graph_2"
        }
    )
    queries, cand_ids, cand_idx, _ = GoldAlignmentEvaluator._build_queries(
        "global", pairs, {}, targets_of, pool_ids, index_of, shared
    )
    ranks: dict[str, np.ndarray] = {}
    confusers: dict[str, list[str]] = {}
    for name, matrix in matrices.items():
        pool = _normalize(matrix.select(pool_ids).vectors)
        sources = sorted({q[0] for q in queries})
        src_row = {s: i for i, s in enumerate(sources)}
        sim = _normalize(matrix.select(sources).vectors) @ pool.T
        out, top_other = [], []
        for qi, (source, target) in enumerate(queries):
            row = sim[src_row[source]]
            row = row if cand_idx[qi] is None else row[cand_idx[qi]]
            ids = cand_ids[qi]
            if keep_candidates is not None:
                sel = [i for i, c in enumerate(ids) if c in keep_candidates or c == target]
                row, ids = row[sel], [ids[i] for i in sel]
            out.append(round(1.0 / mrr_via_argsort(row, ids, {target})))
            top_other.append(next(c for c in top_k_by_similarity(row, ids, 2) if c != target))
        ranks[name] = np.array(out)
        confusers[name] = top_other
    return queries, ranks, confusers


def _mrr(r: np.ndarray) -> float:
    return float(np.mean(1.0 / r))


def main() -> None:
    """Print per-seed MRR and paired wins/losses/ties of each group against SapBERT (and the previous group)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--split", default="val")
    ap.add_argument("--group", action="append", required=True, help="LABEL=model_id, '{seed}' expands per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    root = Path.cwd()
    ingest_dir = _latest(root / "data" / "processed" / args.dataset)
    matrices = {"A0": AlignedMatrix.load(_latest(root / "artifacts" / "embeddings" / args.dataset / "hf_mean_pool-sapbert"))}
    columns: dict[str, list[str]] = {}
    for spec in args.group:
        label, pattern = spec.split("=", 1)
        names = []
        for seed in args.seeds if "{seed}" in pattern else [None]:
            model_id = pattern.format(seed=seed)
            name = f"{label}:{seed}" if seed is not None else label
            matrices[name] = AlignedMatrix.load(_latest(root / "models" / "gcl" / args.dataset / model_id) / "node_embeddings")
            names.append(name)
        columns[label] = names

    queries, ranks, _ = query_ranks(matrices, ingest_dir, args.split)
    base = ranks["A0"]
    print(f"split={args.split} queries={len(queries)}  A0 MRR={_mrr(base):.4f}")
    print("| variant | seed | MRR | wins vs A0 | losses vs A0 | ties |")
    print("|---|---|---|---|---|---|")
    for label, names in columns.items():
        per = []
        for name in names:
            r = ranks[name]
            w, l = int(np.sum(r < base)), int(np.sum(r > base))
            seed = name.split(":")[1] if ":" in name else "-"
            print(f"| {label} | {seed} | {_mrr(r):.4f} | {w} | {l} | {len(r) - w - l} |")
            per.append((_mrr(r), w, l))
        arr = np.array(per)
        if len(per) > 1:
            m, s = arr.mean(axis=0), arr.std(axis=0, ddof=1)
            print(f"| **{label} mean +- std** | | {m[0]:.4f} +- {s[0]:.4f} | {m[1]:.1f} +- {s[1]:.1f} | {m[2]:.1f} +- {s[2]:.1f} | |")

    labels = list(columns)
    if len(labels) > 1:
        print("\nPaired, same seed (wins = first variant has the better rank):")
        print("| comparison | seed | wins | losses | ties |")
        print("|---|---|---|---|---|")
        for a in labels:
            for b in labels:
                if a >= b or len(columns[a]) != len(columns[b]) or len(columns[a]) == 1:
                    continue
                for na, nb in zip(columns[a], columns[b]):
                    w, l = int(np.sum(ranks[na] < ranks[nb])), int(np.sum(ranks[na] > ranks[nb]))
                    print(f"| {a} vs {b} | {na.split(':')[1]} | {w} | {l} | {len(ranks[na]) - w - l} |")


if __name__ == "__main__":
    main()
