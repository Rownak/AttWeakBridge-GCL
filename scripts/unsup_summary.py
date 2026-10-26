"""Summary tables of one unsupervised-ablation ``evaluate`` run (Sprint 9 analysis helper).

Reads ``metrics.parquet`` of one evaluate run and writes ``summary.md`` and ``summary.csv`` into that run's directory.
Methods are grouped by dropping a trailing ``-s<seed>`` from the ``method_id`` (``u2-hard-s43`` -> ``u2-hard``), so
the three seeds of a matrix row become one row reported as mean ± sample std (ddof 1; a single seed shows the value only).

``summary.md`` holds
  * a gold table (Hits@1, Hits@10, MRR from ``gold_alignment``; P, R, F1, #Pred, #Correct from ``gold_alignment_prf``).
    A dataset without gold pairs has neither evaluation, and every cell becomes ``—``;
  * a paper table (``PAPER_ROWS`` order: variant, pseudo-pairs, loss, then the same metric columns) for the matrix rows present;
  * one gold-free table per ``neighborhood`` evaluation (hit@10 and MRR per role and view).

``summary.csv`` is the long format of every metric (group, evaluation, view, role, metric, k, mean, std, n_seeds).

Example::

    uv run python scripts/unsup_summary.py --dataset ncit_doid_unsup --run 20261026T024853Z-3e2fa375
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

MISSING = "—"
_SEED_SUFFIX = re.compile(r"-s\d+$")
_KEYS = ["method_group", "evaluation_id", "evaluation_kind", "view_id", "graph_role", "metric_name", "param_value"]

# (column header, metric_name, k or None for a metric without k)
_RANKING_COLUMNS = [("Hits@1", "hits_at_k", 1), ("Hits@10", "hits_at_k", 10), ("MRR", "mrr", None)]
_PRF_COLUMNS = [
    ("P", "precision", None),
    ("R", "recall", None),
    ("F1", "f1", None),
    ("#Pred", "n_predicted", None),
    ("#Correct", "n_correct", None),
]
# Paper-table rows (method group -> variant, pseudo-pairs, loss), in the order of dev_docs/sprints/sprint_9.md.
PAPER_ROWS = [
    ("u0", "BERT (frozen)", "–", "–"),
    ("u1", "BERT + projection", "top-k", "Triplet"),
    ("u1-i", "BERT + projection", "top-k", "InfoNCE"),
    ("u2", "BERT + GCN (text + structure)", "top-k", "Triplet"),
    ("u2-i", "BERT + GCN (text + structure)", "top-k", "InfoNCE"),
    ("u2-rand", "BERT + GCN", "random neg.", "Triplet"),
    ("u2-hard", "BERT + GCN", "hard neg.", "Triplet"),
    ("u2-mut", "BERT + GCN", "mutual top-k", "Triplet"),
]
_COUNT_METRICS = {"n_predicted", "n_correct", "n_gold", "n_unique_sources", "n_unique_targets"}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean / sample std / seed count of every metric per method group.

    Args:
        df: ``metrics.parquet`` rows.

    Returns:
        One row per ``_KEYS`` combination, with ``mean``, ``std`` (NaN for one seed), ``n_seeds`` and ``method_label``.
    """
    df = df.copy()
    df["method_group"] = df["method_id"].map(lambda m: _SEED_SUFFIX.sub("", m))
    df["param_value"] = df["param_value"].astype(float)
    grouped = df.groupby(_KEYS, dropna=False, sort=False)["metric_value"]
    out = grouped.agg(mean="mean", std=lambda s: s.std(ddof=1) if len(s) > 1 else float("nan"), n_seeds="count")
    out = out.reset_index()
    label = df.groupby("method_group", sort=False)["method_label"].first()
    out["method_label"] = out["method_group"].map(label)
    return out


def _cell(agg: pd.DataFrame, group: str, evaluation_id: str, view_id: str | None, name: str, k: float | None) -> str:
    """Format one table cell as ``mean ± std`` (or ``mean``), or ``—`` when the metric is absent."""
    mask = (agg.method_group == group) & (agg.evaluation_id == evaluation_id) & (agg.metric_name == name)
    if view_id is not None:
        mask &= agg.view_id == view_id
    mask &= agg.param_value.isna() if k is None else (agg.param_value == k)
    hit = agg[mask]
    if hit.empty:
        return MISSING
    row = hit.iloc[0]
    digits = 1 if name in _COUNT_METRICS else 4
    if pd.isna(row["std"]):
        return f"{row['mean']:.{digits}f}"
    return f"{row['mean']:.{digits}f} ± {row['std']:.{digits}f}"


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _first_evaluation(agg: pd.DataFrame, kind: str, requested: str | None) -> str | None:
    ids = list(dict.fromkeys(agg[agg.evaluation_kind == kind].evaluation_id))
    if requested:
        return requested if requested in ids else None
    return ids[0] if ids else None


def build_markdown(agg: pd.DataFrame, *, ranking_eval: str | None, prf_eval: str | None, view: str) -> str:
    """Render the gold table and the gold-free neighborhood tables.

    Args:
        agg: Output of :func:`aggregate`.
        ranking_eval: ``gold_alignment`` evaluation id for the ranking columns (default: the first one).
        prf_eval: ``gold_alignment_prf`` evaluation id for the P/R/F1 columns (default: the first one).
        view: ``gold_alignment`` view for the ranking columns (``global`` or ``candidates``).

    Returns:
        The Markdown text.
    """
    ranking_eval = _first_evaluation(agg, "gold_alignment", ranking_eval)
    prf_eval = _first_evaluation(agg, "gold_alignment_prf", prf_eval)
    groups = list(dict.fromkeys(agg.method_group))
    labels = dict(zip(agg.method_group, agg.method_label))
    n_seeds = agg.groupby("method_group").n_seeds.max()

    header = ["Variant", "Seeds"] + [c[0] for c in _RANKING_COLUMNS + _PRF_COLUMNS]
    rows = []
    for g in groups:
        cells = [_cell(agg, g, ranking_eval, view, n, k) if ranking_eval else MISSING for _, n, k in _RANKING_COLUMNS]
        cells += [_cell(agg, g, prf_eval, None, n, k) if prf_eval else MISSING for _, n, k in _PRF_COLUMNS]
        rows.append([f"{labels[g]} (`{g}`)", str(int(n_seeds[g]))] + cells)

    n_gold = MISSING
    if prf_eval:
        gold = agg[(agg.evaluation_id == prf_eval) & (agg.metric_name == "n_gold")]
        n_gold = f"{int(gold['mean'].iloc[0]):,}" if not gold.empty else MISSING
    parts = [
        "## Gold alignment",
        f"Ranking: `{ranking_eval or MISSING}` ({view}); pair set: `{prf_eval or MISSING}`; #Gold = {n_gold}. "
        "Mean ± sample std over seeds.",
        "",
        _markdown_table(header, rows),
    ]

    for evaluation_id in dict.fromkeys(agg[agg.evaluation_kind == "neighborhood"].evaluation_id):
        sub = agg[(agg.evaluation_id == evaluation_id) & (agg.metric_name.isin(["hit_at_k", "mrr"]))]
        if sub.empty:
            continue
        top_k = sub[sub.metric_name == "hit_at_k"].param_value.max()
        columns = list(dict.fromkeys(zip(sub.graph_role, sub.view_id)))
        head = ["Variant"] + [f"{role} · {view_id} · hit@{int(top_k)}" for role, view_id in columns]
        head += [f"{role} · {view_id} · MRR" for role, view_id in columns]
        body = []
        for g in groups:
            row = [f"{labels[g]} (`{g}`)"]
            for metric in ("hit_at_k", "mrr"):
                for role, view_id in columns:
                    mask = (
                        (sub.method_group == g)
                        & (sub.graph_role == role)
                        & (sub.view_id == view_id)
                        & (sub.metric_name == metric)
                    )
                    if metric == "hit_at_k":
                        mask &= sub.param_value == top_k  # neighborhood MRR repeats per k; its first row is used
                    hit = sub[mask]
                    if hit.empty:
                        row.append(MISSING)
                    else:
                        r = hit.iloc[0]
                        row.append(f"{r['mean']:.4f}" if pd.isna(r["std"]) else f"{r['mean']:.4f} ± {r['std']:.4f}")
            body.append(row)
        parts += ["", f"## Gold-free `{evaluation_id}` (neighborhood)", "", _markdown_table(head, body)]
    return "\n".join(parts) + "\n"


def build_paper_table(agg: pd.DataFrame, *, ranking_eval: str | None, prf_eval: str | None, view: str) -> str:
    """Render the Sprint 9 paper table (rows in :data:`PAPER_ROWS`; rows without results are omitted).

    Args:
        agg: Output of :func:`aggregate`.
        ranking_eval: ``gold_alignment`` evaluation id (default: the first one).
        prf_eval: ``gold_alignment_prf`` evaluation id (default: the first one).
        view: ``gold_alignment`` view for the ranking columns.

    Returns:
        The Markdown table, or an empty string when the run has no gold evaluation.
    """
    ranking_eval = _first_evaluation(agg, "gold_alignment", ranking_eval)
    prf_eval = _first_evaluation(agg, "gold_alignment_prf", prf_eval)
    if not ranking_eval and not prf_eval:
        return ""
    present = set(agg.method_group)
    header = ["Variant", "Pseudo-pairs", "Loss"] + [c[0] for c in _RANKING_COLUMNS + _PRF_COLUMNS]
    rows = []
    for group, variant, pairs, loss in PAPER_ROWS:
        if group not in present:
            continue
        cells = [_cell(agg, group, ranking_eval, view, n, k) if ranking_eval else MISSING for _, n, k in _RANKING_COLUMNS]
        cells += [_cell(agg, group, prf_eval, None, n, k) if prf_eval else MISSING for _, n, k in _PRF_COLUMNS]
        rows.append([variant, pairs, loss] + cells)
    return _markdown_table(header, rows)


def main() -> None:
    """Write ``summary.md`` and ``summary.csv`` into the run directory."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="evaluate run id")
    ap.add_argument("--root", default="artifacts/evaluations", help="evaluations root")
    ap.add_argument("--ranking-eval", help="gold_alignment evaluation_id (default: first)")
    ap.add_argument("--prf-eval", help="gold_alignment_prf evaluation_id (default: first)")
    ap.add_argument("--view", default="global", choices=["global", "candidates"])
    args = ap.parse_args()

    run_dir = Path(args.root) / args.dataset / args.run
    agg = aggregate(pd.read_parquet(run_dir / "metrics.parquet"))
    text = build_markdown(agg, ranking_eval=args.ranking_eval, prf_eval=args.prf_eval, view=args.view)
    paper = build_paper_table(agg, ranking_eval=args.ranking_eval, prf_eval=args.prf_eval, view=args.view)
    if paper:
        text += "\n## Paper table\n\n" + paper + "\n"
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    agg.rename(columns={"mean": "metric_mean", "std": "metric_std"}).to_csv(run_dir / "summary.csv", index=False)
    print(f"wrote {run_dir / 'summary.md'} and {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
