"""Summary tables of one unsupervised-ablation ``evaluate`` run (Sprint 9 analysis helper).

Reads ``metrics.parquet`` of one evaluate run and writes ``summary.md`` and ``summary.csv`` into that run's directory.
Methods are grouped by dropping a trailing ``-s<seed>`` from the ``method_id`` (``u2-hard-s43`` -> ``u2-hard``), so
the three seeds of a matrix row become one row reported as mean ± sample std (ddof 1; a single seed shows the value only).

``summary.md`` holds
  * a gold table (Hits@1, Hits@10, MRR from ``gold_alignment``; P, R, F1, #Pred, #Correct from ``gold_alignment_prf``).
    A dataset without gold pairs has neither evaluation, and every cell becomes ``—``;
  * a paper table (``PAPER_ROWS`` order: variant, pseudo-pairs, loss, then the same metric columns) for the matrix rows present;
  * with ``--prefix`` / ``--text-label``, the paper table of a prefixed run (Phase 9.3: ``sap-``, ``SapBERT``), and with
    ``--compare <run>`` a table of MRR and F1 of the same rows under both encoders;
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
# Phase 9.4 rows (DeepWalk as the only encoder input). Pseudo-pairs of u1 .. u2-i come from SapBERT cosine, u2-own from DeepWalk's.
DEEPWALK_ROWS = [
    ("u0", "DeepWalk (frozen)", "–", "–"),
    ("u1", "DeepWalk + projection", "SapBERT top-k", "Triplet"),
    ("u1-i", "DeepWalk + projection", "SapBERT top-k", "InfoNCE"),
    ("u2", "GCN of DeepWalk", "SapBERT top-k", "Triplet"),
    ("u2-i", "GCN of DeepWalk", "SapBERT top-k", "InfoNCE"),
    ("u2-own", "GCN of DeepWalk", "DeepWalk top-k", "Triplet"),
]
# Phase 9.5 rows (frozen SapBERT block plus GCN residual branches; SapBERT top-k pairs, InfoNCE for the trained rows).
RESIDUAL_ROWS = [
    ("r0", "SapBERT (frozen)", "–", "–"),
    ("a1", "SapBERT + training-free neighbour residual (A1)", "–", "–"),
    ("r1", "SapBERT + GCN(SapBERT)", "SapBERT top-k", "InfoNCE"),
    ("r2", "SapBERT + GCN(DeepWalk)", "SapBERT top-k", "InfoNCE"),
    ("r3", "SapBERT + GCN(SapBERT) + GCN(DeepWalk)", "SapBERT top-k", "InfoNCE"),
]
# Sprint 10 rows (ATT&CK-CWE, SecureBERT2; cross-graph SecureBERT2 top-k pairs, InfoNCE; fusion is score-level, untrained).
ATTACK_ENT_ROWS = [
    ("ent-r1", "SecureBERT2 (frozen)", "–", "–"),
    ("ent-r2", "GCN(SecureBERT2)", "SecureBERT2 top-k", "InfoNCE"),
    ("ent-r3", "SecureBERT2 + 0.25 x GCN(SecureBERT2) (fusion)", "SecureBERT2 top-k", "InfoNCE"),
]
# Sprint 10.5: the same three rows with SecRoBERTa as the text encoder.
ATTACK_ENT_SRB_ROWS = [
    ("ent-s-r1", "SecRoBERTa (frozen)", "–", "–"),
    ("ent-s-r2", "GCN(SecRoBERTa)", "SecRoBERTa top-k", "InfoNCE"),
    ("ent-s-r3", "SecRoBERTa + 0.25 x GCN(SecRoBERTa) (fusion)", "SecRoBERTa top-k", "InfoNCE"),
]
# Sprint 10.6: both encoders and the random baseline, in one evaluate run with a full-CWE view and a shared-pool view.
ATTACK_ENT_POOL_ROWS = ATTACK_ENT_ROWS + ATTACK_ENT_SRB_ROWS + [("ent-rand", "Random ranking", "–", "–")]
# Sprint 11.4: the dual-encoder GCN (triplet / InfoNCE, SecRoBERTa + DeepWalk) against frozen SecRoBERTa, the residual adapter and random.
ATTACK_ENT_DUAL_ROWS = [
    ("ent-d-trip", "Dual GCN, triplet", "SecRoBERTa top-k", "Triplet"),
    ("ent-d-nce", "Dual GCN, InfoNCE", "SecRoBERTa top-k", "InfoNCE"),
    ("ent-s-r1", "SecRoBERTa (frozen)", "–", "–"),
    ("ent-s-r2", "GCN(SecRoBERTa) residual (10.5)", "SecRoBERTa top-k", "InfoNCE"),
    ("ent-rand", "Random ranking", "–", "–"),
]
# Sprint 11.7: set-based evaluation (gold_set_retrieval) of the corrected texts, with the pre-fix SecRoBERTa store as a reference.
ATTACK_ENT_SET_ROWS = [
    ("ent-d-trip", "Dual GCN, triplet", "SecRoBERTa top-k", "Triplet"),
    ("ent-d-nce", "Dual GCN, InfoNCE", "SecRoBERTa top-k", "InfoNCE"),
    ("ent-s-r1", "SecRoBERTa (frozen)", "–", "–"),
    ("ent-s-r1-old", "SecRoBERTa (frozen, pre-fix texts)", "–", "–"),
    ("ent-rand", "Random ranking", "–", "–"),
]
ROW_SETS = {
    "matrix": PAPER_ROWS,
    "deepwalk": DEEPWALK_ROWS,
    "residual": RESIDUAL_ROWS,
    "attack_ent": ATTACK_ENT_ROWS,
    "attack_ent_srb": ATTACK_ENT_SRB_ROWS,
    "attack_ent_pool": ATTACK_ENT_POOL_ROWS,
    "attack_ent_dual": ATTACK_ENT_DUAL_ROWS,
    "attack_ent_set": ATTACK_ENT_SET_ROWS,
}
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


def build_paper_table(
    agg: pd.DataFrame,
    *,
    ranking_eval: str | None,
    prf_eval: str | None,
    view: str,
    prefix: str = "",
    text_label: str = "BERT",
    rows: list[tuple[str, str, str, str]] | None = None,
) -> str:
    """Render the Sprint 9 paper table (rows in :data:`PAPER_ROWS`, or ``rows``; rows without results are omitted).

    Args:
        agg: Output of :func:`aggregate`.
        ranking_eval: ``gold_alignment`` evaluation id (default: the first one).
        prf_eval: ``gold_alignment_prf`` evaluation id (default: the first one).
        view: ``gold_alignment`` view for the ranking columns.
        prefix: Method-group prefix of the run (``sap-`` for Phase 9.3: group ``sap-u2`` is row ``u2``).
        text_label: Name that replaces ``BERT`` in the row labels (``SapBERT``).
        rows: Row set (default :data:`PAPER_ROWS`; Phase 9.4 passes :data:`DEEPWALK_ROWS`).

    Returns:
        The Markdown table, or an empty string when the run has no gold evaluation.
    """
    ranking_eval = _first_evaluation(agg, "gold_alignment", ranking_eval)
    prf_eval = _first_evaluation(agg, "gold_alignment_prf", prf_eval)
    if not ranking_eval and not prf_eval:
        return ""
    present = set(agg.method_group)
    header = ["Variant", "Pseudo-pairs", "Loss"] + [c[0] for c in _RANKING_COLUMNS + _PRF_COLUMNS]
    body = []
    for row_key, variant, pairs, loss in rows or PAPER_ROWS:
        group = prefix + row_key
        if group not in present:
            continue
        variant = variant.replace("BERT", text_label)
        cells = [_cell(agg, group, ranking_eval, view, n, k) if ranking_eval else MISSING for _, n, k in _RANKING_COLUMNS]
        cells += [_cell(agg, group, prf_eval, None, n, k) if prf_eval else MISSING for _, n, k in _PRF_COLUMNS]
        body.append([variant, pairs, loss] + cells)
    return _markdown_table(header, body)


def _mean(agg: pd.DataFrame, group: str, evaluation_id: str | None, view: str | None, name: str, k: float | None):
    """Return the seed-mean of one metric of one group, or None when it is absent."""
    if evaluation_id is None:
        return None
    mask = (agg.method_group == group) & (agg.evaluation_id == evaluation_id) & (agg.metric_name == name)
    if view is not None:
        mask &= agg.view_id == view
    mask &= agg.param_value.isna() if k is None else (agg.param_value == k)
    hit = agg[mask]
    return None if hit.empty else float(hit.iloc[0]["mean"])


def build_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    names: tuple[str, str],
    prefixes: tuple[str, str],
    view: str,
    rows: list[tuple[str, str, str, str]] | None = None,
) -> str:
    """Render MRR and F1 of the same matrix rows under two text encoders, with the difference of the means.

    Args:
        first: :func:`aggregate` output of the first run (e.g. BERT).
        second: :func:`aggregate` output of the second run (e.g. SapBERT).
        names: Column names of the two encoders.
        prefixes: Method-group prefix of each run.
        view: ``gold_alignment`` view for MRR.
        rows: Row set (default :data:`PAPER_ROWS`).

    Returns:
        The Markdown table (rows of :data:`PAPER_ROWS` present in both runs); the difference is ``second - first``.
    """
    evals = [
        (_first_evaluation(a, "gold_alignment", None), _first_evaluation(a, "gold_alignment_prf", None))
        for a in (first, second)
    ]
    header = ["Row", "Pseudo-pairs", "Loss"]
    for metric in ("MRR", "F1"):
        header += [f"{names[0]} {metric}", f"{names[1]} {metric}", f"Δ {metric}"]
    body = []
    for row_key, variant, pairs, loss in rows or PAPER_ROWS:
        groups = (prefixes[0] + row_key, prefixes[1] + row_key)
        if groups[0] not in set(first.method_group) or groups[1] not in set(second.method_group):
            continue
        cells = [variant.replace("BERT", "text encoder"), pairs, loss]
        for metric, name, which in (("MRR", "mrr", 0), ("F1", "f1", 1)):
            values = [
                _mean(agg, g, evals[i][which], view if which == 0 else None, name, None)
                for i, (agg, g) in enumerate(zip((first, second), groups))
            ]
            cells += [MISSING if v is None else f"{v:.4f}" for v in values]
            cells.append(MISSING if None in values else f"{values[1] - values[0]:+.4f}")
        body.append(cells)
    return _markdown_table(header, body)


def build_source_report(
    aggs: dict[str, pd.DataFrame],
    *,
    view: str = "global",
    rows: list[tuple[str, str, str, str]] | None = None,
    ranking_eval: str | None = None,
    prf_eval: str | None = None,
) -> str:
    """Render the Sprint 10 report: one table per gold source and a compact MRR / F1 table (row x source).

    Each source is a separate gold set read by its own evaluate run; scores are never pooled. The per-source tables add
    ``#Gold`` and the recall ceiling ``unique sources / #Gold`` (one-to-one assignment gives each source one prediction).

    Args:
        aggs: ``{source name: aggregate() output of that source's evaluate run}``.
        view: ``gold_alignment`` view for the ranking columns.
        rows: Row set (default :data:`ATTACK_ENT_ROWS`; Phase 10.5 passes :data:`ATTACK_ENT_SRB_ROWS`).
        ranking_eval: ``gold_alignment`` evaluation id (default: the first one of each run; Phase 10.6 selects
            ``gold_full`` or ``gold_pool``).
        prf_eval: ``gold_alignment_prf`` evaluation id (default: the first one of each run).

    Returns:
        The Markdown text.
    """
    header = ["Variant", "Seeds"] + [c[0] for c in _RANKING_COLUMNS + _PRF_COLUMNS] + ["#Gold", "R ceiling"]
    parts: list[str] = []
    compact: dict[str, dict[str, str]] = {}
    for source, agg in aggs.items():
        ranking_eval_s = _first_evaluation(agg, "gold_alignment", ranking_eval)
        prf_eval_s = _first_evaluation(agg, "gold_alignment_prf", prf_eval)
        n_seeds = agg.groupby("method_group").n_seeds.max()
        body = []
        for key, variant, _, _ in rows or ATTACK_ENT_ROWS:
            if key not in set(agg.method_group):
                continue
            cells = [_cell(agg, key, ranking_eval_s, view, n, k) for _, n, k in _RANKING_COLUMNS]
            cells += [_cell(agg, key, prf_eval_s, None, n, k) for _, n, k in _PRF_COLUMNS]
            n_gold = _mean(agg, key, prf_eval_s, None, "n_gold", None)
            n_src = _mean(agg, key, prf_eval_s, None, "n_unique_sources", None)
            cells.append(MISSING if n_gold is None else f"{n_gold:.0f}")
            cells.append(MISSING if not n_gold or n_src is None else f"{n_src / n_gold:.4f}")
            body.append([variant, str(int(n_seeds[key]))] + cells)
            mrr = _cell(agg, key, ranking_eval_s, view, "mrr", None)
            f1 = _cell(agg, key, prf_eval_s, None, "f1", None)
            compact.setdefault(variant, {})[source] = f"{mrr} / {f1}"
        parts += [f"## Gold source `{source}`", "", _markdown_table(header, body), ""]
    sources = list(aggs)
    rows = [[v] + [compact[v].get(s, MISSING) for s in sources] for v in compact]
    parts += ["## MRR / F1 per row and gold source", "", _markdown_table(["Row"] + sources, rows), ""]
    return "\n".join(parts)


def build_set_report(
    aggs: dict[str, pd.DataFrame],
    *,
    set_eval: str,
    ks: tuple[int, ...] = (5, 10, 15, 20),
    lift_k: int = 10,
    rows: list[tuple[str, str, str, str]] | None = None,
    random_group: str = "ent-rand",
) -> str:
    """Render the Sprint 11.7 report of one ``gold_set_retrieval`` evaluation.

    One table per gold source (rows = methods; Hits / Recall / Precision at each ``k``, plus a row with
    the precision ceiling ``min(|R|, k) / k``), then a table of Recall@``lift_k`` and its ratio to the random baseline.

    Args:
        aggs: ``{source name: aggregate() output of that source's evaluate run}``.
        set_eval: ``gold_set_retrieval`` evaluation id (``set_full`` or ``set_pool``).
        ks: Cutoffs shown in the per-source tables.
        lift_k: Cutoff of the Recall lift table.
        rows: Row set (default :data:`ATTACK_ENT_SET_ROWS`).
        random_group: Method group of the random baseline.

    Returns:
        The Markdown text.
    """
    rows = rows or ATTACK_ENT_SET_ROWS
    header = ["Variant"] + [f"{m}@{k}" for k in ks for m in ("Hits", "Recall", "Precision")]
    parts: list[str] = []
    for source, agg in aggs.items():
        present = set(agg.method_group)
        body = []
        for key, variant, _, _ in rows:
            if key not in present:
                continue
            cells = [
                _cell(agg, key, set_eval, "global", name, k)
                for k in ks
                for name in ("hits_at_k", "recall_at_k", "precision_at_k")
            ]
            body.append([variant] + cells)
        first = next(r[0] for r in rows if r[0] in present)
        ceiling = [
            MISSING if name != "precision_at_k" else _cell(agg, first, set_eval, "global", "precision_ceiling_at_k", k)
            for k in ks
            for name in ("hits_at_k", "recall_at_k", "precision_at_k")
        ]
        body.append(["*precision ceiling*"] + ceiling)
        parts += [f"## Gold source `{source}` (`{set_eval}`)", "", _markdown_table(header, body), ""]
    sources = list(aggs)
    lift_header = ["Row"] + [f"{s} R@{lift_k} (x random)" for s in sources]
    lift_body = []
    for key, variant, _, _ in rows:
        cells = [variant]
        for source in sources:
            value = _mean(aggs[source], key, set_eval, "global", "recall_at_k", lift_k)
            base = _mean(aggs[source], random_group, set_eval, "global", "recall_at_k", lift_k)
            if value is None or base is None or base == 0:
                cells.append(MISSING)
            elif key == random_group:
                cells.append(f"{value:.4f}")
            else:
                cells.append(f"{value:.4f} ({value / base:.1f}x)")
        lift_body.append(cells)
    parts += [f"## Recall@{lift_k} and lift over the random baseline (`{set_eval}`)", "", _markdown_table(lift_header, lift_body), ""]
    return "\n".join(parts)


def build_lift_table(
    aggs: dict[str, pd.DataFrame],
    *,
    ranking_eval: str,
    random_group: str = "ent-rand",
    view: str = "global",
    rows: list[tuple[str, str, str, str]] | None = None,
) -> str:
    """Render each row's MRR divided by the random-ranking MRR of the same evaluation (same pool), per gold source.

    Args:
        aggs: ``{source: aggregate()}`` of the evaluate runs.
        ranking_eval: ``gold_alignment`` evaluation id (``gold_full`` or ``gold_pool``).
        random_group: Method group of the random baseline.
        view: ``gold_alignment`` view.
        rows: Row set (default :data:`ATTACK_ENT_POOL_ROWS`); the random row itself is shown as its MRR.

    Returns:
        The Markdown table: MRR (seed mean) and ``MRR / random MRR`` per source.
    """
    sources = list(aggs)
    header = ["Row"] + [f"{s} MRR (x random)" for s in sources]
    body = []
    for key, variant, _, _ in rows or ATTACK_ENT_POOL_ROWS:
        cells = [variant]
        for source in sources:
            value = _mean(aggs[source], key, ranking_eval, view, "mrr", None)
            base = _mean(aggs[source], random_group, ranking_eval, view, "mrr", None)
            if value is None or base is None:
                cells.append(MISSING)
            elif key == random_group:
                cells.append(f"{value:.4f}")
            else:
                cells.append(f"{value:.4f} ({value / base:.1f}x)")
        body.append(cells)
    return _markdown_table(header, body)


def build_encoder_comparison(
    first: dict[str, pd.DataFrame],
    second: dict[str, pd.DataFrame],
    *,
    names: tuple[str, str],
    rows: tuple[list, list],
    view: str = "global",
) -> str:
    """Render MRR and F1 of the three Sprint 10 rows under two text encoders, per gold source, with the difference.

    Args:
        first: ``{source: aggregate()}`` of the first encoder's evaluate runs.
        second: Same for the second encoder (same sources).
        names: Encoder names for the column headers.
        rows: Row sets of the two encoders; rows are matched by position (frozen, GCN, fusion).
        view: ``gold_alignment`` view for MRR.

    Returns:
        The Markdown table; ``Δ`` is ``second - first`` of the seed means.
    """
    header = ["Row", "Source"]
    for metric in ("MRR", "F1"):
        header += [f"{names[0]} {metric}", f"{names[1]} {metric}", f"Δ {metric}"]
    body = []
    for (k1, v1, _, _), (k2, _, _, _) in zip(*rows):
        for source in first:
            if source not in second:
                continue
            cells = [v1.replace("SecureBERT2", "encoder"), source]
            for metric, name, which in (("MRR", "mrr", 0), ("F1", "f1", 1)):
                vals = []
                for agg, key in ((first[source], k1), (second[source], k2)):
                    ev = _first_evaluation(agg, ("gold_alignment", "gold_alignment_prf")[which], None)
                    vals.append(_mean(agg, key, ev, view if which == 0 else None, name, None))
                cells += [MISSING if v is None else f"{v:.4f}" for v in vals]
                cells.append(MISSING if None in vals else f"{vals[1] - vals[0]:+.4f}")
            body.append(cells)
    return _markdown_table(header, body)


def main() -> None:
    """Write ``summary.md`` and ``summary.csv`` into the run directory."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="evaluate run id (with --source-run: the run whose directory receives sources_summary.md)")
    ap.add_argument("--root", default="artifacts/evaluations", help="evaluations root")
    ap.add_argument("--ranking-eval", help="gold_alignment evaluation_id (default: first)")
    ap.add_argument("--prf-eval", help="gold_alignment_prf evaluation_id (default: first)")
    ap.add_argument("--view", default="global", choices=["global", "candidates"])
    ap.add_argument("--prefix", default="", help="method-group prefix of this run (Phase 9.3: sap-)")
    ap.add_argument("--rows", default="matrix", choices=list(ROW_SETS), help="paper-table row set (Phase 9.4: deepwalk)")
    ap.add_argument("--text-label", default="BERT", help="encoder name in the paper-table row labels")
    ap.add_argument("--source-run", action="append", default=[], metavar="NAME=RUN",
                    help="Sprint 10: gold source name and its evaluate run; repeat once per source (writes sources_summary.md)")
    ap.add_argument("--set-eval", help="Sprint 11.7: gold_set_retrieval evaluation id; with --source-run writes build_set_report")
    ap.add_argument("--lift", action="store_true", help="Sprint 10.6: add the MRR / random-MRR table (needs --ranking-eval)")
    ap.add_argument("--out-name", default="sources_summary.md", help="file name written with --source-run")
    ap.add_argument("--compare-source-run", action="append", default=[], metavar="NAME=RUN",
                    help="Sprint 10.5: the other encoder's evaluate run per source (10.4 runs); needs --source-run")
    ap.add_argument("--compare-rows", default="attack_ent", choices=list(ROW_SETS), help="row set of the --compare-source-run runs")
    ap.add_argument("--compare-label", default="BERT", help="encoder name of the --compare run")
    ap.add_argument("--compare", help="evaluate run id of a second run (same dataset) to compare against")
    ap.add_argument("--compare-prefix", default="", help="method-group prefix of the --compare run")
    args = ap.parse_args()

    if args.source_run:
        pairs = [a.split("=", 1) for a in args.source_run]
        aggs = {n: aggregate(pd.read_parquet(Path(args.root) / args.dataset / r / "metrics.parquet")) for n, r in pairs}
        out = Path(args.root) / args.dataset / args.run / args.out_name
        if args.set_eval:
            out.write_text(build_set_report(aggs, set_eval=args.set_eval, rows=ROW_SETS[args.rows]), encoding="utf-8")
            print(f"wrote {out}")
            return
        text = build_source_report(
            aggs, view=args.view, rows=ROW_SETS[args.rows], ranking_eval=args.ranking_eval, prf_eval=args.prf_eval
        )
        if args.lift:
            lift_eval = args.ranking_eval or _first_evaluation(next(iter(aggs.values())), "gold_alignment", None)
            text += (
                f"\n## MRR and lift over the random baseline (`{lift_eval}`)\n\n"
                + build_lift_table(aggs, ranking_eval=lift_eval, view=args.view, rows=ROW_SETS[args.rows])
                + "\n"
            )
        if args.compare_source_run:
            other = {
                n: aggregate(pd.read_parquet(Path(args.root) / args.dataset / r / "metrics.parquet"))
                for n, r in (a.split("=", 1) for a in args.compare_source_run)
            }
            text += (
                f"\n## {args.compare_label} vs {args.text_label}: MRR and F1 per row and gold source\n\n"
                + build_encoder_comparison(
                    other, aggs, names=(args.compare_label, args.text_label),
                    rows=(ROW_SETS[args.compare_rows], ROW_SETS[args.rows]), view=args.view,
                )
                + "\n"
            )
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
        return
    run_dir = Path(args.root) / args.dataset / args.run
    agg = aggregate(pd.read_parquet(run_dir / "metrics.parquet"))
    text = build_markdown(agg, ranking_eval=args.ranking_eval, prf_eval=args.prf_eval, view=args.view)
    paper = build_paper_table(
        agg,
        ranking_eval=args.ranking_eval,
        prf_eval=args.prf_eval,
        view=args.view,
        prefix=args.prefix,
        text_label=args.text_label,
        rows=ROW_SETS[args.rows],
    )
    if paper:
        text += "\n## Paper table\n\n" + paper + "\n"
    if args.compare:
        other = aggregate(pd.read_parquet(Path(args.root) / args.dataset / args.compare / "metrics.parquet"))
        comparison = build_comparison(
            other,
            agg,
            names=(args.compare_label, args.text_label),
            prefixes=(args.compare_prefix, args.prefix),
            view=args.view,
            rows=ROW_SETS[args.rows],
        )
        text += f"\n## {args.compare_label} (run {args.compare}) vs {args.text_label}\n\n" + comparison + "\n"
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    agg.rename(columns={"mean": "metric_mean", "std": "metric_std"}).to_csv(run_dir / "summary.csv", index=False)
    print(f"wrote {run_dir / 'summary.md'} and {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
