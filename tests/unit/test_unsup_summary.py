"""Tests for ``scripts/unsup_summary.py``: seed grouping, mean ± sample std, and ``—`` for missing gold columns."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "unsup_summary", Path(__file__).parents[2] / "scripts" / "unsup_summary.py"
)
unsup_summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(unsup_summary)


def _row(method_id, evaluation_id, kind, metric, value, k=float("nan"), view="global", role="graph_1"):
    return {
        "evaluation_id": evaluation_id,
        "evaluation_kind": kind,
        "method_id": method_id,
        "method_label": "U2 BERT + GCN" if method_id.startswith("u2") else "U0",
        "view_id": view,
        "graph_role": role,
        "param_value": k,
        "metric_name": metric,
        "metric_value": value,
    }


def _gold_frame() -> pd.DataFrame:
    rows = [_row("u0", "gold", "gold_alignment", "mrr", 0.2), _row("u0", "prf", "gold_alignment_prf", "n_gold", 3280.0)]
    for seed, mrr in ((42, 0.30), (43, 0.34), (44, 0.38)):
        rows.append(_row(f"u2-s{seed}", "gold", "gold_alignment", "mrr", mrr))
        rows.append(_row(f"u2-s{seed}", "gold", "gold_alignment", "hits_at_k", mrr / 2, k=10.0))
        rows.append(_row(f"u2-s{seed}", "prf", "gold_alignment_prf", "f1", mrr / 4))
    return pd.DataFrame(rows)


def test_aggregate_groups_seeds_and_uses_sample_std():
    agg = unsup_summary.aggregate(_gold_frame())
    mrr = agg[(agg.method_group == "u2") & (agg.metric_name == "mrr")].iloc[0]
    assert mrr["n_seeds"] == 3
    assert mrr["mean"] == pytest.approx(0.34)
    assert mrr["std"] == pytest.approx(0.04)  # sample std (ddof=1) of 0.30 / 0.34 / 0.38
    assert set(agg.method_group) == {"u0", "u2"}


def test_markdown_shows_mean_std_single_seed_value_and_dashes_for_missing_metrics():
    agg = unsup_summary.aggregate(_gold_frame())
    md = unsup_summary.build_markdown(agg, ranking_eval=None, prf_eval=None, view="global")
    u2 = next(line for line in md.splitlines() if "(`u2`)" in line)
    u0 = next(line for line in md.splitlines() if "(`u0`)" in line)
    assert "0.3400 ± 0.0400" in u2
    assert "0.2000" in u0 and "±" not in u0  # one seed: value only
    assert "—" in u0  # U0 has no Hits@10 / F1 in this fixture
    assert "#Gold = 3,280" in md


def test_gold_free_run_renders_every_gold_cell_as_a_dash():
    rows = [
        _row("u0", "neigh", "neighborhood", "hit_at_k", 0.5, k=10.0, view="structural_r1"),
        _row("u0", "neigh", "neighborhood", "mrr", 0.25, k=10.0, view="structural_r1"),
    ]
    md = unsup_summary.build_markdown(unsup_summary.aggregate(pd.DataFrame(rows)), ranking_eval=None, prf_eval=None, view="global")
    gold_line = next(line for line in md.splitlines() if "(`u0`)" in line)
    assert gold_line.count("—") == 8  # Hits@1, Hits@10, MRR, P, R, F1, #Pred, #Correct
    assert "graph_1 · structural_r1 · hit@10" in md
    assert "0.5000" in md and "0.2500" in md


def test_paper_table_lists_present_rows_in_matrix_order_and_is_empty_without_gold():
    agg = unsup_summary.aggregate(_gold_frame())
    table = unsup_summary.build_paper_table(agg, ranking_eval=None, prf_eval=None, view="global")
    lines = table.splitlines()
    assert lines[0].startswith("| Variant | Pseudo-pairs | Loss | Hits@1")
    assert lines[2].startswith("| BERT (frozen) | – | – |")
    assert lines[3].startswith("| BERT + GCN (text + structure) | top-k | Triplet |")
    assert len(lines) == 4  # u1, u1-i, ... are absent from the fixture

    gold_free = unsup_summary.aggregate(pd.DataFrame([_row("u0", "neigh", "neighborhood", "mrr", 0.25)]))
    assert unsup_summary.build_paper_table(gold_free, ranking_eval=None, prf_eval=None, view="global") == ""
