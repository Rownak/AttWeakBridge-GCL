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


def _prefixed_frame(prefix: str, offset: float) -> pd.DataFrame:
    """U0 plus a three-seed U2 under ``prefix``; every metric is shifted by ``offset``."""
    rows = [
        _row(f"{prefix}u0", "gold", "gold_alignment", "mrr", 0.2 + offset),
        _row(f"{prefix}u0", "prf", "gold_alignment_prf", "f1", 0.1 + offset),
    ]
    for seed in (42, 43, 44):
        rows.append(_row(f"{prefix}u2-s{seed}", "gold", "gold_alignment", "mrr", 0.3 + offset + seed / 1e5))
        rows.append(_row(f"{prefix}u2-s{seed}", "prf", "gold_alignment_prf", "f1", 0.15 + offset))
    return pd.DataFrame(rows)


def test_prefixed_paper_table_maps_groups_to_rows_and_relabels_the_encoder():
    agg = unsup_summary.aggregate(_prefixed_frame("sap-", 0.1))
    table = unsup_summary.build_paper_table(
        agg, ranking_eval=None, prf_eval=None, view="global", prefix="sap-", text_label="SapBERT"
    )
    lines = table.splitlines()
    assert lines[2].startswith("| SapBERT (frozen) | – | – | —")  # no Hits columns in the fixture
    assert lines[3].startswith("| SapBERT + GCN (text + structure) | top-k | Triplet |")
    assert len(lines) == 4
    # Without the prefix nothing matches.
    assert unsup_summary.build_paper_table(agg, ranking_eval=None, prf_eval=None, view="global").count("\n") == 1


def test_comparison_reports_the_difference_of_means_second_minus_first():
    first = unsup_summary.aggregate(_prefixed_frame("", 0.0))
    second = unsup_summary.aggregate(_prefixed_frame("sap-", 0.1))
    table = unsup_summary.build_comparison(
        first, second, names=("BERT", "SapBERT"), prefixes=("", "sap-"), view="global"
    )
    u0 = next(line for line in table.splitlines() if line.startswith("| text encoder (frozen)"))
    cells = [c.strip() for c in u0.strip("|").split("|")]
    assert cells[3:6] == ["0.2000", "0.3000", "+0.1000"]  # MRR
    assert cells[6:9] == ["0.1000", "0.2000", "+0.1000"]  # F1
    assert len(table.splitlines()) == 4  # header, rule, u0, u2


def test_deepwalk_row_set_labels_rows_and_names_the_pseudo_pair_source():
    frame = _prefixed_frame("dw-", 0.0)
    own = [_row(f"dw-u2-own-s{s}", "gold", "gold_alignment", "mrr", 0.001) for s in (42, 43, 44)]
    agg = unsup_summary.aggregate(pd.concat([frame, pd.DataFrame(own)]))
    table = unsup_summary.build_paper_table(
        agg, ranking_eval=None, prf_eval=None, view="global", prefix="dw-", rows=unsup_summary.ROW_SETS["deepwalk"]
    )
    lines = table.splitlines()
    assert lines[2].startswith("| DeepWalk (frozen) | – | – |")
    assert lines[3].startswith("| GCN of DeepWalk | SapBERT top-k | Triplet |")
    assert lines[4].startswith("| GCN of DeepWalk | DeepWalk top-k | Triplet |")
    assert len(lines) == 5


def test_source_report_has_one_table_per_source_and_a_compact_table():
    def frame(mrr, n_gold):
        rows = []
        for seed in (42, 43):
            rows += [_row(f"ent-r2-s{seed}", "gold", "gold_alignment", "mrr", mrr + seed / 1000),
                     _row(f"ent-r2-s{seed}", "prf", "gold_alignment_prf", "f1", 0.5),
                     _row(f"ent-r2-s{seed}", "prf", "gold_alignment_prf", "n_gold", n_gold),
                     _row(f"ent-r2-s{seed}", "prf", "gold_alignment_prf", "n_unique_sources", n_gold / 2)]
        return unsup_summary.aggregate(pd.DataFrame(rows))

    text = unsup_summary.build_source_report({"stix": frame(0.3, 100.0), "union": frame(0.4, 200.0)})
    assert "## Gold source `stix`" in text and "## Gold source `union`" in text
    assert "## MRR / F1 per row and gold source" in text
    assert "| 100 | 0.5000 |" in text and "| 200 | 0.5000 |" in text  # #Gold and recall ceiling


def test_encoder_comparison_matches_rows_by_position_and_gives_the_difference():
    def frame(prefix, mrr, f1):
        rows = []
        for seed in (42, 43):
            rows += [_row(f"{prefix}-r2-s{seed}", "gold", "gold_alignment", "mrr", mrr),
                     _row(f"{prefix}-r2-s{seed}", "prf", "gold_alignment_prf", "f1", f1)]
        return unsup_summary.aggregate(pd.DataFrame(rows))

    first = {"stix": frame("ent", 0.010, 0.002)}
    second = {"stix": frame("ent-s", 0.030, 0.001)}
    text = unsup_summary.build_encoder_comparison(
        first, second, names=("A", "B"),
        rows=(unsup_summary.ATTACK_ENT_ROWS, unsup_summary.ATTACK_ENT_SRB_ROWS),
    )
    assert "| 0.0100 | 0.0300 | +0.0200 | 0.0020 | 0.0010 | -0.0010 |" in text
