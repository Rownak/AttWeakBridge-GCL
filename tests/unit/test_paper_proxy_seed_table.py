"""Tests for ``scripts/paper_proxy_seed_table.py``: grouping by seed suffix, mean and sample std, bolding, determinism."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "paper_proxy_seed_table", Path(__file__).parents[2] / "scripts" / "paper_proxy_seed_table.py"
)
seed_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(seed_mod)
base = seed_mod.base

# (method_id, label, MRR value, P@10 value, silhouette value)
_METHODS = [
    ("ref", "Frozen ref", 0.30, 0.10, 0.05),
    ("g-s42", "G gated seed 42", 0.40, 0.20, 0.10),
    ("g-s43", "G gated seed 43", 0.50, 0.30, 0.20),
    ("g-s44", "G gated seed 44", 0.60, 0.40, 0.30),
]


def _fixture() -> pd.DataFrame:
    rows = []
    for mid, label, mrr, prec, sil in _METHODS:
        common = {"method_id": mid, "method_label": label}
        for view in base.VIEWS:
            for role, _ in base.ROLES:
                n = {**common, "evaluation_kind": "neighborhood", "view_id": view, "graph_role": role, "param_value": 10}
                rows.append({**n, "metric_name": "mrr", "metric_value": mrr})
                rows.append({**n, "metric_name": "normalized_precision_at_k", "metric_value": prec})
        for role, _ in base.ROLES:
            rows.append({**common, "evaluation_kind": "cluster_quality", "view_id": None, "graph_role": role,
                         "param_value": 2, "metric_name": "silhouette", "metric_value": sil})
    return pd.DataFrame(rows)


def test_seed_groups_strip_the_seed_from_the_label():
    groups = seed_mod.seed_groups(_fixture())
    assert groups == {"Frozen ref": ["ref"], "G gated": ["g-s42", "g-s43", "g-s44"]}


def test_mean_and_sample_std_over_seeds():
    mean, std, n = seed_mod.build_seed_table(_fixture())
    assert n == {"Frozen ref": 1, "G gated": 3}
    assert mean.loc["G gated", "textual CWE MRR"] == pytest.approx(0.5)
    assert std.loc["G gated", "textual CWE MRR"] == pytest.approx(0.1)  # sample std (ddof=1) of 0.4, 0.5, 0.6
    assert mean.loc["G gated", "Silhouette"] == pytest.approx(0.2)
    assert mean.loc["Frozen ref", "Silhouette"] == pytest.approx(0.05)
    assert std.loc["Frozen ref"].isna().all()


def test_markdown_shows_std_for_seeded_rows_and_bolds_best_mean():
    mean, std, n = seed_mod.build_seed_table(_fixture())
    md = seed_mod.to_markdown(mean, std, n)
    gated = next(line for line in md.splitlines() if line.startswith("| G gated (3)"))
    ref = next(line for line in md.splitlines() if line.startswith("| Frozen ref (1)"))
    assert "**0.5000 ± 0.1000**" in gated
    assert "±" not in ref and "0.3000" in ref


def test_write_seed_table_is_deterministic_and_csv_has_mean_and_std(tmp_path):
    _fixture().to_parquet(tmp_path / "metrics.parquet")
    md, csv = seed_mod.write_seed_table(tmp_path)
    first = (md.read_bytes(), csv.read_bytes())
    seed_mod.write_seed_table(tmp_path)
    assert (md.read_bytes(), csv.read_bytes()) == first
    header = csv.read_text().splitlines()[0]
    assert header.startswith("method,n_seeds,structural_r1 ATT&CK MRR mean,structural_r1 ATT&CK MRR std")
