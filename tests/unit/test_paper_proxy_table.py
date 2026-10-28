"""Tests for ``scripts/paper_proxy_table.py``: column values, bolding of the best cell, and deterministic output."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "paper_proxy_table", Path(__file__).parents[2] / "scripts" / "paper_proxy_table.py"
)
table_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(table_mod)

_MRR = {"a": 0.5, "b": 0.25}
_PREC = {"a": 0.2, "b": 0.4}


def _fixture() -> pd.DataFrame:
    """Hand-built long metrics frame: two methods, all views / roles, k in (1, 10), n_clusters in (2, 4)."""
    rows = []
    for mid, label in (("a", "Method A"), ("b", "Method B")):
        common = {"method_id": mid, "method_label": label}
        for view in table_mod.VIEWS:
            for role, _ in table_mod.ROLES:
                for k in (1, 10):
                    base = {**common, "evaluation_kind": "neighborhood", "view_id": view, "graph_role": role,
                            "param_value": k}
                    rows.append({**base, "metric_name": "mrr", "metric_value": _MRR[mid]})
                    # precision differs between k=1 and k=10 so that picking the wrong k is detected
                    rows.append({**base, "metric_name": "normalized_precision_at_k",
                                 "metric_value": _PREC[mid] * (1.0 if k == 10 else 2.0)})
        for role, _ in table_mod.ROLES:
            for n, value in ((2, 0.1), (4, 0.3)):
                rows.append({**common, "evaluation_kind": "cluster_quality", "view_id": None, "graph_role": role,
                             "param_value": n, "metric_name": "silhouette",
                             "metric_value": value + (0.1 if mid == "b" else 0.0)})
    return pd.DataFrame(rows)


def test_build_table_values():
    t = table_mod.build_table(_fixture())
    assert list(t.index) == ["Method A", "Method B"]
    assert t.shape == (2, 13)
    assert t.loc["Method A", "textual CWE MRR"] == pytest.approx(0.5)
    assert t.loc["Method B", "structural_r2 ATT&CK P@10"] == pytest.approx(0.4)  # k=10, not k=1
    assert t.loc["Method A", "Silhouette"] == pytest.approx(0.2)
    assert t.loc["Method B", "Silhouette"] == pytest.approx(0.3)


def test_markdown_bolds_best_per_column():
    md = table_mod.to_markdown(table_mod.build_table(_fixture()))
    a_line = next(line for line in md.splitlines() if line.startswith("| Method A"))
    b_line = next(line for line in md.splitlines() if line.startswith("| Method B"))
    assert "**0.5000**" in a_line and "**0.5000**" not in b_line  # A best in MRR
    assert "**0.4000**" in b_line  # B best in precision
    assert "**0.3000**" in b_line  # B best in silhouette


def test_missing_cell_raises():
    df = _fixture()
    df = df[~((df["method_id"] == "a") & (df["view_id"] == "textual") & (df["graph_role"] == "graph_2"))]
    with pytest.raises(ValueError, match="missing cell"):
        table_mod.build_table(df)


def test_write_table_is_deterministic(tmp_path):
    _fixture().to_parquet(tmp_path / "metrics.parquet")
    md, csv = table_mod.write_table(tmp_path)
    first = (md.read_bytes(), csv.read_bytes())
    table_mod.write_table(tmp_path)
    assert (md.read_bytes(), csv.read_bytes()) == first
    assert csv.read_text().splitlines()[0].startswith("method,structural_r1 ATT&CK MRR")
