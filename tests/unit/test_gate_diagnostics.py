"""Tests for ``scripts/gate_diagnostics.py`` on a hand-built run and ingest fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gate_diagnostics", Path(__file__).parents[2] / "scripts" / "gate_diagnostics.py"
)
gd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gd)

IDS = ["a1", "a2", "a3", "a4", "c1", "c2", "c3"]
ROLES = ["graph_1"] * 4 + ["graph_2"] * 3
# degrees: a1=3 (a2, a3, a4), a2=1, a3=1, a4=1 ; c1=1, c2=1, c3=0
EDGES = pd.DataFrame({"src_id": ["a2", "a3", "a4", "c1", "a2"], "dst_id": ["a1", "a1", "a1", "c2", "a1"]})  # dup a2-a1


def _fixture(tmp_path: Path, gate: np.ndarray) -> tuple[Path, Path]:
    run, ingest = tmp_path / "run", tmp_path / "ingest"
    (run / "node_embeddings").mkdir(parents=True)
    ingest.mkdir()
    np.save(run / "node_embeddings" / "gate.npy", gate.astype(np.float32))
    pd.DataFrame({"row": range(len(IDS)), "node_id": IDS}).to_parquet(run / "node_embeddings" / "index.parquet")
    pd.DataFrame({"node_id": IDS, "graph_role": ROLES}).to_parquet(ingest / "nodes.parquet")
    EDGES.to_parquet(ingest / "edges.parquet")
    return run, ingest


def test_node_degree_counts_distinct_undirected_neighbours():
    assert gd.node_degree(EDGES, IDS).tolist() == [3, 1, 1, 1, 1, 1, 0]


def test_gate_stats_values():
    stats = gd.gate_stats(np.array([0.01, 0.5, 0.97, 0.99]), np.array([1, 2, 3, 4]))
    assert stats["n"] == 4 and stats["mean"] == pytest.approx(0.6175)
    assert stats["share_below_0.05"] == pytest.approx(0.25) and stats["share_above_0.95"] == pytest.approx(0.5)
    assert stats["spearman_degree"] == pytest.approx(1.0)


def test_diagnose_text_variant_per_role_and_degree_correlation(tmp_path):
    gate = np.array([0.9, 0.1, 0.2, 0.3, 0.5, 0.4, 0.6])
    run, ingest = _fixture(tmp_path, gate)
    report = gd.diagnose(run, ingest)
    assert set(report) == {"gate"} and set(report["gate"]) == {"graph_1", "graph_2"}
    assert report["gate"]["graph_1"]["n"] == 4 and report["gate"]["graph_2"]["n"] == 3
    assert report["gate"]["graph_1"]["spearman_degree"] > 0.5  # the highest-degree node has the highest gate
    assert report["gate"]["graph_1"]["mean"] == pytest.approx(0.375)


def test_diagnose_dual_variant_has_two_columns_and_markdown_rows(tmp_path):
    gate = np.stack([np.linspace(0.1, 0.7, 7), np.linspace(0.7, 0.1, 7)], axis=1)
    run, ingest = _fixture(tmp_path, gate)
    report = gd.diagnose(run, ingest)
    assert set(report) == {"text", "graph"}
    md = gd.to_markdown(report)
    assert md.count("\n") == 2 + 4  # header, separator, 2 columns x 2 roles


def test_diagnose_rejects_row_mismatch(tmp_path):
    run, ingest = _fixture(tmp_path, np.zeros(len(IDS) - 1))
    with pytest.raises(ValueError):
        gd.diagnose(run, ingest)


def test_plot_writes_figures(tmp_path):
    run, ingest = _fixture(tmp_path, np.linspace(0.1, 0.7, 7))
    out = tmp_path / "out"
    out.mkdir()
    gd.plot(run, ingest, out)
    assert (out / "gate_hist.png").stat().st_size > 0 and (out / "gate_vs_degree.png").stat().st_size > 0
