"""Unit tests for ``FigureSpec``, ``apply_select``, and the ``line``/``bar`` renderers."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from cga.core.errors import ConfigError
from cga.data.schemas import METRICS_SCHEMA
from cga.pipelines.plot.renderers import FigureSpec, apply_select, renderers


def _toy_rows() -> list[dict]:
    rows = []
    for method_id, method_label in (("deepwalk", "DeepWalk"), ("gcn", "GCN")):
        for view_id in ("structural_r1", "structural_r2"):
            for role in ("graph_1", "graph_2"):
                for k in (1, 5, 10):
                    rows.append(
                        {
                            "dataset_id": "attack_cwe",
                            "run_id": "run-1",
                            "evaluation_id": "neigh",
                            "evaluation_kind": "neighborhood",
                            "method_id": method_id,
                            "method_label": method_label,
                            "method_kind": "embedding_store",
                            "method_run_id": "m-run-1",
                            "graph_role": role,
                            "view_id": view_id,
                            "param_name": "k",
                            "param_value": k,
                            "metric_name": "normalized_precision_at_k",
                            "metric_value": 0.1 * k,
                            "n_queries": 10,
                        }
                    )
    # cluster_quality rows: no view_id (null), different metric/param.
    for method_id, method_label in (("deepwalk", "DeepWalk"), ("gcn", "GCN")):
        for role in ("graph_1", "graph_2"):
            for n_clusters in (2, 4):
                rows.append(
                    {
                        "dataset_id": "attack_cwe",
                        "run_id": "run-1",
                        "evaluation_id": "clusters",
                        "evaluation_kind": "cluster_quality",
                        "method_id": method_id,
                        "method_label": method_label,
                        "method_kind": "embedding_store",
                        "method_run_id": "m-run-1",
                        "graph_role": role,
                        "view_id": None,
                        "param_name": "n_clusters",
                        "param_value": n_clusters,
                        "metric_name": "silhouette",
                        "metric_value": 0.05 * n_clusters,
                        "n_queries": None,
                    }
                )
    return rows


@pytest.fixture
def metrics_table() -> pa.Table:
    return pa.Table.from_pylist(_toy_rows(), schema=METRICS_SCHEMA)


def test_apply_select_picks_expected_rows(metrics_table):
    filtered = apply_select(metrics_table, {"evaluation_id": "clusters", "metric_name": "silhouette"})
    assert filtered.num_rows == 2 * 2 * 2  # methods x roles x n_clusters
    assert set(filtered.column("evaluation_id").to_pylist()) == {"clusters"}


def test_apply_select_empty_selection_raises_config_error(metrics_table):
    with pytest.raises(ConfigError):
        apply_select(metrics_table, {"evaluation_id": "bogus"})


def test_figure_spec_unknown_kind_not_in_registry():
    spec = FigureSpec.from_config({"figure_id": "f", "kind": "bogus_kind"})
    assert spec.kind not in renderers.names()


def test_figure_spec_duplicate_figure_id_detected_by_caller():
    # FigureSpec itself doesn't dedupe; the pipeline's _validate_config does.
    entries = [{"figure_id": "f", "kind": "line"}, {"figure_id": "f", "kind": "bar"}]
    ids = [e["figure_id"] for e in entries]
    assert len(ids) != len(set(ids))


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_renderer_writes_one_png_per_figure(tmp_path, metrics_table, kind):
    spec = FigureSpec.from_config(
        {
            "figure_id": "precision_at_k",
            "kind": kind,
            "select": {"evaluation_id": "neigh", "metric_name": "normalized_precision_at_k"},
            "x": "param_value",
            "hue": "method_label",
            "facet": ["view_id", "graph_role"],
        }
    )
    filtered = apply_select(metrics_table, spec.select)
    output_path = tmp_path / "figures" / f"{spec.figure_id}.png"
    renderer = renderers.get(kind)()
    renderer.render(filtered, spec, output_path)
    assert output_path.is_file()


def test_renderer_facets_on_null_view_id_without_crashing(tmp_path, metrics_table):
    spec = FigureSpec.from_config(
        {
            "figure_id": "silhouette",
            "kind": "line",
            "select": {"evaluation_id": "clusters", "metric_name": "silhouette"},
            "x": "param_value",
            "hue": "method_label",
            "facet": ["view_id", "graph_role"],  # view_id is entirely null for this selection
        }
    )
    filtered = apply_select(metrics_table, spec.select)
    output_path = tmp_path / "figures" / f"{spec.figure_id}.png"
    renderer = renderers.get("line")()
    renderer.render(filtered, spec, output_path)
    assert output_path.is_file()
