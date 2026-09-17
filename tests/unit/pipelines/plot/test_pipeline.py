"""End-to-end tests for the ``plot`` pipeline against a fabricated ``evaluate`` run."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

import cga.pipelines  # noqa: F401  (registers pipelines)
from cga.core.artifacts import write_latest
from cga.core.errors import ConfigError
from cga.core.io import read_json, write_parquet
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.schemas import METRICS_SCHEMA

NEIGHBORHOOD_FIGURE = {
    "figure_id": "precision_at_k",
    "kind": "line",
    "select": {"evaluation_id": "neigh", "metric_name": "normalized_precision_at_k"},
    "x": "param_value",
    "hue": "method_label",
    "facet": ["view_id", "graph_role"],
}

CLUSTER_FIGURE = {
    "figure_id": "silhouette",
    "kind": "line",
    "select": {"evaluation_id": "clusters", "metric_name": "silhouette"},
    "x": "param_value",
    "hue": "method_label",
    "facet": ["graph_role"],
}


def _metrics_rows() -> list[dict]:
    rows = []
    for method_id, method_label in (("deepwalk", "DeepWalk"), ("gcn", "GCN")):
        for role in ("graph_1", "graph_2"):
            for k in (1, 5):
                rows.append(
                    {
                        "dataset_id": "attack_cwe",
                        "run_id": "eval-run-1",
                        "evaluation_id": "neigh",
                        "evaluation_kind": "neighborhood",
                        "method_id": method_id,
                        "method_label": method_label,
                        "method_kind": "embedding_store",
                        "method_run_id": "m-run-1",
                        "graph_role": role,
                        "view_id": "structural_r1",
                        "param_name": "k",
                        "param_value": k,
                        "metric_name": "normalized_precision_at_k",
                        "metric_value": 0.1 * k,
                        "n_queries": 10,
                    }
                )
    for method_id, method_label in (("deepwalk", "DeepWalk"), ("gcn", "GCN")):
        for role in ("graph_1", "graph_2"):
            rows.append(
                {
                    "dataset_id": "attack_cwe",
                    "run_id": "eval-run-1",
                    "evaluation_id": "clusters",
                    "evaluation_kind": "cluster_quality",
                    "method_id": method_id,
                    "method_label": method_label,
                    "method_kind": "embedding_store",
                    "method_run_id": "m-run-1",
                    "graph_role": role,
                    "view_id": None,
                    "param_name": "n_clusters",
                    "param_value": 2,
                    "metric_name": "silhouette",
                    "metric_value": 0.2,
                    "n_queries": None,
                }
            )
    return rows


def _write_evaluate_run(evaluate_dir: Path) -> None:
    table = pa.Table.from_pylist(_metrics_rows(), schema=METRICS_SCHEMA)
    write_parquet(table, evaluate_dir / "metrics.parquet", schema=METRICS_SCHEMA)


def _build_config(tmp_path: Path, *, figures: list[dict] | None = None, loss_curves_enabled: bool = False) -> dict:
    return {
        "project": "cga",
        "seed": 42,
        "device": "cpu",
        "paths": {
            "raw": str(tmp_path / "raw"),
            "processed": str(tmp_path / "processed"),
            "artifacts": str(tmp_path / "artifacts"),
            "models": str(tmp_path / "models"),
        },
        "logging": {"level": "INFO", "file": False},
        "run": {"name": None},
        "dataset": {"id": "attack_cwe"},
        "plot": {
            "inputs": {"evaluate_run": "latest"},
            "figures": figures if figures is not None else [],
            "format": {"ext": "png", "dpi": 200},
            "loss_curves": {"enabled": loss_curves_enabled},
        },
    }


def _setup_evaluate_run(config: dict) -> Path:
    dataset_id = config["dataset"]["id"]
    evaluate_dir = Path(config["paths"]["artifacts"]) / "evaluations" / dataset_id / "20260101T000000Z-aaaaaaaa"
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    _write_evaluate_run(evaluate_dir)
    write_latest(evaluate_dir.parent, evaluate_dir.name)
    return evaluate_dir


def _setup_gcl_run(config: dict, model_id: str, *, n_epochs: int = 3) -> Path:
    dataset_id = config["dataset"]["id"]
    model_dir = Path(config["paths"]["models"]) / "gcl" / dataset_id / model_id
    run_dir = model_dir / f"run-{model_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for epoch in range(1, n_epochs + 1):
            record = {
                "epoch": epoch,
                "loss": 1.0 / epoch,
                "loss_text": 1.1 / epoch,
                "loss_graph": 0.9 / epoch,
                "lr": 0.001,
                "elapsed_s": 0.01,
            }
            f.write(json.dumps(record) + "\n")
    write_latest(model_dir, run_dir.name)
    return run_dir


def _run_plot(config: dict) -> RunContext:
    pipeline_cls = pipelines.get("plot")
    pipeline = pipeline_cls()
    run_root = Path(config["paths"]["artifacts"]) / "plots" / config["dataset"]["id"]
    ctx = RunContext.create(
        pipeline_name="plot",
        pipeline_version="0.0.0-test",
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=run_root,
    )
    pipeline.run(ctx)
    return ctx


def test_plot_end_to_end_writes_one_png_per_figure(tmp_path):
    config = _build_config(tmp_path, figures=[NEIGHBORHOOD_FIGURE, CLUSTER_FIGURE])
    _setup_evaluate_run(config)

    ctx = _run_plot(config)

    assert (ctx.run_dir / "figures" / "precision_at_k.png").is_file()
    assert (ctx.run_dir / "figures" / "silhouette.png").is_file()

    manifest = read_json(ctx.run_dir / "manifest.json")
    assert manifest["status"] == "success"
    assert set(manifest["summary"]["figures"]) == {"precision_at_k", "silhouette"}
    assert any(inp["name"] == "evaluate_run" for inp in manifest["inputs"])
    assert any(o["name"] == "figure_precision_at_k" for o in manifest["outputs"])
    assert any(o["name"] == "figure_silhouette" for o in manifest["outputs"])


def test_plot_empty_selection_raises_config_error(tmp_path):
    figure = dict(NEIGHBORHOOD_FIGURE)
    figure["select"] = {"evaluation_id": "does_not_exist"}
    config = _build_config(tmp_path, figures=[figure])
    _setup_evaluate_run(config)

    with pytest.raises(ConfigError):
        _run_plot(config)


def test_plot_unknown_kind_raises_config_error(tmp_path):
    figure = dict(NEIGHBORHOOD_FIGURE)
    figure["kind"] = "bogus"
    config = _build_config(tmp_path, figures=[figure])
    _setup_evaluate_run(config)

    with pytest.raises(ConfigError):
        _run_plot(config)


def test_plot_duplicate_figure_id_raises_config_error(tmp_path):
    config = _build_config(tmp_path, figures=[NEIGHBORHOOD_FIGURE, dict(NEIGHBORHOOD_FIGURE)])
    _setup_evaluate_run(config)

    with pytest.raises(ConfigError):
        _run_plot(config)


def test_plot_empty_figures_raises_config_error(tmp_path):
    config = _build_config(tmp_path, figures=[])
    _setup_evaluate_run(config)

    with pytest.raises(ConfigError):
        _run_plot(config)


def test_plot_facets_on_null_view_id_without_crashing(tmp_path):
    config = _build_config(tmp_path, figures=[CLUSTER_FIGURE])
    _setup_evaluate_run(config)

    ctx = _run_plot(config)
    assert (ctx.run_dir / "figures" / "silhouette.png").is_file()


def test_plot_records_evaluate_run_as_manifest_input(tmp_path):
    config = _build_config(tmp_path, figures=[NEIGHBORHOOD_FIGURE])
    evaluate_dir = _setup_evaluate_run(config)

    ctx = _run_plot(config)
    manifest = read_json(ctx.run_dir / "manifest.json")
    evaluate_input = next(inp for inp in manifest["inputs"] if inp["name"] == "evaluate_run")
    assert evaluate_input["run_id"] == evaluate_dir.name


def test_plot_loss_curves_disabled_by_default_produces_no_loss_output(tmp_path):
    config = _build_config(tmp_path, figures=[NEIGHBORHOOD_FIGURE])
    _setup_evaluate_run(config)
    _setup_gcl_run(config, "gcn")

    ctx = _run_plot(config)

    assert not (ctx.run_dir / "figures" / "loss_curve_gcn.png").exists()
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert not any(inp["name"].startswith("gcl_run:") for inp in manifest["inputs"])
    assert manifest["summary"]["loss_curve_model_ids"] == []


def test_plot_loss_curves_enabled_writes_one_png_per_model_id(tmp_path):
    config = _build_config(tmp_path, figures=[], loss_curves_enabled=True)
    _setup_gcl_run(config, "gcn")
    _setup_gcl_run(config, "gat")

    ctx = _run_plot(config)

    assert (ctx.run_dir / "figures" / "loss_curve_gcn.png").is_file()
    assert (ctx.run_dir / "figures" / "loss_curve_gat.png").is_file()
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert set(manifest["summary"]["loss_curve_model_ids"]) == {"gcn", "gat"}
    assert any(inp["name"] == "gcl_run:gcn" for inp in manifest["inputs"])
    assert any(inp["name"] == "gcl_run:gat" for inp in manifest["inputs"])
    assert any(o["name"] == "figure_loss_curve_gcn" for o in manifest["outputs"])


def test_plot_loss_curves_alongside_metrics_figures(tmp_path):
    config = _build_config(tmp_path, figures=[NEIGHBORHOOD_FIGURE], loss_curves_enabled=True)
    _setup_evaluate_run(config)
    _setup_gcl_run(config, "gcn")

    ctx = _run_plot(config)

    assert (ctx.run_dir / "figures" / "precision_at_k.png").is_file()
    assert (ctx.run_dir / "figures" / "loss_curve_gcn.png").is_file()


def test_plot_loss_curves_enabled_with_empty_figures_does_not_raise(tmp_path):
    """loss_curves alone is a valid run: no evaluate_run figures required."""
    config = _build_config(tmp_path, figures=[], loss_curves_enabled=True)
    _setup_gcl_run(config, "gcn")

    ctx = _run_plot(config)

    assert (ctx.run_dir / "figures" / "loss_curve_gcn.png").is_file()
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert not any(inp["name"] == "evaluate_run" for inp in manifest["inputs"])


def test_plot_loss_curves_skips_model_id_without_successful_run(tmp_path):
    config = _build_config(tmp_path, figures=[], loss_curves_enabled=True)
    _setup_gcl_run(config, "gcn")
    # gat has a model_id directory but no successful (LATEST) run.
    gat_dir = Path(config["paths"]["models"]) / "gcl" / config["dataset"]["id"] / "gat"
    gat_dir.mkdir(parents=True, exist_ok=True)

    ctx = _run_plot(config)

    assert (ctx.run_dir / "figures" / "loss_curve_gcn.png").is_file()
    assert not (ctx.run_dir / "figures" / "loss_curve_gat.png").exists()
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert manifest["summary"]["loss_curve_model_ids"] == ["gcn"]
