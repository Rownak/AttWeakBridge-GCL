"""Unit tests for ``read_loss_curve``, ``render_loss_curve``, and ``discover_gcl_runs``."""

from __future__ import annotations

import json
from pathlib import Path

from cga.core.artifacts import write_latest
from cga.pipelines.plot.loss_curves import discover_gcl_runs, read_loss_curve, render_loss_curve


def _write_metrics_jsonl(run_dir: Path, n_epochs: int = 5) -> None:
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


def test_read_loss_curve_parses_expected_rows(tmp_path):
    run_dir = tmp_path / "run-1"
    _write_metrics_jsonl(run_dir, n_epochs=3)

    curve = read_loss_curve(run_dir)

    assert curve.epoch == [1, 2, 3]
    assert curve.loss == [1.0, 0.5, 1.0 / 3]
    assert curve.loss_text == [1.1, 0.55, 1.1 / 3]
    assert curve.loss_graph == [0.9, 0.45, 0.9 / 3]


def test_render_loss_curve_writes_one_png(tmp_path):
    run_dir = tmp_path / "run-1"
    _write_metrics_jsonl(run_dir, n_epochs=5)
    curve = read_loss_curve(run_dir)

    output_path = tmp_path / "figures" / "loss_curve_gcn.png"
    render_loss_curve(curve, output_path, title="gcn training loss")

    assert output_path.is_file()


def test_discover_gcl_runs_finds_every_model_id_with_latest(tmp_path):
    gcl_dir = tmp_path / "models" / "gcl" / "attack_cwe"
    for model_id in ("gcn", "gat"):
        model_dir = gcl_dir / model_id
        run_dir = model_dir / f"run-{model_id}"
        _write_metrics_jsonl(run_dir, n_epochs=2)
        write_latest(model_dir, run_dir.name)

    runs = discover_gcl_runs(gcl_dir)

    assert set(runs.keys()) == {"gcn", "gat"}
    assert runs["gcn"].name == "run-gcn"


def test_discover_gcl_runs_skips_model_id_without_successful_run(tmp_path):
    gcl_dir = tmp_path / "models" / "gcl" / "attack_cwe"
    good_model_dir = gcl_dir / "gcn"
    good_run_dir = good_model_dir / "run-gcn"
    _write_metrics_jsonl(good_run_dir, n_epochs=2)
    write_latest(good_model_dir, good_run_dir.name)

    # gat has no LATEST at all (e.g. training never succeeded).
    (gcl_dir / "gat").mkdir(parents=True, exist_ok=True)

    runs = discover_gcl_runs(gcl_dir)

    assert set(runs.keys()) == {"gcn"}


def test_discover_gcl_runs_skips_run_without_metrics_jsonl(tmp_path):
    gcl_dir = tmp_path / "models" / "gcl" / "attack_cwe"
    model_dir = gcl_dir / "gcn"
    run_dir = model_dir / "run-gcn"
    run_dir.mkdir(parents=True, exist_ok=True)  # no metrics.jsonl written
    write_latest(model_dir, run_dir.name)

    runs = discover_gcl_runs(gcl_dir)

    assert runs == {}


def test_discover_gcl_runs_ignores_stray_flat_latest_file(tmp_path):
    """A pre-model_id-partition flat LATEST at the dataset root is not itself a model_id dir."""
    gcl_dir = tmp_path / "models" / "gcl" / "attack_cwe"
    gcl_dir.mkdir(parents=True, exist_ok=True)
    (gcl_dir / "LATEST").write_text("some-old-run-id", encoding="utf-8")

    model_dir = gcl_dir / "gcn"
    run_dir = model_dir / "run-gcn"
    _write_metrics_jsonl(run_dir, n_epochs=2)
    write_latest(model_dir, run_dir.name)

    runs = discover_gcl_runs(gcl_dir)

    assert set(runs.keys()) == {"gcn"}


def test_discover_gcl_runs_on_missing_directory_returns_empty(tmp_path):
    assert discover_gcl_runs(tmp_path / "does_not_exist") == {}
