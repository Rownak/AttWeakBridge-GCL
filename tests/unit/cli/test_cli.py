"""Tests for the ``cga`` CLI against a temporary configs/ + data/raw/ tree."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import cga.cli as cli_module
from cga.cli import build_parser, main

FIXTURES = Path(__file__).parents[1] / "adapters" / "fixtures"


@pytest.fixture
def cli_project(tmp_path, monkeypatch):
    """Build a minimal configs/ tree (pointed at the small attack_cwe fixtures) and chdir into it."""
    configs_dir = tmp_path / "configs"
    (configs_dir / "datasets").mkdir(parents=True)
    (configs_dir / "pipelines").mkdir(parents=True)

    raw_dir = tmp_path / "data" / "raw" / "attack_cwe"
    raw_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "stix_sample.json", raw_dir / "stix_sample.json")
    shutil.copy(FIXTURES / "cwe_sample.xml", raw_dir / "cwe_sample.xml")

    (configs_dir / "base.yaml").write_text(
        f"""
project: cga
seed: 42
device: cpu
paths:
  raw: {raw_dir.as_posix()}
  processed: {(tmp_path / "data" / "processed").as_posix()}
  artifacts: {(tmp_path / "artifacts").as_posix()}
  models: {(tmp_path / "models").as_posix()}
logging:
  level: INFO
  file: false
run:
  name: null
""",
        encoding="utf-8",
    )
    (configs_dir / "datasets" / "attack_cwe.yaml").write_text(
        """
dataset:
  id: attack_cwe
  adapter: attack_cwe
  root: ${paths.raw}
  adapter_config:
    attack:
      file: stix_sample.json
    cwe:
      file: cwe_sample.xml
      version: "4.12"
      edge_relations: [ChildOf]
  text_variants:
    description:
      attack: [description]
      cwe: [name, description, extended_description]
  alignments: null
""",
        encoding="utf-8",
    )
    (configs_dir / "pipelines" / "ingest.yaml").write_text(
        "ingest:\n  text_variant_selection: null\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_CONFIG_ROOT", configs_dir)
    return tmp_path


def test_list_shows_registered_pipelines_and_adapters(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ingest" in out
    assert "attack_cwe" in out
    assert "ncit_doid" in out


def test_run_dry_run_does_not_write_output(cli_project, capsys):
    exit_code = main(["run", "ingest", "--dataset", "attack_cwe", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Would run pipeline 'ingest'" in out
    assert not (cli_project / "data" / "processed").exists()


def test_run_unknown_dataset_exits_nonzero(cli_project, capsys):
    exit_code = main(["run", "ingest", "--dataset", "nope", "--dry-run"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Unknown dataset" in err


def test_run_writes_manifest_and_latest(cli_project, capsys):
    exit_code = main(["run", "ingest", "--dataset", "attack_cwe"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "succeeded" in out

    lineage_dir = cli_project / "data" / "processed" / "attack_cwe"
    latest_run_id = (lineage_dir / "LATEST").read_text(encoding="utf-8").strip()
    assert (lineage_dir / latest_run_id / "manifest.json").is_file()


def test_validate_succeeds_on_a_successful_run(cli_project, capsys):
    main(["run", "ingest", "--dataset", "attack_cwe"])
    capsys.readouterr()

    lineage_dir = cli_project / "data" / "processed" / "attack_cwe"
    latest_run_id = (lineage_dir / "LATEST").read_text(encoding="utf-8").strip()

    exit_code = main(["validate", str(lineage_dir / latest_run_id)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_inspect_prints_manifest_chain(cli_project, capsys):
    main(["run", "ingest", "--dataset", "attack_cwe"])
    capsys.readouterr()

    lineage_dir = cli_project / "data" / "processed" / "attack_cwe"
    latest_run_id = (lineage_dir / "LATEST").read_text(encoding="utf-8").strip()

    exit_code = main(["inspect", str(lineage_dir / latest_run_id)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ingest run=" in out
    assert "status=success" in out


def test_config_show_prints_resolved_config(cli_project, capsys):
    exit_code = main(["config", "show", "ingest", "--dataset", "attack_cwe"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"dataset"' in out
    assert '"attack_cwe"' in out


def test_build_parser_requires_a_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_sweep_runs_every_combination_and_partitions_latest(cli_project, capsys):
    spec_path = cli_project / "sweep_seed.yaml"
    spec_path.write_text(
        "pipeline: ingest\ndataset: attack_cwe\nsweep:\n  seed: [1, 2]\n", encoding="utf-8"
    )

    exit_code = main(["sweep", "--spec", str(spec_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "2/2 combinations succeeded" in out

    lineage_dir = cli_project / "data" / "processed" / "attack_cwe"
    latest_run_id = (lineage_dir / "LATEST").read_text(encoding="utf-8").strip()
    manifest = (lineage_dir / latest_run_id / "manifest.json").read_text(encoding="utf-8")
    assert '"sweep_point_id"' in manifest
    assert '"seed2"' in manifest


def test_sweep_continues_past_a_failed_combination(cli_project, capsys):
    spec_path = cli_project / "sweep_bad.yaml"
    spec_path.write_text(
        "pipeline: ingest\ndataset: attack_cwe\nsweep:\n"
        "  dataset.adapter: [attack_cwe, no_such_adapter]\n",
        encoding="utf-8",
    )

    exit_code = main(["sweep", "--spec", str(spec_path)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "1/2 combinations succeeded" in out

    lineage_dir = cli_project / "data" / "processed" / "attack_cwe"
    assert (lineage_dir / "LATEST").is_file()  # the successful point still wrote LATEST
