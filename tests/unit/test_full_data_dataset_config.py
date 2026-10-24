"""The Sprint 6.3 full-data dataset config differs from ``ncit_doid_srga`` only by its id and the absent val carve."""

from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2] / "configs" / "datasets"


def _load(name: str) -> dict:
    return yaml.safe_load((_ROOT / f"{name}.yaml").read_text(encoding="utf-8"))["dataset"]


def test_full_data_config_has_no_val_split_and_own_id():
    full, base = _load("ncit_doid_srga_full"), _load("ncit_doid_srga")
    assert full["id"] == "ncit_doid_srga_full"
    assert "val_split" not in full["alignments"]
    assert "val_split" in base["alignments"]  # the train-only lineage is unchanged


def test_full_data_config_matches_the_train_only_config_otherwise():
    full, base = _load("ncit_doid_srga_full"), _load("ncit_doid_srga")
    full["id"] = base["id"]
    full["alignments"] = {**full["alignments"], "val_split": base["alignments"]["val_split"]}
    assert full == base
