"""Tests for ArtifactRef resolution of latest / explicit run_id / path."""

from __future__ import annotations

import pytest

from cga.core.artifacts import ArtifactRef, read_latest, write_latest
from cga.core.errors import MissingInputError


def test_resolve_latest_reads_latest_file(tmp_path):
    run_dir = tmp_path / "20260101T000000Z-abcd1234"
    run_dir.mkdir()
    write_latest(tmp_path, run_dir.name)

    ref = ArtifactRef(lineage_dir=tmp_path, ref="latest")
    assert ref.resolve() == run_dir
    assert ref.resolved_run_id() == run_dir.name


def test_resolve_explicit_run_id(tmp_path):
    run_dir = tmp_path / "some-run-id"
    run_dir.mkdir()

    ref = ArtifactRef(lineage_dir=tmp_path, ref="some-run-id")
    assert ref.resolve() == run_dir


def test_resolve_latest_missing_file_raises(tmp_path):
    ref = ArtifactRef(lineage_dir=tmp_path, ref="latest")
    with pytest.raises(MissingInputError):
        ref.resolve()


def test_resolve_nonexistent_run_dir_raises(tmp_path):
    ref = ArtifactRef(lineage_dir=tmp_path, ref="does-not-exist")
    with pytest.raises(MissingInputError):
        ref.resolve()


def test_write_latest_then_read_latest_roundtrip(tmp_path):
    write_latest(tmp_path, "run-123")
    assert read_latest(tmp_path) == "run-123"


def test_read_latest_missing_returns_none(tmp_path):
    assert read_latest(tmp_path) is None
