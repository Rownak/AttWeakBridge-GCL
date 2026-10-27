"""Tests for the generic gold CSV reader and grouped split."""

from __future__ import annotations

import hashlib

import pytest

from cga.core.errors import ConfigError
from cga.data.gold_csv import GoldAlignmentConfig, assign_grouped_split, group_id, read_gold_csv

HEADER = "attack_id_cited,attack_id_current,capec_id,cwe_id,edge_source\n"
SRC = {t: f"attack:{t}" for t in ("T1003", "T1003.001", "T1059", "T1078")}
TGT = {c: f"cwe:{c[4:]}" for c in ("CWE-20", "CWE-79", "CWE-89")}
REGEX = r"^(T\d+)"


def _cfg(source="union", fractions=None, seed=42) -> GoldAlignmentConfig:
    split = {"seed": seed, "group_regex": REGEX}
    if fractions is not None:
        split["fractions"] = fractions
    return GoldAlignmentConfig(
        source=source,
        sources={"stix": "a.csv", "union": "b.csv"},
        columns={"source": "attack_id_current", "target": "cwe_id"},
        split=split,
    )


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(HEADER + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")
    return p


ROWS = [
    ("T1003", "T1003", "CAPEC-1", "CWE-20", "capec-3.9"),
    ("T1003", "T1003", "CAPEC-2", "CWE-20", "v12.1"),  # same pair, other path
    ("T1003.001", "T1003.001", "CAPEC-1", "CWE-79", "capec-3.9"),
    ("T1059", "T1059", "CAPEC-3", "CWE-89", "capec-3.9"),
    ("T1078", "T1078", "CAPEC-4", "CWE-20", "capec-3.9"),
    ("T9999", "T9999", "CAPEC-5", "CWE-20", "capec-3.9"),  # unknown technique
    ("T1059", "T1059", "CAPEC-6", "CWE-9999", "capec-3.9"),  # unknown CWE
]


def test_unknown_source_key_raises():
    with pytest.raises(ConfigError):
        _cfg(source="bogus").selected_path()


def test_each_source_key_selects_its_file():
    assert _cfg("stix").selected_path() == "a.csv"
    assert _cfg("union").selected_path() == "b.csv"


def test_dedupe_drops_and_counts(tmp_path):
    p = _write(tmp_path, "b.csv", ROWS)
    res = read_gold_csv(p, cfg=_cfg(), source_ids=SRC, target_ids=TGT, root=tmp_path)
    assert len(res.alignments) == 4
    assert res.report["dropped"] == {"source_not_in_graph": 1, "target_not_in_graph": 1, "duplicate": 1}
    assert res.report["n_csv_rows"] == 7
    pairs = {(a.node_1_id, a.node_2_id) for a in res.alignments}
    assert ("attack:T9999", "cwe:20") not in pairs  # never guessed


def test_provenance_one_row_per_alignment(tmp_path):
    p = _write(tmp_path, "b.csv", ROWS)
    res = read_gold_csv(p, cfg=_cfg(), source_ids=SRC, target_ids=TGT, root=tmp_path)
    assert [a.alignment_id for a in res.alignments] == [r["alignment_id"] for r in res.provenance]
    merged = next(r for r in res.provenance if r["n_csv_rows"] == 2)
    assert merged["capec_ids"] == ["CAPEC-1", "CAPEC-2"]
    assert merged["edge_source"] == "capec-3.9|v12.1"
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert {r["source_sha256"] for r in res.provenance} == {sha}
    assert res.report["source_sha256"] == sha
    assert {r["source_file"] for r in res.provenance} == {"b.csv"}


def test_default_split_is_all_test(tmp_path):
    p = _write(tmp_path, "b.csv", ROWS)
    res = read_gold_csv(p, cfg=_cfg(), source_ids=SRC, target_ids=TGT, root=tmp_path)
    assert {a.split for a in res.alignments} == {"test"}
    assert res.report["n_pairs_by_split"] == {"train": 0, "val": 0, "test": 4}


def test_non_default_split_keeps_families_together(tmp_path):
    p = _write(tmp_path, "b.csv", ROWS)
    cfg = _cfg(fractions=[0.5, 0.0, 0.5])
    res = read_gold_csv(p, cfg=cfg, source_ids=SRC, target_ids=TGT, root=tmp_path)
    by_group: dict[str, set[str]] = {}
    for a in res.alignments:
        by_group.setdefault(group_id(a.node_1_id.split(":")[1], REGEX), set()).add(a.split)
    assert all(len(s) == 1 for s in by_group.values())  # T1003 and T1003.001 together


def test_split_seed_determinism_and_difference():
    groups = [f"T{i}" for i in range(200)]
    frac = [0.6, 0.2, 0.2]
    a = assign_grouped_split(groups, fractions=frac, seed=42)
    assert a == assign_grouped_split(reversed(groups), fractions=frac, seed=42)
    assert a != assign_grouped_split(groups, fractions=frac, seed=43)
    counts = {s: list(a.values()).count(s) for s in ("train", "val", "test")}
    assert 100 < counts["train"] < 140 and counts["test"] > 20


def test_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        read_gold_csv(p, cfg=_cfg(), source_ids=SRC, target_ids=TGT, root=tmp_path)


def test_bad_fractions_raise():
    with pytest.raises(ConfigError):
        assign_grouped_split(["a"], fractions=[0.5, 0.5], seed=1)
