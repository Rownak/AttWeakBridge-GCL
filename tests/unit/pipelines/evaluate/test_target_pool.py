"""Tests for the ``target_pool`` evaluator option and the pool / random-store scripts (Sprint 10.6).

Fixture of ``test_one_to_many_gold``: sources s1 0 deg, s2 90 deg; targets a 10, d 15, b 20, c 80. Test gold
(s1,a) (s1,b) (s2,c); train pair (s1,d). With ``sibling_filter_splits: [test]`` and no pool, (s1,b) ranks 2nd behind
the train sibling d (MRR 5/6). A pool {a,b,c} removes d from the candidates, so (s1,b) ranks 1st and MRR is 1.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cga.core.errors import ConfigError
from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.evaluators.target_pool import pool_sha256

from .test_one_to_many_gold import _res


def _pool_file(tmp_path: Path, ids: list[str], *, sha: str | None = None) -> str:
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"node_ids": ids, "sha256": sha or pool_sha256(ids)}), encoding="utf-8")
    return str(path)


def _rank(**params) -> dict:
    rows = evaluators.get("gold_alignment")().evaluate(
        _res(), "gold", dict(ks=[1, 2], modes=["global"], sibling_filter_splits=["test"], **params)
    )
    return {(r.metric_name, r.param_value): r.metric_value for r in rows}


def _prf(**params) -> dict:
    rows = evaluators.get("gold_alignment_prf")().evaluate(
        _res(), "prf", dict(split="test", assignment="greedy_one_to_one", top_k=3, threshold=0.0, **params)
    )
    return {r.metric_name: r.metric_value for r in rows}


def test_pool_off_equals_default_and_pool_removes_the_train_sibling(tmp_path):
    assert _rank()[("mrr", None)] == pytest.approx(5 / 6)
    pooled = _rank(target_pool=_pool_file(tmp_path, ["t:a", "t:b", "t:c"]))
    assert pooled[("mrr", None)] == pytest.approx(1.0)
    assert pooled[("hits_at_k", 1)] == pytest.approx(1.0)


def test_full_graph_2_pool_equals_no_pool(tmp_path):
    full = _pool_file(tmp_path, ["t:a", "t:b", "t:c", "t:d"])
    assert _rank(target_pool=full) == _rank()
    assert _prf(target_pool=full) == _prf()


def test_prf_with_pool_keeps_one_to_one_and_n_gold(tmp_path):
    out = _prf(target_pool=_pool_file(tmp_path, ["t:a", "t:b", "t:c"]))
    assert (out["n_predicted"], out["n_correct"], out["n_gold"]) == (2, 2, 3)
    assert out["f1"] == pytest.approx(4 / 5)


def test_gold_target_outside_the_pool_raises(tmp_path):
    bad = _pool_file(tmp_path, ["t:a", "t:c"])  # t:b is a test target
    with pytest.raises(ConfigError, match="outside target_pool"):
        _rank(target_pool=bad)
    with pytest.raises(ConfigError, match="outside target_pool"):
        _prf(target_pool=bad)


def test_missing_or_tampered_pool_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        _rank(target_pool=str(tmp_path / "nope.json"))
    with pytest.raises(ConfigError, match="sha256"):
        _rank(target_pool=_pool_file(tmp_path, ["t:a", "t:b", "t:c"], sha="0" * 64))


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[4] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ingest(tmp_path: Path, name: str, targets: list[str]) -> Path:
    d = tmp_path / name
    d.mkdir()
    pq.write_table(pa.table({"node_2_id": targets}), d / "alignments.parquet")
    return d


def test_make_target_pool_builds_union_and_checks_the_others(tmp_path):
    mod = _script("make_target_pool")
    union = _ingest(tmp_path, "union", ["cwe:1", "cwe:2", "cwe:2", "cwe:3"])
    payload = mod.build_pool(union, [_ingest(tmp_path, "stix", ["cwe:1"]), _ingest(tmp_path, "inter", ["cwe:2", "cwe:3"])])
    assert payload["node_ids"] == ["cwe:1", "cwe:2", "cwe:3"] and payload["sha256"] == pool_sha256(payload["node_ids"])
    with pytest.raises(ValueError, match="outside the pool"):
        mod.build_pool(union, [_ingest(tmp_path, "capec", ["cwe:9"])])


def test_make_random_store_is_seeded_and_matches_the_reference(tmp_path):
    mod = _script("make_random_store")
    ref = tmp_path / "ref"
    AlignedMatrix(vectors=np.ones((4, 8), dtype=np.float32), node_ids=["a", "b", "c", "d"]).save(ref)
    one = mod.make_random_store(ref, tmp_path / "out1", seed=7, label="r7", dataset="ds")
    two = mod.make_random_store(ref, tmp_path / "out2", seed=7, label="r7", dataset="ds")
    other = mod.make_random_store(ref, tmp_path / "out3", seed=8, label="r8", dataset="ds")
    a, b, c = (AlignedMatrix.load(d) for d in (one, two, other))
    assert a.node_ids == ["a", "b", "c", "d"] and a.vectors.shape == (4, 8)
    assert np.array_equal(a.vectors, b.vectors) and not np.array_equal(a.vectors, c.vectors)
    assert one.name == two.name != other.name
