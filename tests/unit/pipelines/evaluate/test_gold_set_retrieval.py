"""Tests for the ``gold_set_retrieval`` evaluator and the ``precision_at_k`` metric helpers.

Hand-built fixture, 2-D unit vectors placed by angle (degrees): sources s1 0, s2 90, s3 180 and
targets g 5, a 10, b 20, c 30, d 95, e 100, f 185. Gold: s1 -> {a, c}, s2 -> {d}, s3 -> {f, x}
where ``x`` has no embedding (a relevant target outside the candidate pool). Rankings by angle:

    s1: g a b c ...      s2: d e c ...      s3: f e d ...
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cga.core.errors import ConfigError
from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.metrics import precision_at_k, precision_ceiling_at_k
from cga.pipelines.evaluate.methods import MethodSpec
from cga.pipelines.evaluate.rows import MetricsDigest
from .test_gold_alignment_prf import _alignments

ANGLES = {"s:1": 0, "s:2": 90, "s:3": 180, "t:g": 5, "t:a": 10, "t:b": 20, "t:c": 30, "t:d": 95, "t:e": 100, "t:f": 185}
ROLES = {n: ("graph_1" if n.startswith("s:") else "graph_2") for n in [*ANGLES, "t:x"]}
GOLD = [("s:1", "t:a"), ("s:1", "t:c"), ("s:2", "t:d"), ("s:3", "t:f"), ("s:3", "t:x")]


def _matrix(angles: dict[str, float]) -> AlignedMatrix:
    ids = list(angles)
    rad = np.deg2rad([angles[i] for i in ids])
    return AlignedMatrix(vectors=np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32), node_ids=ids)


def _res(gold, matrix=None, roles=ROLES) -> SimpleNamespace:
    matrix = matrix or _matrix(ANGLES)
    return SimpleNamespace(
        methods={"m": (matrix, "run-m")},
        method_specs={"m": MethodSpec(method_id="m", label="m", kind="embedding_store", ref="latest", embedder_id="e")},
        graph_role_by_node=roles,
        alignments=_alignments(gold),
        digest=MetricsDigest(),
    )


def _run(res, **params):
    rows = evaluators.get("gold_set_retrieval")().evaluate(res, "set", params)
    return {(r.metric_name, r.param_value): r.metric_value for r in rows}, rows


def test_metrics_match_hand_computed_values():
    out, rows = _run(_res(GOLD), ks=[1, 2])
    assert {r.n_queries for r in rows} == {3}
    assert out[("hits_at_k", 1)] == pytest.approx(2 / 3)
    assert out[("recall_at_k", 1)] == pytest.approx(2 / 3)
    assert out[("precision_at_k", 1)] == pytest.approx(2 / 3)
    assert out[("hits_at_k", 2)] == pytest.approx(1.0)
    assert out[("recall_at_k", 2)] == pytest.approx((0.5 + 1 + 1) / 3)  # s1 finds a of {a, c}
    assert out[("precision_at_k", 2)] == pytest.approx(0.5)


def test_precision_ceiling_uses_the_relevant_set_inside_the_pool():
    out, _ = _run(_res(GOLD), ks=[1, 2, 10])
    # |R| = 2, 1, 1 (t:x has no embedding and is dropped from s3's relevant set)
    assert out[("precision_ceiling_at_k", 1)] == pytest.approx(1.0)
    assert out[("precision_ceiling_at_k", 2)] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert out[("precision_ceiling_at_k", 10)] == pytest.approx((0.2 + 0.1 + 0.1) / 3)


def test_k_larger_than_the_relevant_set_and_the_pool():
    out, _ = _run(_res(GOLD), ks=[50])
    assert out[("recall_at_k", 50)] == pytest.approx(1.0)  # every pool node is in the top 50
    assert out[("hits_at_k", 50)] == pytest.approx(1.0)
    assert out[("precision_at_k", 50)] == pytest.approx((2 + 1 + 1) / 3 / 50)


def test_ties_break_by_node_id():
    angles = {"s:1": 0, "t:a": 10, "t:b": 10}
    roles = {"s:1": "graph_1", "t:a": "graph_2", "t:b": "graph_2"}
    first, _ = _run(_res([("s:1", "t:a")], matrix=_matrix(angles), roles=roles), ks=[1])
    second, _ = _run(_res([("s:1", "t:b")], matrix=_matrix(angles), roles=roles), ks=[1])
    assert first[("hits_at_k", 1)] == 1.0
    assert second[("hits_at_k", 1)] == 0.0


def test_target_pool_restricts_candidates_and_rejects_outside_targets(tmp_path):
    from cga.pipelines.evaluate.evaluators.target_pool import pool_sha256

    def pool(ids):
        path = tmp_path / "pool.json"
        path.write_text(json.dumps({"node_ids": ids, "source_run": "r", "sha256": pool_sha256(ids)}))
        return str(path)

    gold = [("s:1", "t:a"), ("s:2", "t:d")]
    out, _ = _run(_res(gold), ks=[1], target_pool=pool(["t:a", "t:d", "t:e"]))
    assert out[("hits_at_k", 1)] == pytest.approx(1.0)  # t:g is no longer a candidate for s:1
    with pytest.raises(ConfigError):
        _run(_res(gold), ks=[1], target_pool=pool(["t:a", "t:e"]))  # t:d is a gold target outside the pool


def test_invalid_ks_raise():
    for bad in ([], [0], [1.5]):
        with pytest.raises(ConfigError):
            _run(_res(GOLD), ks=bad)


def test_random_vectors_recall_is_close_to_k_over_pool_size():
    rng = np.random.default_rng(0)
    n_src, n_pool, k = 400, 200, 10
    ids = [f"s:{i}" for i in range(n_src)] + [f"t:{i:03d}" for i in range(n_pool)]
    matrix = AlignedMatrix(vectors=rng.standard_normal((len(ids), 16)).astype(np.float32), node_ids=ids)
    roles = {n: ("graph_1" if n.startswith("s:") else "graph_2") for n in ids}
    gold = [(f"s:{i}", f"t:{rng.integers(n_pool):03d}") for i in range(n_src)]
    out, _ = _run(_res(gold, matrix=matrix, roles=roles), ks=[k])
    assert out[("recall_at_k", k)] == pytest.approx(k / n_pool, abs=0.03)


def test_provenance_records_relevant_set_sizes():
    res = _res(GOLD)
    _run(res, ks=[1])
    prov = res.digest._provenance[("set", "global", "graph_1")]
    assert prov["n_eligible_queries"] == 3
    assert (prov["n_relevant_mean"], prov["n_relevant_median"], prov["n_relevant_max"]) == (pytest.approx(4 / 3), 1.0, 2)
    assert prov["n_candidates"] == 7


def test_precision_helpers():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], set(), 1) == 0.0
    assert precision_ceiling_at_k(1, 15) == pytest.approx(1 / 15)
    assert precision_ceiling_at_k(30, 15) == 1.0
