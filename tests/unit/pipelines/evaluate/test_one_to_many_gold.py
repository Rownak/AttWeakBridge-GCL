"""Hand-computed 1:n gold fixture for Sprint 10.4 (technique -> several CWEs).

Vectors are 2-D unit vectors placed by angle (degrees). Sources: s1 0, s2 90. Targets: a 10, d 15, b 20, c 80.
Gold (test): (s1,a) (s1,b) (s2,c); the pair (s1,d) is *train* (a sibling target of s1 outside the scored split).

Cosine order: s1 -> a, d, b, c;  s2 -> c, b, d, a.

``gold_alignment`` (global, ``sibling_filter_splits: [test]``): each query drops the source's *other test* targets only.
  (s1,a): b dropped      -> a is rank 1
  (s1,b): a dropped      -> d (train sibling, kept) is rank 1, b is rank 2
  (s2,c): nothing to drop -> c is rank 1
  Hits@1 = 2/3, Hits@2 = 1, MRR = (1 + 1/2 + 1) / 3 = 5/6.
With the default filter (every split) d is dropped too, so (s1,b) ranks 1 and MRR = 1.

``gold_alignment_prf`` (top-3, greedy one-to-one, threshold 0): rows s1-a .985, s1-d .966, s1-b .940, s2-c .985,
s2-b .342, s2-d .259. One-to-one keeps one prediction per source: s1-a and s2-c, both correct.
  n_predicted 2, n_correct 2, n_gold 3 -> P 1, R 2/3, F1 4/5. The recall ceiling is 2 sources / 3 pairs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from cga.data.aligned import AlignedMatrix
from cga.data.schemas import ALIGNMENTS_SCHEMA
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.methods import MethodSpec
from cga.pipelines.evaluate.rows import MetricsDigest

ANGLES = {"s:1": 0, "s:2": 90, "t:a": 10, "t:d": 15, "t:b": 20, "t:c": 80}
ROLES = {n: ("graph_1" if n.startswith("s:") else "graph_2") for n in ANGLES}
PAIRS = [("s:1", "t:a", "test"), ("s:1", "t:b", "test"), ("s:2", "t:c", "test"), ("s:1", "t:d", "train")]


def _res() -> SimpleNamespace:
    ids = list(ANGLES)
    rad = np.deg2rad([ANGLES[i] for i in ids])
    matrix = AlignedMatrix(vectors=np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32), node_ids=ids)
    table = pa.Table.from_pylist(
        [
            {"alignment_id": f"{a}|{b}", "node_1_id": a, "node_2_id": b, "relation": "equivalence",
             "score": 1.0, "split": s, "candidates": None}
            for a, b, s in PAIRS
        ],
        schema=ALIGNMENTS_SCHEMA,
    )
    spec = MethodSpec(method_id="m", label="m", kind="embedding_store", ref="latest", embedder_id="e")
    return SimpleNamespace(
        methods={"m": (matrix, "run-m")}, method_specs={"m": spec}, graph_role_by_node=ROLES,
        alignments=table, digest=MetricsDigest(),
    )


def _rank(**params) -> dict:
    rows = evaluators.get("gold_alignment")().evaluate(_res(), "gold", dict(ks=[1, 2], modes=["global"], **params))
    return {(r.metric_name, r.param_value): r for r in rows}


def test_ranking_with_test_only_sibling_filter():
    out = _rank(sibling_filter_splits=["test"])
    assert out[("mrr", None)].n_queries == 3
    assert out[("mrr", None)].metric_value == pytest.approx(5 / 6)
    assert out[("hits_at_k", 1)].metric_value == pytest.approx(2 / 3)
    assert out[("hits_at_k", 2)].metric_value == pytest.approx(1.0)


def test_default_filter_also_drops_the_train_sibling():
    out = _rank()
    assert out[("mrr", None)].metric_value == pytest.approx(1.0)
    assert out[("hits_at_k", 1)].metric_value == pytest.approx(1.0)


def test_prf_under_one_to_one_assignment():
    rows = evaluators.get("gold_alignment_prf")().evaluate(
        _res(), "prf", dict(split="test", assignment="greedy_one_to_one", top_k=3, threshold=0.0, biogitom_filters=False)
    )
    out = {r.metric_name: r.metric_value for r in rows}
    assert (out["n_predicted"], out["n_correct"], out["n_gold"]) == (2, 2, 3)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(4 / 5)
    assert out["recall"] <= out["n_unique_sources"] / out["n_gold"] + 1e-9  # ceiling: sources / pairs
