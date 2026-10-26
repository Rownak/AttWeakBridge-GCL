"""Tests for the ``gold_alignment`` evaluator on a hand-built 2 source + 4 target fixture.

Vectors are 2-D unit vectors placed by angle, so cosine rank order is known:
``s1`` (0 deg) sees ``t_a`` (10) > ``t_b`` (20) > ``t_c`` (30) > ``t_d`` (80);
``s2`` (90 deg) sees ``t_d`` (80) first.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from cga.core.errors import ConfigError
from cga.data.aligned import AlignedMatrix
from cga.data.schemas import ALIGNMENTS_SCHEMA
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.methods import MethodSpec
from cga.pipelines.evaluate.rows import MetricsDigest

ANGLES = {"s:1": 0, "s:2": 90, "t:a": 10, "t:b": 20, "t:c": 30, "t:d": 80}
ROLES = {"s:1": "graph_1", "s:2": "graph_1", "t:a": "graph_2", "t:b": "graph_2", "t:c": "graph_2", "t:d": "graph_2"}


def _matrix(angles: dict[str, float]) -> AlignedMatrix:
    ids = list(angles)
    rad = np.deg2rad([angles[i] for i in ids])
    return AlignedMatrix(vectors=np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32), node_ids=ids)


def _alignments(rows: list[tuple[str, str, str, list[str] | None]]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "alignment_id": f"{n1}|{n2}",
                "node_1_id": n1,
                "node_2_id": n2,
                "relation": "equivalence",
                "score": 1.0,
                "split": split,
                "candidates": cands,
            }
            for n1, n2, split, cands in rows
        ],
        schema=ALIGNMENTS_SCHEMA,
    )


# (s1, t_c) test, (s2, t_d) test, plus a train sibling target (s1, t_a).
BASE_ALIGNMENTS = [
    ("s:1", "t:c", "test", ["t:a", "t:b", "t:c"]),
    ("s:2", "t:d", "test", None),
    ("s:1", "t:a", "train", None),
]


def _res(alignments: pa.Table | None, methods: dict[str, AlignedMatrix] | None = None) -> SimpleNamespace:
    methods = methods or {"m": _matrix(ANGLES)}
    return SimpleNamespace(
        methods={mid: (m, f"run-{mid}") for mid, m in methods.items()},
        method_specs={mid: MethodSpec(method_id=mid, label=mid, kind="embedding_store", ref="latest", embedder_id="e") for mid in methods},
        graph_role_by_node=ROLES,
        alignments=alignments,
        digest=MetricsDigest(),
    )


def _run(res, **params):
    rows = evaluators.get("gold_alignment")().evaluate(res, "gold", params)
    return {(r.method_id, r.view_id, r.metric_name, r.param_value): r for r in rows}


def test_global_mode_hits_and_mrr_match_hand_computed_values():
    # Global drops s1's sibling target t_a, so s1->t_c ranks 2nd (t_b first); s2->t_d ranks 1st.
    out = _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1, 2, 5], modes=["global"])
    assert out[("m", "global", "mrr", None)].metric_value == pytest.approx((1 / 2 + 1) / 2)
    assert out[("m", "global", "hits_at_k", 1)].metric_value == pytest.approx(0.5)
    assert out[("m", "global", "hits_at_k", 2)].metric_value == pytest.approx(1.0)
    assert out[("m", "global", "hits_at_k", 5)].metric_value == pytest.approx(1.0)
    assert out[("m", "global", "mrr", None)].n_queries == 2
    assert out[("m", "global", "mrr", None)].graph_role == "graph_1"


def test_target_at_rank_three_gives_mrr_one_third():
    # candidates mode keeps the sibling: pool [t_a, t_b, t_c] ranks t_c 3rd. s2 has no pool (skipped).
    out = _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1, 5], modes=["candidates"])
    assert out[("m", "candidates", "mrr", None)].metric_value == pytest.approx(1 / 3)
    assert out[("m", "candidates", "hits_at_k", 1)].metric_value == 0.0
    assert out[("m", "candidates", "hits_at_k", 5)].metric_value == 1.0
    assert out[("m", "candidates", "mrr", None)].n_queries == 1


def test_global_filter_drops_sibling_but_candidates_mode_does_not():
    out = _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1], modes=["global", "candidates"])
    s1_global_rank_mrr = 1 / 2  # t_a (sibling) removed
    s1_candidates_mrr = 1 / 3  # t_a kept in the pool
    assert out[("m", "global", "mrr", None)].metric_value == pytest.approx((s1_global_rank_mrr + 1) / 2)
    assert out[("m", "candidates", "mrr", None)].metric_value == pytest.approx(s1_candidates_mrr)


def test_candidates_mode_ranks_only_within_pool_and_always_includes_target():
    # Pool omits the target t_c; it is added back, and t_d (closer than none) is not in the pool.
    rows = [("s:1", "t:c", "test", ["t:b"])]
    out = _run(_res(_alignments(rows)), ks=[1, 2], modes=["candidates"])
    assert out[("m", "candidates", "mrr", None)].metric_value == pytest.approx(1 / 2)  # t_b, then t_c
    assert out[("m", "candidates", "hits_at_k", 2)].metric_value == 1.0


def test_ties_break_by_node_id():
    tied = dict(ANGLES, **{"t:b": 10})  # t_b identical to t_a
    # Pair (s1, t_b): t_a and t_b tie at the top; id order puts t_a first -> t_b ranks 2nd.
    out = _run(_res(_alignments([("s:1", "t:b", "test", None)]), {"m": _matrix(tied)}), ks=[1, 2], modes=["global"])
    assert out[("m", "global", "mrr", None)].metric_value == pytest.approx(1 / 2)
    assert out[("m", "global", "hits_at_k", 1)].metric_value == 0.0
    # Deterministic across calls.
    again = _run(_res(_alignments([("s:1", "t:b", "test", None)]), {"m": _matrix(tied)}), ks=[1, 2], modes=["global"])
    assert {k: v.metric_value for k, v in out.items()} == {k: v.metric_value for k, v in again.items()}


def test_only_the_selected_split_is_queried():
    res = _res(_alignments(BASE_ALIGNMENTS))
    test_out = _run(res, ks=[1], modes=["global"])
    assert test_out[("m", "global", "mrr", None)].n_queries == 2  # train pair (s1, t_a) not queried
    train_out = _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1], split="train", modes=["global"])
    assert train_out[("m", "global", "mrr", None)].n_queries == 1


def test_pair_missing_from_one_method_is_dropped_for_every_method():
    full = _matrix(ANGLES)
    without_s2 = _matrix({k: v for k, v in ANGLES.items() if k != "s:2"})
    res = _res(_alignments(BASE_ALIGNMENTS), {"a": full, "b": without_s2})
    out = _run(res, ks=[1], modes=["global"])
    assert out[("a", "global", "mrr", None)].n_queries == 1
    assert out[("b", "global", "mrr", None)].n_queries == 1
    # Both methods score the surviving query (s1, t_c) identically.
    assert out[("a", "global", "mrr", None)].metric_value == pytest.approx(1 / 2)
    assert out[("b", "global", "mrr", None)].metric_value == pytest.approx(1 / 2)


def test_scope_provenance_records_query_set_hash():
    res = _res(_alignments(BASE_ALIGNMENTS))
    _run(res, ks=[1], modes=["global"])
    scopes = res.digest.to_dict()  # rows are attached by the pipeline; provenance alone must not crash
    assert scopes["evaluations"] == {}
    assert ("gold", "global", "graph_1") in res.digest._provenance
    prov = res.digest._provenance[("gold", "global", "graph_1")]
    assert len(prov["query_set_sha256"]) == 64
    assert prov["n_eligible_queries"] == 2


def test_no_alignments_raises_config_error():
    with pytest.raises(ConfigError):
        _run(_res(None), ks=[1])
    with pytest.raises(ConfigError):
        _run(_res(_alignments([])), ks=[1])
    with pytest.raises(ConfigError):  # table has pairs but none in the selected split
        _run(_res(_alignments([("s:1", "t:a", "train", None)])), ks=[1], split="test")


def test_invalid_params_raise_config_error():
    res = _res(_alignments(BASE_ALIGNMENTS))
    with pytest.raises(ConfigError):
        _run(res, ks=[])
    with pytest.raises(ConfigError):
        _run(res, ks=[1], modes=["nope"])


def test_source_with_several_test_targets_yields_one_query_per_pair():
    # (s1, t_c) and (s1, t_d) are both test pairs. Global mode removes the other target for each,
    # so both rank 3rd behind t_a and t_b (s1 order: t_a, t_b, t_c, t_d).
    rows = [("s:1", "t:c", "test", None), ("s:1", "t:d", "test", None)]
    out = _run(_res(_alignments(rows)), ks=[1], modes=["global"])
    assert out[("m", "global", "mrr", None)].n_queries == 2
    assert out[("m", "global", "mrr", None)].metric_value == pytest.approx(1 / 3)


def test_val_split_queries_only_val_pairs():
    rows = BASE_ALIGNMENTS + [("s:2", "t:c", "val", None)]
    out = _run(_res(_alignments(rows)), ks=[1], split="val", modes=["global"])
    assert out[("m", "global", "mrr", None)].n_queries == 1
    # s2's test target t_d is another gold target of s2, so global mode drops it and t_c ranks first.
    assert out[("m", "global", "mrr", None)].metric_value == pytest.approx(1.0)


def test_sibling_filter_splits_restricts_which_splits_feed_the_filter():
    # BASE_ALIGNMENTS: the sibling t_a of (s1, t_c) is a *train* pair. By default it is filtered out of the
    # global pool (s1 -> t_c ranks 2nd); restricted to [test] the train pair is ignored and t_a stays (3rd).
    default = _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1], modes=["global"])
    test_only = _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1], modes=["global"], sibling_filter_splits=["test"])
    both = _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1], modes=["global"], sibling_filter_splits=["train", "test"])
    assert default[("m", "global", "mrr", None)].metric_value == pytest.approx((1 / 2 + 1) / 2)
    assert test_only[("m", "global", "mrr", None)].metric_value == pytest.approx((1 / 3 + 1) / 2)
    assert both[("m", "global", "mrr", None)].metric_value == default[("m", "global", "mrr", None)].metric_value


def test_sibling_filter_splits_still_filters_siblings_inside_the_scored_split():
    rows = [("s:1", "t:c", "test", None), ("s:1", "t:a", "test", None)]
    out = _run(_res(_alignments(rows)), ks=[1], modes=["global"], sibling_filter_splits=["test"])
    # (s1, t_c) drops sibling t_a -> rank 2; (s1, t_a) drops sibling t_c -> rank 1.
    assert out[("m", "global", "mrr", None)].metric_value == pytest.approx((1 / 2 + 1) / 2)


def test_sibling_filter_splits_must_be_a_list_of_names():
    with pytest.raises(ConfigError, match="sibling_filter_splits"):
        _run(_res(_alignments(BASE_ALIGNMENTS)), ks=[1], modes=["global"], sibling_filter_splits="test")
