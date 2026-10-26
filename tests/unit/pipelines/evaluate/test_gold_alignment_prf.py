"""Tests for the ``gold_alignment_prf`` evaluator on a hand-built 4 source x 4 target fixture.

Vectors are 2-D unit vectors placed by angle (degrees): sources s1 0, s2 5, s3 90, s4 180 and
targets a 10, b 20, c 85, d 175. Cosine top-3 per source (score):

    s1: a .985, b .940, c .087      s2: a .996, b .966, c .174
    s3: c .996, b .342, a .174      s4: d .996, c -.087, b -.940

Gold: (s1,a) (s2,b) (s3,c) (s4,d). The three .996 rows are independent, so the greedy
assignment is s2-a, s3-c, s4-d, then s1-b (a is taken): 2 of 4 correct.
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
from cga.pipelines.evaluate.evaluators.gold_alignment_prf import apply_filters, assign_pairs, score_pairs
from cga.pipelines.evaluate.methods import MethodSpec, fuse_matrices
from cga.pipelines.evaluate.rows import MetricsDigest

ANGLES = {"s:1": 0, "s:2": 5, "s:3": 90, "s:4": 180, "t:a": 10, "t:b": 20, "t:c": 85, "t:d": 175}
ROLES = {n: ("graph_1" if n.startswith("s:") else "graph_2") for n in ANGLES}
GOLD = [("s:1", "t:a"), ("s:2", "t:b"), ("s:3", "t:c"), ("s:4", "t:d")]


def _matrix(angles: dict[str, float]) -> AlignedMatrix:
    ids = list(angles)
    rad = np.deg2rad([angles[i] for i in ids])
    return AlignedMatrix(vectors=np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32), node_ids=ids)


def _alignments(pairs: list[tuple[str, str]], split: str = "test", train: list[tuple[str, str]] = ()) -> pa.Table:
    rows = [(a, b, split) for a, b in pairs] + [(a, b, "train") for a, b in train]
    return pa.Table.from_pylist(
        [
            {
                "alignment_id": f"{a}|{b}",
                "node_1_id": a,
                "node_2_id": b,
                "relation": "equivalence",
                "score": 1.0,
                "split": s,
                "candidates": None,
            }
            for a, b, s in rows
        ],
        schema=ALIGNMENTS_SCHEMA,
    )


def _res(alignments, methods=None, roles=ROLES) -> SimpleNamespace:
    methods = methods or {"m": _matrix(ANGLES)}
    return SimpleNamespace(
        methods={mid: (m, f"run-{mid}") for mid, m in methods.items()},
        method_specs={
            mid: MethodSpec(method_id=mid, label=mid, kind="embedding_store", ref="latest", embedder_id="e")
            for mid in methods
        },
        graph_role_by_node=roles,
        alignments=alignments,
        digest=MetricsDigest(),
    )


def _run(res, method="m", **params) -> dict[str, float]:
    rows = evaluators.get("gold_alignment_prf")().evaluate(res, "prf", params)
    return {r.metric_name: r.metric_value for r in rows if r.method_id == method}


def test_greedy_one_to_one_matches_hand_computed_values():
    out = _run(_res(_alignments(GOLD)))
    assert (out["n_predicted"], out["n_correct"], out["n_gold"]) == (4, 2, 4)
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


def test_top1_has_no_assignment_so_a_shared_target_is_reused():
    out = _run(_res(_alignments(GOLD)), assignment="top1")
    # s1->a (correct), s2->a (wrong), s3->c, s4->d.
    assert (out["n_predicted"], out["n_correct"]) == (4, 3)
    assert out["f1"] == pytest.approx(0.75)


def test_threshold_drops_low_scoring_rows():
    out = _run(_res(_alignments(GOLD)), threshold=0.97)
    # Only the three .996 rows survive: s2-a, s3-c, s4-d.
    assert (out["n_predicted"], out["n_correct"]) == (3, 2)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert _run(_res(_alignments(GOLD)), threshold=2.0)["n_predicted"] == 0


def test_negative_cosines_are_dropped_at_threshold_zero():
    # top_k=3 gives s4 two negative rows (c, b); with top_k=1 nothing is negative.
    out = _run(_res(_alignments(GOLD)), assignment="top1", top_k=3)
    assert out["n_predicted"] == 4
    far = dict(ANGLES, **{"t:d": 355})  # s4 (180) now sees only negative cosines; s3 (90) too
    out = _run(_res(_alignments(GOLD), {"m": _matrix(far)}), assignment="top1", top_k=1)
    assert out["n_predicted"] < 4


def test_filters_remove_train_only_endpoints():
    # t:e is a train-only target sitting next to s1 and s2; s:0 is a train-only source.
    angles = dict(ANGLES, **{"t:e": 2})
    roles = dict(ROLES, **{"t:e": "graph_2"})
    res = lambda: _res(_alignments(GOLD, train=[("s:0", "t:e")]), {"m": _matrix(angles)}, roles)  # noqa: E731
    unfiltered = _run(res(), assignment="top1")
    assert (unfiltered["n_predicted"], unfiltered["n_correct"]) == (4, 2)  # s1 and s2 both pick e
    filtered = _run(res(), assignment="top1", biogitom_filters=True)
    assert (filtered["n_predicted"], filtered["n_correct"]) == (4, 3)


def test_apply_filters_drops_train_only_sources_and_foreign_sources():
    rows = [(0.9, "s:1", "t:a"), (0.8, "s:0", "t:a"), (0.7, "s:1", "t:e"), (0.6, "s:9", "t:a")]
    kept = apply_filters(rows, excluded_nodes={"s:0", "t:e"}, split_sources={"s:1", "s:0"})
    assert kept == [(0.9, "s:1", "t:a")]


def test_shared_target_caps_greedy_recall_but_not_top1():
    pairs = [("s:1", "t:a"), ("s:2", "t:a")]
    greedy = _run(_res(_alignments(pairs)))
    top1 = _run(_res(_alignments(pairs)), assignment="top1")
    assert greedy["n_unique_targets"] == 1 and greedy["n_unique_sources"] == 2
    assert greedy["recall"] == pytest.approx(0.5)  # a is used once
    assert top1["recall"] == pytest.approx(1.0)


def test_source_with_two_gold_targets_is_capped_under_both_assignments():
    pairs = [("s:1", "t:a"), ("s:1", "t:b"), ("s:3", "t:c")]
    for assignment in ("greedy_one_to_one", "top1"):
        out = _run(_res(_alignments(pairs)), assignment=assignment)
        assert out["n_gold"] == 3 and out["n_unique_sources"] == 2
        assert out["recall"] == pytest.approx(2 / 3)
        assert out["precision"] == pytest.approx(1.0)


def test_gold_pair_with_a_missing_endpoint_counts_as_unpredicted():
    roles = dict(ROLES, **{"s:9": "graph_1"})
    out = _run(_res(_alignments(GOLD + [("s:9", "t:a")]), roles=roles), assignment="top1")
    assert out["n_gold"] == 5
    assert out["recall"] == pytest.approx(3 / 5)
    assert out["precision"] == pytest.approx(3 / 4)


def test_ties_break_by_node_id():
    tied = dict(ANGLES, **{"t:b": 10})  # t:b identical to t:a
    out = _run(_res(_alignments([("s:1", "t:a")]), {"m": _matrix(tied)}), assignment="top1")
    assert out["precision"] == 1.0  # t:a (lower id) wins the tie
    out = _run(_res(_alignments([("s:1", "t:b")]), {"m": _matrix(tied)}), assignment="top1")
    assert out["precision"] == 0.0


def test_greedy_is_independent_of_row_order():
    rows = [(0.9, "s:1", "t:a"), (0.9, "s:2", "t:a"), (0.5, "s:2", "t:b")]
    assert assign_pairs(rows, "greedy_one_to_one", 0.0) == assign_pairs(rows[::-1], "greedy_one_to_one", 0.0)
    assert assign_pairs(rows, "greedy_one_to_one", 0.0) == [("s:1", "t:a"), ("s:2", "t:b")]


def test_score_pairs_handles_empty_predictions():
    out = score_pairs([], {("s:1", "t:a")})
    assert (out["precision"], out["recall"], out["f1"]) == (0.0, 0.0, 0.0)


def test_fusion_matches_its_explicit_weighted_mean_scores():
    m1 = _matrix(ANGLES)
    m2 = _matrix(dict(ANGLES, **{"t:a": 60, "t:b": 5}))
    fused = fuse_matrices([(m1, 2.0), (m2, 1.0)])
    got = _run(_res(_alignments(GOLD), {"f": fused}, ), method="f", assignment="top1")

    def sim(m: AlignedMatrix) -> np.ndarray:
        s = m.select([n for n in ANGLES if n.startswith("s:")]).vectors
        t = m.select(sorted(n for n in ANGLES if n.startswith("t:"))).vectors
        return s @ t.T

    mean = (2.0 * sim(m1) + sim(m2)) / 3.0
    targets = sorted(n for n in ANGLES if n.startswith("t:"))
    sources = [n for n in ANGLES if n.startswith("s:")]
    rows = [
        (float(mean[i, j]), s, targets[j])
        for i, s in enumerate(sources)
        for j in np.argsort(-mean[i], kind="stable")[:3]
    ]
    expected = score_pairs(assign_pairs(rows, "top1", 0.0), set(GOLD))
    for key in ("precision", "recall", "f1", "n_predicted", "n_correct"):
        assert got[key] == pytest.approx(expected[key])


def test_val_split_scores_val_pairs_only():
    table = _alignments([("s:1", "t:a")], split="val", train=[("s:3", "t:c")])
    out = _run(_res(table), split="val", assignment="top1")
    assert out["n_gold"] == 1 and out["f1"] == pytest.approx(1.0)


def test_invalid_params_and_missing_split_raise_config_error():
    res = _res(_alignments(GOLD))
    with pytest.raises(ConfigError):
        _run(res, assignment="nope")
    with pytest.raises(ConfigError):
        _run(res, top_k=0)
    with pytest.raises(ConfigError):
        _run(res, split="val")
    with pytest.raises(ConfigError):
        _run(_res(None))


def test_provenance_records_the_options():
    res = _res(_alignments(GOLD))
    _run(res, biogitom_filters=True, threshold=0.1)
    prov = res.digest._provenance[("prf", "greedy_one_to_one", "graph_1")]
    assert prov["biogitom_filters"] is True and prov["threshold"] == 0.1 and len(prov["gold_set_sha256"]) == 64
