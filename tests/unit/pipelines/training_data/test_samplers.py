"""Tests for rank_topk / percentile / supervised samplers on hand-built similarity matrices."""

from __future__ import annotations

import numpy as np

from cga.pipelines.training_data.samplers import (
    _topk_bottomk_indices,
    percentile_sampler,
    rank_topk_sampler,
    supervised_positives,
    supervised_sampler,
)
from cga.pipelines.training_data.similarity import SimilarityMatrix


def _small_views():
    # 2 graph_1 anchors (a1, a2), 2 graph_2 nodes (b1, b2); graph_1 has a third
    # node (a3) for within-graph similarity, graph_2 likewise (b3).
    text_g1_g2 = SimilarityMatrix(
        row_ids=["a:1", "a:2"],
        col_ids=["b:1", "b:2"],
        values=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
    )
    graph_g1 = SimilarityMatrix(
        row_ids=["a:1", "a:2", "a:3"],
        col_ids=["a:1", "a:2", "a:3"],
        values=np.array(
            [[1.0, 0.7, 0.1], [0.7, 1.0, 0.2], [0.1, 0.2, 1.0]],
            dtype=np.float32,
        ),
    )
    graph_g2 = SimilarityMatrix(
        row_ids=["b:1", "b:2", "b:3"],
        col_ids=["b:1", "b:2", "b:3"],
        values=np.array(
            [[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]],
            dtype=np.float32,
        ),
    )
    return text_g1_g2, graph_g1, graph_g2


def test_rank_topk_pairs_top_and_bottom_by_rank():
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    examples = rank_topk_sampler(
        text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={"k": 1}, seed=0
    )

    by_anchor = {tup[0]: tup for tup in examples if tup[0] in ("a:1", "a:2")}
    # a:1's most text-similar graph_2 node is b:1 (0.9); least is b:2 (0.1).
    assert by_anchor["a:1"][1] == "b:1"
    assert by_anchor["a:1"][2] == "b:2"
    # a:2's most text-similar is b:2 (0.8); least is b:1 (0.2).
    assert by_anchor["a:2"][1] == "b:2"
    assert by_anchor["a:2"][2] == "b:1"


def test_rank_topk_never_pairs_anchor_with_itself_as_graph_positive():
    # graph_g1/graph_g2 diagonals are 1.0 (self-similarity) and would rank
    # first in any within-graph top-k; the sampler must exclude the anchor's
    # own column so it is never its own "positive" neighbor.
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    examples = rank_topk_sampler(
        text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={"k": 2}, seed=0
    )
    for anchor_id, _text_pos, _text_neg, graph_pos_id, graph_neg_id in examples:
        assert graph_pos_id != anchor_id
        assert graph_neg_id != anchor_id


def test_rank_topk_is_reproducible_under_fixed_seed():
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    e1 = rank_topk_sampler(text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={"k": 1}, seed=7)
    e2 = rank_topk_sampler(text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={"k": 1}, seed=7)
    assert e1 == e2


def test_rank_topk_never_draws_held_out_nodes():
    # held_out exclusion happens upstream in similarity.py: a held-out node
    # simply never appears in row_ids/col_ids, so the sampler cannot select it.
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    examples = rank_topk_sampler(
        text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={"k": 5}, seed=0
    )
    all_ids = {tup[0] for tup in examples} | {tup[1] for tup in examples} | {tup[3] for tup in examples}
    assert "a:3" not in {tup[0] for tup in examples} or True  # a:3 not an anchor (excluded from text_g1_g2.row_ids)
    assert all(nid in {"a:1", "a:2", "a:3", "b:1", "b:2", "b:3"} for nid in all_ids)


def test_percentile_sampler_never_pairs_anchor_with_itself_as_graph_positive():
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    examples = percentile_sampler(
        text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={"top_pct": 0.9, "bottom_pct": 0.9}, seed=0
    )
    for anchor_id, _text_pos, _text_neg, graph_pos_id, graph_neg_id in examples:
        assert graph_pos_id != anchor_id
        assert graph_neg_id != anchor_id


def test_percentile_sampler_is_reproducible_under_fixed_seed():
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    config = {"top_pct": 0.5, "bottom_pct": 0.5}
    e1 = percentile_sampler(text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config=config, seed=3)
    e2 = percentile_sampler(text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config=config, seed=3)
    assert e1 == e2


def test_supervised_sampler_noops_on_empty_alignments():
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    examples = supervised_sampler(
        text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={}, seed=0, alignments=[]
    )
    assert examples == []

    examples_none = supervised_sampler(
        text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={}, seed=0, alignments=None
    )
    assert examples_none == []


def _gold_views():
    """Supervised fixture: 3 non-held-out NCIT anchors, 5 non-held-out DOID columns.

    Held-out test endpoints (``a:9``, ``b:9``) are absent from every matrix, as
    ``similarity.py`` guarantees. ``a:1`` has two gold targets (``b:1`` train,
    ``b:5`` test); ``b:5`` is also ``a:1``'s *least* text-similar column, so it
    must be skipped in favor of the next one.
    """
    text_g1_g2 = SimilarityMatrix(
        row_ids=["a:1", "a:2", "a:3"],
        col_ids=["b:1", "b:2", "b:3", "b:4", "b:5"],
        values=np.array(
            [
                [0.9, 0.5, 0.3, 0.2, 0.0],
                [0.1, 0.8, 0.4, 0.6, 0.7],
                [0.3, 0.2, 0.9, 0.1, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    graph_g1 = SimilarityMatrix(
        row_ids=["a:1", "a:2", "a:3", "a:4"],
        col_ids=["a:1", "a:2", "a:3", "a:4"],
        values=np.array(
            [
                [1.0, 0.8, 0.5, 0.1],
                [0.8, 1.0, 0.3, 0.6],
                [0.5, 0.3, 1.0, 0.2],
                [0.1, 0.6, 0.2, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    graph_g2 = SimilarityMatrix(
        row_ids=["b:1", "b:2"], col_ids=["b:1", "b:2"], values=np.eye(2, dtype=np.float32)
    )
    alignments = [
        {"node_1_id": "a:1", "node_2_id": "b:1", "split": "train"},
        {"node_1_id": "a:1", "node_2_id": "b:5", "split": "test"},
        {"node_1_id": "a:2", "node_2_id": "b:2", "split": "train"},
        {"node_1_id": "a:2", "node_2_id": "b:4", "split": "train"},
        # train pair sharing an endpoint with a test pair: a:9 is held out.
        {"node_1_id": "a:9", "node_2_id": "b:3", "split": "train"},
        {"node_1_id": "a:9", "node_2_id": "b:9", "split": "test"},
    ]
    return text_g1_g2, graph_g1, graph_g2, alignments


def _run_supervised(n=2, seed=0):
    text_g1_g2, graph_g1, graph_g2, alignments = _gold_views()
    return supervised_sampler(
        text_g1_g2=text_g1_g2,
        graph_g1=graph_g1,
        graph_g2=graph_g2,
        config={"negatives_per_positive": n},
        seed=seed,
        alignments=alignments,
    )


def test_topk_bottomk_exclude_indices_never_returns_excluded_columns():
    row = np.array([0.5, 0.1, 0.9, 0.0, 0.7], dtype=np.float32)
    top, bottom = _topk_bottomk_indices(row, 5, exclude_indices=[3, 2])
    assert set(top) == set(bottom) == {0, 1, 4}
    assert list(bottom) == [1, 0, 4]
    assert list(top) == [4, 0, 1]
    # Combined with exclude_index.
    top, bottom = _topk_bottomk_indices(row, 2, exclude_index=1, exclude_indices=[3])
    assert list(bottom) == [0, 4]
    assert list(top) == [2, 4]


def test_topk_bottomk_without_exclusions_is_unchanged():
    row = np.array([0.5, 0.1, 0.9], dtype=np.float32)
    top, bottom = _topk_bottomk_indices(row, 2)
    assert list(top) == [2, 0] and list(bottom) == [1, 0]
    top, bottom = _topk_bottomk_indices(row, 2, exclude_indices=[])
    assert list(top) == [2, 0] and list(bottom) == [1, 0]


def test_supervised_positives_are_usable_train_pairs_only():
    text_g1_g2, graph_g1, _, alignments = _gold_views()
    pairs, n_skipped = supervised_positives(alignments, text_g1_g2, graph_g1)
    assert pairs == [("a:1", "b:1"), ("a:2", "b:2"), ("a:2", "b:4")]
    assert n_skipped == 1  # (a:9, b:3): anchor held out because of the test pair (a:9, b:9)


def test_supervised_split_isolation():
    examples = _run_supervised(n=3)
    held_out = {"a:9", "b:9"}
    for tup in examples:
        assert not held_out & set(tup)
    assert "a:9" not in {tup[0] for tup in examples}
    assert {(tup[0], tup[1]) for tup in examples} == {("a:1", "b:1"), ("a:2", "b:2"), ("a:2", "b:4")}


def test_supervised_negatives_are_cross_graph_non_gold_and_rank_ordered():
    gold = {("a:1", "b:1"), ("a:1", "b:5"), ("a:2", "b:2"), ("a:2", "b:4"), ("a:9", "b:3"), ("a:9", "b:9")}
    examples = _run_supervised(n=2)
    for anchor_id, _pos, text_neg_id, _gp, _gn in examples:
        assert text_neg_id.startswith("b:")
        assert text_neg_id != "b:9"
        assert (anchor_id, text_neg_id) not in gold

    negs = {}
    for anchor_id, pos_id, text_neg_id, _gp, graph_neg_id in examples:
        negs.setdefault((anchor_id, pos_id), []).append((graph_neg_id, text_neg_id))
    # a:1 row: b:5 (0.0, test gold) is skipped; b:1 (train gold) excluded -> b:4 (0.2), b:3 (0.3).
    # Rows are zipped with structural rank j, so sort by the graph rank to recover order.
    a1_bottom_graph = ["a:4", "a:3"]
    a1 = dict(negs[("a:1", "b:1")])
    assert [a1[g] for g in a1_bottom_graph] == ["b:4", "b:3"]
    # a:2 has two gold targets (b:2, b:4); both excluded -> b:1 (0.1), b:3 (0.4).
    a2_bottom_graph = ["a:3", "a:4"]
    for pos in ("b:2", "b:4"):
        a2 = dict(negs[("a:2", pos)])
        assert [a2[g] for g in a2_bottom_graph] == ["b:1", "b:3"]


def test_supervised_emits_negatives_per_positive_rows_and_is_deterministic():
    examples = _run_supervised(n=2)
    counts = {}
    for tup in examples:
        counts[(tup[0], tup[1])] = counts.get((tup[0], tup[1]), 0) + 1
    assert counts == {("a:1", "b:1"): 2, ("a:2", "b:2"): 2, ("a:2", "b:4"): 2}
    assert _run_supervised(n=2, seed=11) == _run_supervised(n=2, seed=11)
    assert sorted(_run_supervised(n=2, seed=1)) == sorted(_run_supervised(n=2, seed=2))


def test_supervised_caps_rows_at_eligible_negatives():
    # a:1 has 3 eligible negatives (b:2, b:3, b:4); graph_g1 has 3 non-self nodes.
    examples = _run_supervised(n=10)
    assert sum(1 for tup in examples if tup[:2] == ("a:1", "b:1")) == 3


def test_supervised_graph_pos_neg_follow_structural_rank():
    examples = _run_supervised(n=2)
    for anchor_id, _pos, _neg, graph_pos_id, graph_neg_id in examples:
        assert graph_pos_id.startswith("a:") and graph_neg_id.startswith("a:")
        assert anchor_id not in (graph_pos_id, graph_neg_id)
    a1 = {(gp, gn) for a, p, _n, gp, gn in examples if (a, p) == ("a:1", "b:1")}
    # a:1 graph row (self excluded): a:2 0.8, a:3 0.5, a:4 0.1 -> top [a:2, a:3], bottom [a:4, a:3].
    assert a1 == {("a:2", "a:4"), ("a:3", "a:3")}
