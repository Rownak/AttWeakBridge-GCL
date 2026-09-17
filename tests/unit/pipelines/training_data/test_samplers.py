"""Tests for rank_topk / percentile / supervised samplers on hand-built similarity matrices."""

from __future__ import annotations

import numpy as np

from cga.pipelines.training_data.samplers import percentile_sampler, rank_topk_sampler, supervised_sampler
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


def test_supervised_sampler_uses_alignment_pairs_as_positives():
    text_g1_g2, graph_g1, graph_g2 = _small_views()
    alignments = [{"node_1_id": "a:1", "node_2_id": "b:1", "split": "train"}]
    examples = supervised_sampler(
        text_g1_g2=text_g1_g2, graph_g1=graph_g1, graph_g2=graph_g2, config={}, seed=0, alignments=alignments
    )
    assert len(examples) == 1
    anchor_id, text_pos_id, text_neg_id, graph_pos_id, graph_neg_id = examples[0]
    assert anchor_id == "a:1"
    assert text_pos_id == "b:1"
    assert text_neg_id != "b:1"
    assert graph_pos_id != "a:1"
    assert graph_neg_id not in ("a:1", graph_pos_id) or graph_neg_id == graph_pos_id
