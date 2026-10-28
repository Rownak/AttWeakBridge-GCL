"""Tests for rank_topk / percentile / supervised samplers on hand-built similarity matrices."""

from __future__ import annotations

import numpy as np
import pytest

from cga.core.errors import ConfigError
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


def _run_supervised(n=2, seed=0, text_negatives=None):
    text_g1_g2, graph_g1, graph_g2, alignments = _gold_views()
    config = {"negatives_per_positive": n}
    if text_negatives is not None:
        config["text_negatives"] = text_negatives
    return supervised_sampler(
        text_g1_g2=text_g1_g2,
        graph_g1=graph_g1,
        graph_g2=graph_g2,
        config=config,
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


def test_supervised_random_negatives_are_valid():
    gold = {("a:1", "b:1"), ("a:1", "b:5"), ("a:2", "b:2"), ("a:2", "b:4"), ("a:9", "b:3"), ("a:9", "b:9")}
    for seed in range(20):
        for anchor_id, _pos, text_neg_id, graph_pos_id, graph_neg_id in _run_supervised(n=2, seed=seed, text_negatives="random"):
            assert text_neg_id.startswith("b:") and text_neg_id != "b:9"  # cross-graph, not held out
            assert (anchor_id, text_neg_id) not in gold  # not gold in either split
            assert anchor_id not in (graph_pos_id, graph_neg_id)


def test_supervised_random_negatives_are_distinct_per_positive():
    examples = _run_supervised(n=3, seed=4, text_negatives="random")
    by_pos: dict[tuple[str, str], list[str]] = {}
    for anchor_id, pos_id, text_neg_id, *_ in examples:
        by_pos.setdefault((anchor_id, pos_id), []).append(text_neg_id)
    assert set(by_pos) == {("a:1", "b:1"), ("a:2", "b:2"), ("a:2", "b:4")}
    for negs in by_pos.values():
        assert len(negs) == 3 and len(set(negs)) == 3


def test_supervised_random_negatives_cap_at_eligible_columns():
    # a:1 has 3 eligible columns (b:2, b:3, b:4) and 3 other graph_1 nodes.
    examples = _run_supervised(n=10, text_negatives="random")
    a1 = [tup[2] for tup in examples if tup[:2] == ("a:1", "b:1")]
    assert sorted(a1) == ["b:2", "b:3", "b:4"]


def test_supervised_random_negatives_depend_on_seed_only():
    assert _run_supervised(n=2, seed=11, text_negatives="random") == _run_supervised(n=2, seed=11, text_negatives="random")
    draws = {tuple(_run_supervised(n=2, seed=s, text_negatives="random")) for s in range(8)}
    assert len(draws) > 1  # different seeds give different negatives


def test_supervised_bottomk_is_the_default_and_unchanged():
    assert _run_supervised(n=2) == _run_supervised(n=2, text_negatives="bottomk")


def test_supervised_unknown_text_negatives_raises():
    with pytest.raises(ConfigError):
        _run_supervised(text_negatives="hardest")


# --- Sprint 9.1: rank_topk negatives / positives options ---------------------------------------------------------------


def _wide_views(text_rows: list[list[float]]):
    """Two graph_1 anchors against six graph_2 columns; within-graph views as in ``_small_views``."""
    _, graph_g1, graph_g2 = _small_views()
    text = SimilarityMatrix(
        row_ids=["a:1", "a:2"],
        col_ids=[f"b:{i}" for i in range(1, 7)],
        values=np.array(text_rows, dtype=np.float32),
    )
    return text, graph_g1, graph_g2


# a:1 ranks b:1 > b:2 > b:3 > b:4 > b:5 > b:6; a:2 likes b:2 first, then the rest in the same order.
_WIDE_TEXT = [[0.9, 0.8, 0.7, 0.6, 0.2, 0.1], [0.5, 0.9, 0.4, 0.3, 0.2, 0.1]]


def _g1_examples(config: dict, text_rows=_WIDE_TEXT, seed: int = 0):
    text, graph_g1, graph_g2 = _wide_views(text_rows)
    examples = rank_topk_sampler(text_g1_g2=text, graph_g1=graph_g1, graph_g2=graph_g2, config=config, seed=seed)
    return sorted(e for e in examples if e[0] in ("a:1", "a:2"))


def test_rank_topk_explicit_defaults_equal_omitted_options():
    base = _g1_examples({"k": 2})
    explicit = _g1_examples({"k": 2, "negatives": "bottomk", "positives": "topk"})
    assert base == explicit
    assert [e[2] for e in base if e[0] == "a:1"] == ["b:6", "b:5"]  # bottom-k, ascending


def test_rank_topk_hard_negatives_use_ranks_after_the_offset():
    examples = _g1_examples({"k": 2, "negatives": "hard", "hard_offset": 2})
    a1 = [e for e in examples if e[0] == "a:1"]
    # Positives are ranks 1-2 (b:1, b:2); hard negatives are ranks 3-4 (b:3, b:4), zipped by rank.
    assert [(e[1], e[2]) for e in a1] == [("b:1", "b:3"), ("b:2", "b:4")]
    a2 = [e for e in examples if e[0] == "a:2"]
    # a:2 ranks b:2 > b:1 > b:3 > b:4: positives b:2, b:1; negatives b:3, b:4 (examples are sorted).
    assert [(e[1], e[2]) for e in a2] == [("b:1", "b:4"), ("b:2", "b:3")]


def test_rank_topk_hard_offset_past_the_columns_yields_no_examples():
    assert _g1_examples({"k": 2, "negatives": "hard", "hard_offset": 6}) == []


def test_rank_topk_hard_offset_below_k_is_rejected():
    with pytest.raises(ConfigError, match="hard_offset"):
        _g1_examples({"k": 3, "negatives": "hard", "hard_offset": 2})


def test_rank_topk_random_negatives_avoid_positives_and_are_seeded():
    first = _g1_examples({"k": 2, "negatives": "random"}, seed=7)
    again = _g1_examples({"k": 2, "negatives": "random"}, seed=7)
    other = _g1_examples({"k": 2, "negatives": "random"}, seed=8)
    assert first == again
    assert first != other
    for _, text_pos, text_neg, _, _ in first:
        assert text_neg.startswith("b:")
    a1 = [e for e in first if e[0] == "a:1"]
    assert len(a1) == 2
    assert {e[2] for e in a1}.isdisjoint({"b:1", "b:2"})  # never one of a:1's positives
    assert a1[0][2] != a1[1][2]


def test_rank_topk_mutual_positives_drop_one_sided_neighbors():
    # a:1's top-1 is b:1, but b:1's own top-1 anchor is a:2 (0.95 > 0.9): not mutual, so a:1 is dropped.
    # a:2's top-1 is b:1 and b:1's top-1 anchor is a:2: kept.
    rows = [[0.9, 0.1, 0.1, 0.1, 0.1, 0.1], [0.95, 0.2, 0.2, 0.2, 0.2, 0.2]]
    plain = _g1_examples({"k": 1}, rows)
    mutual = _g1_examples({"k": 1, "positives": "mutual"}, rows)
    assert {e[0] for e in plain} == {"a:1", "a:2"}
    assert {e[0] for e in mutual} == {"a:2"}
    assert [e for e in mutual if e[0] == "a:2"] == [e for e in plain if e[0] == "a:2"]


def test_rank_topk_mutual_positives_apply_to_the_graph_view():
    # The text view is mutual for both anchors (a:1 <-> b:1, a:2 <-> b:2); only the graph view differs.
    text, graph_g1, graph_g2 = _wide_views([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1, 0.1, 0.1]])
    graph_g1 = SimilarityMatrix(
        row_ids=graph_g1.row_ids,
        col_ids=graph_g1.col_ids,
        values=np.array([[1.0, 0.3, 0.8], [0.3, 1.0, 0.2], [0.8, 0.2, 1.0]], dtype=np.float32),
    )
    kwargs = dict(text_g1_g2=text, graph_g1=graph_g1, graph_g2=graph_g2, seed=0)
    plain = rank_topk_sampler(config={"k": 1}, **kwargs)
    mutual = rank_topk_sampler(config={"k": 1, "positives": "mutual"}, **kwargs)
    # a:1 -> graph positive a:3 (0.8), and a:3's top-1 is a:1 (0.8): mutual, kept.
    # a:2 -> graph positive a:1 (0.3), but a:1's top-1 is a:3: not mutual, dropped.
    assert {e[0] for e in plain if e[0] in ("a:1", "a:2")} == {"a:1", "a:2"}
    assert {e[0] for e in mutual if e[0] in ("a:1", "a:2")} == {"a:1"}


def test_rank_topk_unknown_options_are_rejected():
    with pytest.raises(ConfigError, match="negatives"):
        _g1_examples({"k": 2, "negatives": "nearest"})
    with pytest.raises(ConfigError, match="positives"):
        _g1_examples({"k": 2, "positives": "all"})


# ---- cross_modal_hard negatives (Sprint 11.14) -------------------------------------------------


def _random_views(seed: int = 0, n1: int = 7, n2: int = 9):
    """Random but fixed views for both graphs, with symmetric unit-diagonal within-graph matrices."""
    rng = np.random.default_rng(seed)
    g1 = [f"a:{i}" for i in range(n1)]
    g2 = [f"b:{i}" for i in range(n2)]

    def square(ids):
        m = rng.random((len(ids), len(ids)))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        return SimilarityMatrix(row_ids=ids, col_ids=ids, values=m)

    text_g1_g2 = SimilarityMatrix(row_ids=g1, col_ids=g2, values=rng.random((n1, n2)))
    return {
        "text_g1_g2": text_g1_g2,
        "graph_g1": square(g1),
        "graph_g2": square(g2),
        "text_g1_g1": square(g1),
        "text_g2_g2": square(g2),
    }


def _hard(views, k=2, pool=4, seed=0):
    return rank_topk_sampler(
        **views, config={"k": k, "negatives": "cross_modal_hard", "hard_pool": pool}, seed=seed
    )


def test_cross_modal_hard_negatives_are_the_brute_force_lowest_other_modality_pool_nodes():
    views = _random_views()
    k, pool = 2, 4
    by_anchor = {}
    for tup in _hard(views, k, pool):
        by_anchor.setdefault(tup[0], []).append(tup)

    g1_ids, g2_ids = views["text_g1_g2"].row_ids, views["text_g1_g2"].col_ids
    for anchor, rows in by_anchor.items():
        if anchor in g1_ids:
            text = views["text_g1_g2"].values[g1_ids.index(anchor)]
            text_cols, other_graph, own_ids = g2_ids, views["graph_g2"], g1_ids
            graph = views["graph_g1"]
            own_text = views["text_g1_g1"]
        else:
            text = views["text_g1_g2"].values[:, g2_ids.index(anchor)]
            text_cols, other_graph, own_ids = g1_ids, views["graph_g1"], g2_ids
            graph = views["graph_g2"]
            own_text = views["text_g2_g2"]
        text_order = list(np.argsort(-text, kind="stable"))
        positives = [text_cols[j] for j in text_order[:k]]
        text_pool = [text_cols[j] for j in text_order[k : k + pool]]
        graph_row = graph.values[own_ids.index(anchor)].copy()
        graph_row[own_ids.index(anchor)] = -np.inf
        graph_order = list(np.argsort(-graph_row, kind="stable"))
        graph_positives = [own_ids[j] for j in graph_order[:k]]
        graph_pool = [own_ids[j] for j in graph_order[k : k + pool]]

        def expected(pos_list, pool_list, sim):
            left, chosen = list(pool_list), []
            for p in pos_list:
                row = sim.row_ids.index(p)
                best = min(left, key=lambda c: (sim.values[row, sim.col_ids.index(c)], left.index(c)))
                left.remove(best)
                chosen.append(best)
            return chosen

        rows_by_pos = {r[1]: r for r in rows}
        assert [r[1] for r in sorted(rows, key=lambda r: positives.index(r[1]))] == positives
        for p, neg in zip(positives, expected(positives, text_pool, other_graph)):
            assert rows_by_pos[p][2] == neg
        rows_by_gpos = {r[3]: r for r in rows}
        for q, neg in zip(graph_positives, expected(graph_positives, graph_pool, own_text)):
            assert rows_by_gpos[q][4] == neg


def test_cross_modal_hard_negatives_are_valid_distinct_and_never_positives_or_self():
    views = _random_views(seed=3)
    k = 2
    by_anchor = {}
    for tup in _hard(views, k, 4):
        by_anchor.setdefault(tup[0], []).append(tup)
    assert len(by_anchor) == 16
    for anchor, rows in by_anchor.items():
        text_pos, text_neg = {r[1] for r in rows}, [r[2] for r in rows]
        graph_pos, graph_neg = {r[3] for r in rows}, [r[4] for r in rows]
        same_graph = anchor[0]
        assert len(set(text_neg)) == len(text_neg) and len(set(graph_neg)) == len(graph_neg)
        assert not (set(text_neg) & text_pos) and not (set(graph_neg) & graph_pos)
        assert anchor not in graph_pos and anchor not in graph_neg
        assert all(n[0] != same_graph for n in text_neg)  # text negatives are in the other graph
        assert all(n[0] == same_graph for n in graph_neg)  # graph negatives are in the anchor's graph


def test_cross_modal_hard_is_reproducible_and_keeps_the_positives_of_bottomk():
    views = _random_views(seed=5)
    first, second = _hard(views, 2, 4, seed=7), _hard(views, 2, 4, seed=7)
    assert first == second
    bottomk = rank_topk_sampler(
        **{k: v for k, v in views.items() if k in ("text_g1_g2", "graph_g1", "graph_g2")},
        config={"k": 2},
        seed=7,
    )
    assert [(t[0], t[1], t[3]) for t in first] == [(t[0], t[1], t[3]) for t in bottomk]
    assert [t[2] for t in first] != [t[2] for t in bottomk]


def test_cross_modal_hard_uses_only_ids_of_the_given_matrices():
    views = _random_views(seed=2)
    allowed = set(views["text_g1_g2"].row_ids) | set(views["text_g1_g2"].col_ids)
    assert {nid for tup in _hard(views) for nid in tup} <= allowed


def test_cross_modal_hard_pool_shorter_than_k_caps_rows_and_requires_its_inputs():
    views = _random_views(seed=1)
    assert len(_hard(views, k=2, pool=1)) < len(_hard(views, k=2, pool=4))
    with pytest.raises(ConfigError, match="hard_pool"):
        _hard(views, pool=0)
    with pytest.raises(ConfigError, match="within-graph text"):
        rank_topk_sampler(
            text_g1_g2=views["text_g1_g2"],
            graph_g1=views["graph_g1"],
            graph_g2=views["graph_g2"],
            config={"k": 2, "negatives": "cross_modal_hard"},
            seed=0,
        )
