"""Unit tests for set-based neighborhood-preservation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cga.pipelines.evaluate.metrics import (
    hit_at_k,
    mrr,
    mrr_via_argsort,
    normalized_precision_at_k,
    rank_by_similarity,
    recall_at_k,
    top_k,
    top_k_by_similarity,
)

LEARNED_RANKED = ["b", "a", "d", "c"]


def test_rank_by_similarity_orders_descending_ties_by_id():
    sim = np.array([0.5, 0.9, 0.9, 0.1])
    ids = ["z", "b", "a", "m"]
    assert rank_by_similarity(sim, ids) == ["a", "b", "z", "m"]


def test_rank_by_similarity_length_mismatch_raises():
    with pytest.raises(ValueError):
        rank_by_similarity(np.array([0.1, 0.2]), ["a"])


def test_top_k_truncates_and_rejects_non_positive_k():
    assert top_k(LEARNED_RANKED, 2) == ["b", "a"]
    assert top_k(LEARNED_RANKED, 10) == LEARNED_RANKED
    with pytest.raises(ValueError):
        top_k(LEARNED_RANKED, 0)


# --- the min(k, |R|) denominator: the single most important case -----------


def test_normalized_precision_denominator_is_min_k_and_relevant_size():
    # |R| = 2, k = 5: every relevant item retrieved -> 1.0, not 2/5 = 0.4.
    learned = ["a", "c", "b", "d"]
    relevant = {"a", "c"}
    assert normalized_precision_at_k(learned, relevant, 5) == pytest.approx(1.0)


def test_normalized_precision_and_recall_coincide_when_k_ge_relevant_size():
    # min(k, |R|) == |R| once k >= |R|, so both denominators become |R| and
    # the two metrics report the same value.
    learned = ["b", "a", "d", "c"]
    relevant = {"a", "c"}  # |R| = 2
    k = 3  # k >= |R|
    assert normalized_precision_at_k(learned, relevant, k) == pytest.approx(
        recall_at_k(learned, relevant, k)
    )


def test_normalized_precision_and_recall_diverge_when_k_lt_relevant_size():
    # k < |R|: normalized precision divides by min(k,|R|)=k (ordinary
    # precision), recall divides by |R| -- the two report different values
    # for a partial retrieval.
    learned = ["a", "b", "c", "d"]
    relevant = {"a", "d"}  # |R| = 2
    k = 1  # k < |R|, only "a" retrieved
    assert normalized_precision_at_k(learned, relevant, k) == pytest.approx(1.0)  # 1/min(1,2)=1/1
    assert recall_at_k(learned, relevant, k) == pytest.approx(0.5)  # 1/|R|=1/2


def test_normalized_precision_hand_computed():
    # From the original Phase 1.8 fixture: reference top-3 = {a, c, b}.
    learned = ["b", "a", "d", "c"]
    relevant = {"a", "c", "b"}
    # top-2 learned = {b, a}; intersection with relevant = {a, b} -> 2/min(2,3)=2/2=1.0
    assert normalized_precision_at_k(learned, relevant, 2) == pytest.approx(1.0)


def test_normalized_precision_zero_k_or_empty_relevant():
    assert normalized_precision_at_k(LEARNED_RANKED, {"a"}, 0) == 0.0
    assert normalized_precision_at_k(LEARNED_RANKED, set(), 2) == 0.0


def test_recall_zero_when_relevant_empty():
    assert recall_at_k(LEARNED_RANKED, set(), 2) == 0.0


# --- hit_at_k fires on ANY relevant item, not just a designated "nearest" --


def test_hit_at_k_fires_on_any_relevant_item():
    learned = ["x", "y", "c", "z"]
    relevant = {"a", "c"}  # "c" is relevant even though not "first"
    assert hit_at_k(learned, relevant, 3) == 1.0
    assert hit_at_k(learned, relevant, 2) == 0.0


def test_hit_at_k_empty_relevant_or_zero_k():
    assert hit_at_k(LEARNED_RANKED, set(), 2) == 0.0
    assert hit_at_k(LEARNED_RANKED, {"a"}, 0) == 0.0


# --- mrr returns reciprocal rank of the FIRST relevant item ----------------


def test_mrr_returns_first_relevant_rank():
    learned = ["b", "a", "d", "c"]
    relevant = {"d", "c"}  # first hit is "d" at rank 3
    assert mrr(learned, relevant) == pytest.approx(1 / 3)


def test_mrr_empty_relevant_or_not_found():
    assert mrr(LEARNED_RANKED, set()) == 0.0
    assert mrr(LEARNED_RANKED, {"q"}) == 0.0


# --- top_k_by_similarity (argpartition) agrees with rank_by_similarity -----


@pytest.mark.parametrize("seed", range(50))
def test_top_k_by_similarity_matches_full_rank_across_seeds(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 40)
    k = int(rng.integers(1, n + 5))
    ids = [f"id{i}" for i in range(n)]
    sims = rng.normal(size=n)
    expected = rank_by_similarity(sims, ids)[: min(k, n)]
    actual = top_k_by_similarity(sims, ids, k)
    assert actual == expected


def test_top_k_by_similarity_with_exact_ties():
    ids = ["z", "b", "a", "m", "q"]
    sims = np.array([0.5, 0.5, 0.5, 0.1, 0.9])
    # ties at 0.5 among z/b/a must break by id: a, b, z (m is excluded, lowest sim)
    assert top_k_by_similarity(sims, ids, 3) == ["q", "a", "b"]
    assert top_k_by_similarity(sims, ids, 4) == ["q", "a", "b", "z"]


def test_top_k_by_similarity_rejects_bad_k_or_mismatched_lengths():
    with pytest.raises(ValueError):
        top_k_by_similarity(np.array([0.1]), ["a", "b"], 1)
    with pytest.raises(ValueError):
        top_k_by_similarity(np.array([0.1, 0.2]), ["a", "b"], 0)


# --- mrr_via_argsort agrees with mrr(rank_by_similarity(...)) --------------


@pytest.mark.parametrize("seed", range(50))
def test_mrr_via_argsort_matches_scalar_mrr_across_seeds(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 30)
    ids = [f"id{i}" for i in range(n)]
    sims = rng.normal(size=n)
    n_relevant = int(rng.integers(0, n + 1))
    relevant = set(rng.choice(ids, size=n_relevant, replace=False)) if n_relevant else set()

    expected = mrr(rank_by_similarity(sims, ids), relevant)
    actual = mrr_via_argsort(sims, ids, relevant)
    assert actual == pytest.approx(expected)


def test_mrr_via_argsort_with_exact_ties_breaks_by_id():
    ids = ["z", "b", "a"]
    sims = np.array([0.5, 0.5, 0.5])  # exact 3-way tie
    # rank_by_similarity breaks ties by id: a, b, z. Relevant = {z} -> rank 3.
    assert mrr_via_argsort(sims, ids, {"z"}) == pytest.approx(1 / 3)
    # Relevant = {a} -> rank 1.
    assert mrr_via_argsort(sims, ids, {"a"}) == pytest.approx(1.0)


def test_mrr_via_argsort_empty_relevant():
    assert mrr_via_argsort(np.array([0.1, 0.2]), ["a", "b"], set()) == 0.0
