"""Hand-computed checks for Precision@k, Hit@k, MRR, and ranking helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cga.pipelines.evaluate.metrics import hit_at_k, mrr, precision_at_k, rank_by_similarity, top_k


def test_rank_by_similarity_orders_descending():
    ranked = rank_by_similarity(np.array([0.1, 0.9, 0.5]), ["a", "b", "c"])
    assert ranked == ["b", "c", "a"]


def test_rank_by_similarity_breaks_ties_by_id():
    # "a" and "c" tie at 0.5; alphabetical id order breaks the tie deterministically.
    ranked = rank_by_similarity(np.array([0.5, 0.1, 0.5]), ["c", "b", "a"])
    assert ranked == ["a", "c", "b"]


def test_rank_by_similarity_rejects_length_mismatch():
    with pytest.raises(ValueError):
        rank_by_similarity(np.array([0.1, 0.2]), ["a", "b", "c"])


def test_top_k_truncates():
    assert top_k(["a", "b", "c", "d"], 2) == ["a", "b"]


def test_top_k_rejects_zero():
    with pytest.raises(ValueError):
        top_k(["a", "b"], 0)


LEARNED_RANKED = ["b", "a", "d", "c"]
REFERENCE_TOP = ["a", "c", "b"]  # reference's nearest neighbor is "a"


def test_precision_at_k_hand_computed():
    # learned top-2 = {b, a}; reference top-2 = {a, c}; overlap = {a} -> 1/2.
    assert precision_at_k(LEARNED_RANKED, REFERENCE_TOP[:2], 2) == pytest.approx(0.5)


def test_precision_at_k_full_overlap():
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)


def test_precision_at_k_no_overlap():
    assert precision_at_k(["a", "b"], ["c", "d"], 2) == pytest.approx(0.0)


def test_precision_at_k_zero_k_returns_zero():
    assert precision_at_k(LEARNED_RANKED, REFERENCE_TOP, 0) == 0.0


def test_hit_at_k_true_when_nearest_within_cutoff():
    # reference nearest = "a"; learned top-2 = {b, a} -> hit.
    assert hit_at_k(LEARNED_RANKED, REFERENCE_TOP, 2) == 1.0


def test_hit_at_k_false_when_nearest_outside_cutoff():
    # learned top-1 = {b}; "a" is not in it -> miss.
    assert hit_at_k(LEARNED_RANKED, REFERENCE_TOP, 1) == 0.0


def test_hit_at_k_empty_reference_is_zero():
    assert hit_at_k(LEARNED_RANKED, [], 5) == 0.0


def test_mrr_hand_computed():
    # "a" sits at rank 2 (1-indexed) in LEARNED_RANKED -> reciprocal rank 1/2.
    assert mrr(LEARNED_RANKED, REFERENCE_TOP) == pytest.approx(0.5)


def test_mrr_rank_one_is_one():
    assert mrr(["a", "b", "c"], ["a"]) == pytest.approx(1.0)


def test_mrr_missing_reference_neighbor_is_zero():
    assert mrr(["b", "c"], ["z"]) == 0.0


def test_mrr_empty_reference_is_zero():
    assert mrr(LEARNED_RANKED, []) == 0.0
