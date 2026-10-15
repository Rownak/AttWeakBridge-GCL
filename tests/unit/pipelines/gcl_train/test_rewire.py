"""Degree-preserving rewiring (structural control A5)."""

from __future__ import annotations

from collections import Counter

import pytest

from cga.pipelines.gcl_train.rewire import rewire_edges

ROLES = ["g1"] * 8 + ["g2"] * 6


def _edges() -> list[tuple[int, int]]:
    g1 = [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 2), (7, 2), (7, 3), (5, 3)]
    g2 = [(9, 8), (10, 8), (11, 9), (12, 9), (13, 10), (13, 11)]
    return g1 + g2


def test_rewire_preserves_in_and_out_degree_roles_and_simplicity():
    edges = _edges()
    out = rewire_edges(edges, ROLES, seed=3)
    assert len(out) == len(set(out)) == len(set(edges))
    assert Counter(s for s, _ in out) == Counter(s for s, _ in edges)
    assert Counter(d for _, d in out) == Counter(d for _, d in edges)
    assert all(s != d for s, d in out)
    assert all(ROLES[s] == ROLES[d] for s, d in out)  # no cross-graph edge


def test_rewire_changes_the_graph_and_is_seed_deterministic():
    edges = _edges()
    a, b, c = rewire_edges(edges, ROLES, seed=3), rewire_edges(edges, ROLES, seed=3), rewire_edges(edges, ROLES, seed=4)
    assert a == b
    assert a != sorted(set(edges)) and a != c


def test_rewire_collapses_duplicates_and_rejects_cross_graph_edges():
    assert len(rewire_edges(_edges() + _edges(), ROLES, seed=0)) == len(set(_edges()))
    with pytest.raises(ValueError):
        rewire_edges([(0, 9)], ROLES, seed=0)
    assert rewire_edges([(1, 0)], ROLES, seed=0) == [(1, 0)]  # a single edge cannot be swapped
