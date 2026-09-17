"""Tests for the seeded per-graph_role train/val/test partition."""

from __future__ import annotations

import pytest

from cga.core.errors import ConfigError
from cga.pipelines.ingest.splitter import split_nodes, validate_fractions


def test_validate_fractions_accepts_valid_input():
    validate_fractions([0.8, 0.1, 0.1])  # no raise


def test_validate_fractions_rejects_wrong_length():
    with pytest.raises(ConfigError):
        validate_fractions([0.8, 0.2])


def test_validate_fractions_rejects_non_summing():
    with pytest.raises(ConfigError):
        validate_fractions([0.8, 0.1, 0.2])


def test_validate_fractions_rejects_negative():
    with pytest.raises(ConfigError):
        validate_fractions([1.1, -0.1, 0.0])


def test_split_nodes_proportional_per_graph_role():
    node_ids_by_role = {
        "graph_1": [f"a:{i}" for i in range(100)],
        "graph_2": [f"b:{i}" for i in range(50)],
    }
    result = split_nodes(node_ids_by_role, fractions=[0.8, 0.1, 0.1], seed=42)

    g1_counts = {s: 0 for s in ("train", "val", "test")}
    for nid in node_ids_by_role["graph_1"]:
        g1_counts[result[nid]] += 1
    assert g1_counts == {"train": 80, "val": 10, "test": 10}

    g2_counts = {s: 0 for s in ("train", "val", "test")}
    for nid in node_ids_by_role["graph_2"]:
        g2_counts[result[nid]] += 1
    assert g2_counts == {"train": 40, "val": 5, "test": 5}


def test_split_nodes_covers_every_node_exactly_once():
    node_ids_by_role = {
        "graph_1": [f"a:{i}" for i in range(37)],
        "graph_2": [f"b:{i}" for i in range(13)],
    }
    result = split_nodes(node_ids_by_role, fractions=[0.8, 0.1, 0.1], seed=1)

    all_ids = set(node_ids_by_role["graph_1"]) | set(node_ids_by_role["graph_2"])
    assert set(result.keys()) == all_ids
    assert all(v in ("train", "val", "test") for v in result.values())


def test_split_nodes_is_reproducible_under_fixed_seed():
    node_ids_by_role = {"graph_1": [f"a:{i}" for i in range(20)], "graph_2": [f"b:{i}" for i in range(20)]}
    r1 = split_nodes(node_ids_by_role, fractions=[0.8, 0.1, 0.1], seed=99)
    r2 = split_nodes(node_ids_by_role, fractions=[0.8, 0.1, 0.1], seed=99)
    assert r1 == r2


def test_split_nodes_differs_under_different_seed():
    node_ids_by_role = {"graph_1": [f"a:{i}" for i in range(50)], "graph_2": [f"b:{i}" for i in range(50)]}
    r1 = split_nodes(node_ids_by_role, fractions=[0.8, 0.1, 0.1], seed=1)
    r2 = split_nodes(node_ids_by_role, fractions=[0.8, 0.1, 0.1], seed=2)
    assert r1 != r2


def test_split_nodes_small_graph_never_drops_or_duplicates():
    # Small enough that rounding could over/under-allocate without the min() guards.
    node_ids_by_role = {"graph_1": [f"a:{i}" for i in range(3)], "graph_2": [f"b:{i}" for i in range(1)]}
    result = split_nodes(node_ids_by_role, fractions=[0.8, 0.1, 0.1], seed=5)
    all_ids = set(node_ids_by_role["graph_1"]) | set(node_ids_by_role["graph_2"])
    assert set(result.keys()) == all_ids
