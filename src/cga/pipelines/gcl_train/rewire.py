"""Degree-preserving edge rewiring: the structural control A5 of Sprint 5 (dev_docs/sprints/sprint_5.md).

SRGA's relational adapter is compared against the *same* model on a rewired graph. If it still wins
there, the gain came from model capacity, not from the ontology's structure. The rewiring keeps every
node's in- and out-degree and every edge inside its own graph, so degree statistics are identical and
only *who is connected to whom* is destroyed.
"""

from __future__ import annotations

import random


def rewire_edges(
    edges: list[tuple[int, int]], graph_roles: list[str], *, seed: int, swaps_per_edge: int = 10
) -> list[tuple[int, int]]:
    """Randomize directed edges with degree-preserving double-edge swaps, within each graph role.

    A swap takes two stored edges ``(a, b)`` and ``(c, d)`` and replaces them with ``(a, d)`` and
    ``(c, b)``. Every node keeps its out-degree (as a source) and in-degree (as a target). A swap is
    skipped when it would create a self-loop or an edge that already exists, so the result is a
    simple graph. Roles are processed independently, so no cross-graph edge can appear.

    Args:
        edges: Stored ``(src_row, dst_row)`` edges (child -> parent). Duplicates are collapsed.
        graph_roles: ``graph_roles[row]`` is the graph role of each universe row.
        seed: Seed for ``random.Random``; the same seed gives the same result.
        swaps_per_edge: Swap attempts per edge in each role (10 mixes the graph well in practice).

    Returns:
        The rewired edges, sorted. Same length as the de-duplicated input.

    Raises:
        ValueError: If an edge joins two different graph roles.
    """
    unique = sorted(set(edges))
    by_role: dict[str, list[tuple[int, int]]] = {}
    for s, d in unique:
        if graph_roles[s] != graph_roles[d]:
            raise ValueError(f"edge ({s}, {d}) joins graph roles {graph_roles[s]} and {graph_roles[d]}")
        by_role.setdefault(graph_roles[s], []).append((s, d))

    rng = random.Random(seed)
    out: list[tuple[int, int]] = []
    for role in sorted(by_role):
        role_edges = by_role[role]
        present = set(role_edges)
        n = len(role_edges)
        if n >= 2:
            for _ in range(swaps_per_edge * n):
                i, j = rng.randrange(n), rng.randrange(n)
                (a, b), (c, d) = role_edges[i], role_edges[j]
                if i == j or a == d or c == b or (a, d) in present or (c, b) in present:
                    continue
                present.difference_update(((a, b), (c, d)))
                present.update(((a, d), (c, b)))
                role_edges[i], role_edges[j] = (a, d), (c, b)
        out.extend(role_edges)
    return sorted(out)
