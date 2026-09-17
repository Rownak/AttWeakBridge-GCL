"""Relevance providers for the ``neighborhood`` evaluator (target_architecture.md §5.5a).

A ``RelevanceProvider`` returns the relevant-item **set** for a query, never
a ranked list -- this is what lets every metric in ``metrics.py`` be a pure
set operation regardless of whether the underlying view is itself ranked
(``cosine_top_k``) or not (``hop_radius``). Deliberately no ``is_ranked``
flag on the protocol: that would force the scoring loop to branch per view
and reintroduce the two code paths this abstraction exists to remove.

Registered by ``kind`` in ``relevance_providers`` so ``neighborhood``'s
``views[].provider`` config dispatches generically, with a ``ConfigError``
on typo.
"""

from __future__ import annotations

from typing import Any, Protocol

import networkx as nx
import numpy as np

from cga.core.errors import ConfigError
from cga.core.registry import Registry
from cga.data.aligned import AlignedMatrix

relevance_providers: Registry = Registry("relevance provider")


class RelevanceProvider(Protocol):
    """Defines ground-truth relevance for one view, over a frozen candidate pool.

    Contract: :meth:`prepare` is called exactly once per (view, role) with
    the final, method-intersected candidate list; a provider precomputes
    whatever it needs there. :meth:`defined_for` returns the subset of
    ``candidate_ids`` with meaningful, non-empty ground truth -- this is the
    hook that expresses "structural excludes isolates" without the pipeline
    hardcoding that rule. :meth:`relevant` always excludes the query itself
    and returns a subset of ``candidate_ids``.
    """

    def prepare(self, *, candidate_ids: list[str]) -> None:
        """Precompute this provider's relevance sets over ``candidate_ids``."""
        ...

    def defined_for(self) -> set[str]:
        """Return the subset of the prepared candidates with non-empty ground truth."""
        ...

    def relevant(self, query_id: str) -> set[str]:
        """Return the relevant set for ``query_id`` (excludes ``query_id`` itself)."""
        ...


class HopRadiusRelevance:
    """Relevant = every node within ``radius`` hops, by shortest path.

    A set rather than a top-k, because hop distance is integer-valued and
    massively tied -- a node's "10 nearest by shortest path" is usually 2
    nodes at distance 1 plus an arbitrary slice of a much larger distance-2
    ring, which would make any top-k definition an artifact of the tie-break
    rule. Defining relevance as the full set within a radius removes the
    arbitrariness: no tie-break is needed because no ordering is needed.

    Distances are computed with a radius-bounded BFS (``cutoff=radius``) on
    the *candidate-restricted* subgraph passed to :meth:`prepare` -- same-role
    containment becomes the structural boundary, and a node whose only graph
    neighbors fall outside the candidate pool is correctly treated as
    isolated within this scope. Isolates get an empty relevant set and so
    are naturally excluded via :meth:`defined_for`, with no isolate-specific
    branch anywhere in this class.
    """

    def __init__(self, graph: nx.Graph, radius: int) -> None:
        if radius < 1:
            raise ConfigError(f"hop_radius radius must be >= 1, got {radius}")
        self._graph = graph
        self._radius = radius
        self._relevant_by_id: dict[str, set[str]] = {}

    def prepare(self, *, candidate_ids: list[str]) -> None:
        candidate_set = set(candidate_ids)
        sub = self._graph.subgraph(candidate_ids)
        self._relevant_by_id = {}
        for node_id in candidate_ids:
            if node_id not in sub:
                self._relevant_by_id[node_id] = set()
                continue
            distances = nx.single_source_shortest_path_length(sub, node_id, cutoff=self._radius)
            self._relevant_by_id[node_id] = {
                nid for nid in distances if nid != node_id and nid in candidate_set
            }

    def defined_for(self) -> set[str]:
        return {nid for nid, relevant in self._relevant_by_id.items() if relevant}

    def relevant(self, query_id: str) -> set[str]:
        return self._relevant_by_id.get(query_id, set())


class CosineTopKRelevance:
    """Relevant = the top-``top_k`` most cosine-similar nodes in a reference embedding store.

    Used e.g. to contrast a domain text method against a *generic* text
    reference (a plain ``bert-base-uncased`` store), or to define a textual
    ground truth independent of any method under comparison.
    """

    def __init__(self, reference_matrix: AlignedMatrix, top_k: int) -> None:
        if top_k < 1:
            raise ConfigError(f"cosine_top_k top_k must be >= 1, got {top_k}")
        self._reference_matrix = reference_matrix
        self._top_k = top_k
        self._relevant_by_id: dict[str, set[str]] = {}

    def prepare(self, *, candidate_ids: list[str]) -> None:
        from cga.pipelines.evaluate.metrics import top_k_by_similarity

        available = [nid for nid in candidate_ids if nid in self._reference_matrix.node_ids]
        self._relevant_by_id = {}
        if len(available) < 2:
            return
        sub = self._reference_matrix.select(available)
        sim = _cosine_similarity(sub.vectors, sub.vectors)
        for i, node_id in enumerate(available):
            others = available[:i] + available[i + 1 :]
            row = np.concatenate([sim[i, :i], sim[i, i + 1 :]])
            self._relevant_by_id[node_id] = set(top_k_by_similarity(row, others, self._top_k))

    def defined_for(self) -> set[str]:
        return {nid for nid, relevant in self._relevant_by_id.items() if relevant}

    def relevant(self, query_id: str) -> set[str]:
        return self._relevant_by_id.get(query_id, set())


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the ``(len(a), len(b))`` cosine similarity matrix between two stacks of vectors."""
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


@relevance_providers.register("hop_radius")
def _build_hop_radius(params: dict[str, Any], *, graph: nx.Graph, **_: Any) -> HopRadiusRelevance:
    return HopRadiusRelevance(graph, radius=params["radius"])


@relevance_providers.register("cosine_top_k")
def _build_cosine_top_k(
    params: dict[str, Any], *, load_reference: Any, **_: Any
) -> CosineTopKRelevance:
    reference_matrix = load_reference(params["reference"])
    return CosineTopKRelevance(reference_matrix, top_k=params["top_k"])
