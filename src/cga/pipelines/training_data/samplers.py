"""Registered samplers emitting anchor/positive/negative example rows (target_architecture.md §5.3 step 3).

Every sampler here only ever sees the non-``held_out`` similarity matrices
built by :mod:`cga.pipelines.training_data.similarity`, so a held-out
(alignment-test) node can never be drawn as an anchor, positive, or negative by
``rank_topk``/``percentile`` — the leakage-prevention property is structural,
not a filter applied after the fact.

Each sampler returns a list of ``(anchor_id, text_pos_id, text_neg_id,
graph_pos_id, graph_neg_id)`` tuples; the pipeline attaches ``example_id``,
``sampler``, and ``split``.
"""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np

from cga.core.registry import samplers
from cga.pipelines.training_data.similarity import SimilarityMatrix

ExampleTuple = tuple[str, str, str, str, str]


class Sampler(Protocol):
    """A registered sampler: similarity views + config -> example tuples."""

    def __call__(
        self,
        *,
        text_g1_g2: SimilarityMatrix,
        graph_g1: SimilarityMatrix,
        graph_g2: SimilarityMatrix,
        config: dict,
        seed: int,
        alignments: list[dict] | None = None,
    ) -> list[ExampleTuple]: ...


def _topk_bottomk_indices(row: np.ndarray, k: int, *, exclude_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (most-similar-k, least-similar-k) column indices for one similarity row, by rank.

    ``exclude_index`` drops one column before ranking — used to remove a
    within-graph anchor's self-similarity (always 1.0, always rank-0) from its
    own top-k, so an anchor is never paired with itself as a "positive".
    """
    if exclude_index is not None:
        keep = np.arange(len(row)) != exclude_index
        candidate_positions = np.flatnonzero(keep)
        order = candidate_positions[np.argsort(row[keep], kind="stable")]
    else:
        order = np.argsort(row, kind="stable")
    k = min(k, len(order))
    bottom = order[:k]
    top = order[-k:][::-1]
    return top, bottom


def _rank_topk_one_direction(
    text_sim: SimilarityMatrix,
    graph_sim: SimilarityMatrix,
    k: int,
) -> list[ExampleTuple]:
    """Emit rank-zipped (anchor, text_pos, text_neg, graph_pos, graph_neg) tuples for one anchor set.

    Only anchors present in **both** ``text_sim.row_ids`` and
    ``graph_sim.row_ids`` are used — the two views can disagree on which nodes
    are eligible (e.g. a node present in the graph store but missing from the
    text store), so each anchor is looked up by id in each view independently
    rather than assuming identical row ordering/length.

    ``graph_sim`` is a within-graph (square, self-similarity-on-diagonal)
    matrix, so the anchor's own column is excluded from its top-k/bottom-k —
    otherwise the anchor's guaranteed rank-0 similarity to itself (1.0) would
    make it its own "positive" neighbor. ``text_sim`` is cross-graph and has
    no such diagonal, so no exclusion is needed there.
    """
    text_row_of = {nid: i for i, nid in enumerate(text_sim.row_ids)}
    graph_row_of = {nid: i for i, nid in enumerate(graph_sim.row_ids)}
    graph_col_of = {nid: i for i, nid in enumerate(graph_sim.col_ids)}
    anchors = [nid for nid in text_sim.row_ids if nid in graph_row_of]

    examples: list[ExampleTuple] = []
    for anchor_id in anchors:
        text_top, text_bottom = _topk_bottomk_indices(text_sim.values[text_row_of[anchor_id]], k)
        graph_top, graph_bottom = _topk_bottomk_indices(
            graph_sim.values[graph_row_of[anchor_id]], k, exclude_index=graph_col_of.get(anchor_id)
        )
        n = min(len(text_top), len(graph_top))
        for j in range(n):
            examples.append(
                (
                    anchor_id,
                    text_sim.col_ids[text_top[j]],
                    text_sim.col_ids[text_bottom[j]],
                    graph_sim.col_ids[graph_top[j]],
                    graph_sim.col_ids[graph_bottom[j]],
                )
            )
    return examples


@samplers.register("rank_topk")
def rank_topk_sampler(
    *,
    text_g1_g2: SimilarityMatrix,
    graph_g1: SimilarityMatrix,
    graph_g2: SimilarityMatrix,
    config: dict,
    seed: int,
    alignments: list[dict] | None = None,
) -> list[ExampleTuple]:
    """Top-k/bottom-k by argsort, pairs zipped by rank (legacy ``sample_10``).

    For each ``graph_1`` anchor: text positive/negative are the k most/least
    text-similar ``graph_2`` nodes; graph positive/negative are the k
    most/least structurally-similar **other** ``graph_1`` nodes by argsort
    rank — the anchor's own diagonal self-similarity (always 1.0) is excluded
    before ranking, so an anchor can never be paired with itself as a graph
    "positive" (a deviation from the legacy ``get_sample_10``, which did not
    exclude it; see the Sprint 1.5 execution log). Symmetric for ``graph_2``
    anchors.
    """
    k = config.get("k", 30)

    g1_examples = _rank_topk_one_direction(text_g1_g2, graph_g1, k)

    # graph_2 direction: text view is transposed (graph_2 anchors -> graph_1 columns).
    text_g2_g1 = SimilarityMatrix(
        row_ids=text_g1_g2.col_ids, col_ids=text_g1_g2.row_ids, values=text_g1_g2.values.T
    )
    g2_examples = _rank_topk_one_direction(text_g2_g1, graph_g2, k)

    examples = g1_examples + g2_examples
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def _percentile_pairs_row(
    row: np.ndarray, col_ids: list[str], top_pct: float, bottom_pct: float, *, exclude_index: int | None = None
) -> tuple[list[str], list[str]]:
    """Return (ids at/above the top percentile, ids at/below the bottom percentile), sorted by score.

    ``exclude_index`` drops one column before computing percentile thresholds
    and pools — used to remove a within-graph anchor's self-similarity (always
    1.0) so it can never fall into its own top-percentile pool.
    """
    candidate_idx = [i for i in range(len(row)) if i != exclude_index]
    if not candidate_idx:
        return [], []
    values = row[candidate_idx]
    top_threshold = np.percentile(values, 100 - top_pct * 100)
    bottom_threshold = np.percentile(values, bottom_pct * 100)
    top = sorted((i for i in candidate_idx if row[i] >= top_threshold), key=lambda idx: -row[idx])
    bottom = sorted((i for i in candidate_idx if row[i] <= bottom_threshold), key=lambda idx: row[idx])
    return [col_ids[i] for i in top], [col_ids[i] for i in bottom]


def _percentile_one_direction(
    text_sim: SimilarityMatrix,
    graph_sim: SimilarityMatrix,
    top_pct: float,
    bottom_pct: float,
    rng: random.Random,
    *,
    rank_zip_view: str,
) -> list[ExampleTuple]:
    """Emit percentile-thresholded tuples; one view zipped by rank, the other sampled uniformly.

    Only anchors present in both views are used, looked up by id independently
    in each (see ``_rank_topk_one_direction`` for why row order/length cannot
    be assumed to match). ``rank_zip_view`` selects which view is zipped by
    rank ("text" or "graph") and which is paired by a uniform-random index
    within its own percentile pool, matching the legacy ``get_sample_11``
    asymmetric pairing. As in ``rank_topk``, ``graph_sim``'s diagonal
    self-similarity is excluded so an anchor is never its own graph
    "positive"/"negative" (a deviation from the legacy script — see the
    Sprint 1.5 execution log).
    """
    text_row_of = {nid: i for i, nid in enumerate(text_sim.row_ids)}
    graph_row_of = {nid: i for i, nid in enumerate(graph_sim.row_ids)}
    graph_col_of = {nid: i for i, nid in enumerate(graph_sim.col_ids)}
    anchors = [nid for nid in text_sim.row_ids if nid in graph_row_of]

    examples: list[ExampleTuple] = []
    for anchor_id in anchors:
        text_top, text_bottom = _percentile_pairs_row(
            text_sim.values[text_row_of[anchor_id]], text_sim.col_ids, top_pct, bottom_pct
        )
        graph_top, graph_bottom = _percentile_pairs_row(
            graph_sim.values[graph_row_of[anchor_id]],
            graph_sim.col_ids,
            top_pct,
            bottom_pct,
            exclude_index=graph_col_of.get(anchor_id),
        )
        if not text_top or not text_bottom or not graph_top or not graph_bottom:
            continue

        if rank_zip_view == "text":
            n = min(len(text_top), len(text_bottom))
            for j in range(n):
                gi = rng.randrange(0, min(len(graph_top), len(graph_bottom)))
                examples.append((anchor_id, text_top[j], text_bottom[j], graph_top[gi], graph_bottom[gi]))
        else:
            n = min(len(graph_top), len(graph_bottom))
            for j in range(n):
                ti = rng.randrange(0, min(len(text_top), len(text_bottom)))
                examples.append((anchor_id, text_top[ti], text_bottom[ti], graph_top[j], graph_bottom[j]))
    return examples


@samplers.register("percentile")
def percentile_sampler(
    *,
    text_g1_g2: SimilarityMatrix,
    graph_g1: SimilarityMatrix,
    graph_g2: SimilarityMatrix,
    config: dict,
    seed: int,
    alignments: list[dict] | None = None,
) -> list[ExampleTuple]:
    """Percentile thresholds with randomized pairing (legacy ``sample_11``), seeded.

    Unlike the legacy script (unseeded ``random``, non-reproducible), the
    random index used to pair the non-rank-zipped view is drawn from a
    ``random.Random(seed)`` instance so output is reproducible under a fixed
    seed, per the Phase 1.5 test requirement.
    """
    top_pct = config.get("top_pct", 0.04)
    bottom_pct = config.get("bottom_pct", 0.04)
    rng = random.Random(seed)

    g1_examples = _percentile_one_direction(
        text_g1_g2,
        graph_g1,
        top_pct,
        bottom_pct,
        rng,
        rank_zip_view="text",
    )

    text_g2_g1 = SimilarityMatrix(
        row_ids=text_g1_g2.col_ids, col_ids=text_g1_g2.row_ids, values=text_g1_g2.values.T
    )
    g2_examples = _percentile_one_direction(
        text_g2_g1,
        graph_g2,
        top_pct,
        bottom_pct,
        rng,
        rank_zip_view="graph",
    )

    examples = g1_examples + g2_examples
    rng.shuffle(examples)
    return examples


@samplers.register("supervised")
def supervised_sampler(
    *,
    text_g1_g2: SimilarityMatrix,
    graph_g1: SimilarityMatrix,
    graph_g2: SimilarityMatrix,
    config: dict,
    seed: int,
    alignments: list[dict] | None = None,
) -> list[ExampleTuple]:
    """Positives from ``alignments.parquet``'s gold pairs; negatives sampled from the non-positive pool.

    For each gold ``(node_1_id, node_2_id)`` alignment pair (any split — the
    caller is responsible for keeping ``test``-split pairs out of what gets
    written as a training example; see ``training_data.pipeline``), the
    positive text pair is the alignment pair itself; the negative text pair
    reuses the anchor with a uniformly random *other* ``graph_2`` node drawn
    from ``text_g1_g2.col_ids`` (excluding the true positive). Graph
    positive/negative reuse the same random-negative construction within each
    anchor's own graph, since ``alignments.parquet`` carries no structural
    ground truth. No-ops (returns ``[]``) when ``alignments`` is empty.
    """
    if not alignments:
        return []

    rng = random.Random(seed)
    text_col_ids = text_g1_g2.col_ids
    text_col_set = set(text_col_ids)
    graph_g1_ids = graph_g1.col_ids
    graph_g2_ids = graph_g2.col_ids

    examples: list[ExampleTuple] = []
    for pair in alignments:
        anchor_id = pair["node_1_id"]
        pos_id = pair["node_2_id"]
        if anchor_id not in set(text_g1_g2.row_ids) or pos_id not in text_col_set:
            continue

        neg_candidates = [nid for nid in text_col_ids if nid != pos_id]
        if not neg_candidates:
            continue
        text_neg_id = neg_candidates[rng.randrange(0, len(neg_candidates))]

        graph_pos_candidates = [nid for nid in graph_g1_ids if nid != anchor_id]
        if not graph_pos_candidates:
            continue
        graph_pos_id = graph_pos_candidates[rng.randrange(0, len(graph_pos_candidates))]
        graph_neg_candidates = [nid for nid in graph_g1_ids if nid not in (anchor_id, graph_pos_id)]
        if not graph_neg_candidates:
            graph_neg_candidates = graph_pos_candidates
        graph_neg_id = graph_neg_candidates[rng.randrange(0, len(graph_neg_candidates))]

        examples.append((anchor_id, pos_id, text_neg_id, graph_pos_id, graph_neg_id))

    return examples
