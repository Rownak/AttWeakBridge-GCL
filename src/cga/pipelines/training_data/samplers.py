"""Registered samplers emitting anchor/positive/negative example rows (target_architecture.md §5.3 step 3).

Every sampler here only ever sees the non-``held_out`` similarity matrices
built by :mod:`cga.pipelines.training_data.similarity`, so a held-out
(alignment-test) node can never be drawn as an anchor, positive, or negative —
the leakage-prevention property is structural, not a filter applied after the
fact. ``supervised`` additionally receives the full gold alignment set (all
splits) so it can keep every gold target, train or test, out of its negatives.

Each sampler returns a list of ``(anchor_id, text_pos_id, text_neg_id,
graph_pos_id, graph_neg_id)`` tuples; the pipeline attaches ``example_id``,
``sampler``, and ``split``.
"""

from __future__ import annotations

import random
from collections.abc import Collection
from typing import Protocol

import numpy as np

from cga.core.errors import ConfigError
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


def _topk_bottomk_indices(
    row: np.ndarray,
    k: int,
    *,
    exclude_index: int | None = None,
    exclude_indices: Collection[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (most-similar-k, least-similar-k) column indices for one similarity row, by rank.

    Args:
        row: One row of a similarity matrix.
        k: Number of indices to return from each end (capped at the number of
            candidate columns).
        exclude_index: One column dropped before ranking — used to remove a
            within-graph anchor's self-similarity (always 1.0, always rank-0)
            from its own top-k, so an anchor is never paired with itself as a
            "positive".
        exclude_indices: Further columns dropped before ranking — used by the
            ``supervised`` sampler to keep an anchor's gold targets out of its
            text negatives. Combined with ``exclude_index``.

    Returns:
        ``(top, bottom)``: ``top`` in descending similarity, ``bottom`` in
        ascending similarity; ties keep column order (stable sort).
    """
    excluded = set(exclude_indices or ())
    if exclude_index is not None:
        excluded.add(exclude_index)
    if excluded:
        keep = np.ones(len(row), dtype=bool)
        keep[[i for i in excluded if 0 <= i < len(row)]] = False
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


def supervised_positives(
    alignments: list[dict], text_g1_g2: SimilarityMatrix, graph_g1: SimilarityMatrix
) -> tuple[list[tuple[str, str]], int]:
    """Return the usable gold train pairs and the number of train rows skipped.

    A ``train``-split row ``(node_1_id, node_2_id)`` is usable when the NCIT
    anchor is a row of both ``text_g1_g2`` and ``graph_g1`` and the target is a
    column of ``text_g1_g2``, i.e. neither endpoint is held out or outside the
    embedding universe. Duplicate pairs are kept once, in first-seen order, so
    the same pair never yields duplicate ``example_id``s. Shared by the sampler
    and ``training_data``'s ``stats.json`` so both count the same pairs.

    Args:
        alignments: Gold alignment rows (``node_1_id``, ``node_2_id``, ``split``).
        text_g1_g2: Cross-graph text similarity over non-held-out nodes.
        graph_g1: Within-``graph_1`` structural similarity over non-held-out nodes.

    Returns:
        ``(pairs, n_skipped)`` where ``n_skipped`` counts ``train`` rows that
        were not usable (duplicates are not counted as skipped).
    """
    text_rows = set(text_g1_g2.row_ids)
    text_cols = set(text_g1_g2.col_ids)
    graph_rows = set(graph_g1.row_ids)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    n_skipped = 0
    for row in alignments:
        if row.get("split") != "train":
            continue
        pair = (row["node_1_id"], row["node_2_id"])
        if pair[0] not in text_rows or pair[0] not in graph_rows or pair[1] not in text_cols:
            n_skipped += 1
            continue
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs, n_skipped


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
    """Gold ``train`` pairs as text positives; bottom-k or random text negatives, rank-based graph pos/neg.

    ``alignments`` must be the **full** gold set (every split): positives are
    taken from ``train`` rows only (see :func:`supervised_positives`), while
    the gold targets of **every** split are excluded from negatives. For each
    positive ``(a, p)``, with ``n = negatives_per_positive`` (default 5):

    - ``text_neg_id``: ``n`` ``graph_2`` columns of ``a``'s ``text_g1_g2`` row,
      after dropping all of ``a``'s gold targets (held-out targets are already
      absent from the matrix). ``config["text_negatives"]`` picks how:
      ``"bottomk"`` (default) takes the ``n`` least text-similar columns;
      ``"random"`` draws ``n`` distinct columns uniformly with the seeded RNG
      (all eligible columns if fewer than ``n``).
    - ``graph_pos_id``/``graph_neg_id``: the j-th most/least structurally
      similar other ``graph_1`` node of ``a`` (self excluded), the same rule
      as ``rank_topk``, so the graph-view loss sees a real structural signal.

    Negative j is zipped with structural rank j, giving one example row per
    negative. Only NCIT (``graph_1``) anchors are used because the gold files
    are NCIT->DOID. ``bottomk`` is deterministic and ``seed`` only drives the
    final shuffle; ``random`` also draws its negatives from the seed. No-ops
    (returns ``[]``) when ``alignments`` is empty.

    Raises:
        ConfigError: If ``text_negatives`` is not ``bottomk`` or ``random``.
    """
    mode = config.get("text_negatives", "bottomk")
    if mode not in ("bottomk", "random"):
        raise ConfigError(f"Unknown supervised text_negatives '{mode}'; expected 'bottomk' or 'random'")
    if not alignments:
        return []

    n = config.get("negatives_per_positive", 5)
    # One RNG for negatives and the final shuffle. bottomk never draws from it
    # before the shuffle, so its output is identical to before this option existed.
    rng = random.Random(seed)
    n_text_cols = len(text_g1_g2.col_ids)
    positives, _ = supervised_positives(alignments, text_g1_g2, graph_g1)

    gold_targets: dict[str, set[str]] = {}
    for row in alignments:
        gold_targets.setdefault(row["node_1_id"], set()).add(row["node_2_id"])

    text_row_of = {nid: i for i, nid in enumerate(text_g1_g2.row_ids)}
    text_col_of = {nid: i for i, nid in enumerate(text_g1_g2.col_ids)}
    graph_row_of = {nid: i for i, nid in enumerate(graph_g1.row_ids)}
    graph_col_of = {nid: i for i, nid in enumerate(graph_g1.col_ids)}

    examples: list[ExampleTuple] = []
    for anchor_id, pos_id in positives:
        gold_cols = [text_col_of[t] for t in gold_targets[anchor_id] if t in text_col_of]
        if mode == "random":
            gold_set = set(gold_cols)
            eligible = [i for i in range(n_text_cols) if i not in gold_set]
            text_neg_cols = rng.sample(eligible, min(n, len(eligible)))
        else:
            _, text_neg_cols = _topk_bottomk_indices(
                text_g1_g2.values[text_row_of[anchor_id]], n, exclude_indices=gold_cols
            )
        graph_top, graph_bottom = _topk_bottomk_indices(
            graph_g1.values[graph_row_of[anchor_id]], n, exclude_index=graph_col_of.get(anchor_id)
        )
        for j in range(min(len(text_neg_cols), len(graph_top))):
            examples.append(
                (
                    anchor_id,
                    pos_id,
                    text_g1_g2.col_ids[text_neg_cols[j]],
                    graph_g1.col_ids[graph_top[j]],
                    graph_g1.col_ids[graph_bottom[j]],
                )
            )

    rng.shuffle(examples)
    return examples
