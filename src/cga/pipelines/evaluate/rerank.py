"""Anchor-based second-stage re-ranker over a first stage's top-K (Sprint 6.2, design §8).

NCIT and DOID are disconnected graphs, so structure can be compared across them only through
known gold pairs (*anchors*). This module turns that into features for a small listwise linear
model that reorders the first stage's top-K candidates for each query:

- **scores**: cosines of the first-stage members and of SapBERT, the first-stage score, and its gap to
  the top-1 score;
- **anchor agreement**: parents / children of the query source whose anchor is a parent / child of the
  candidate, and the share of anchors within 2 hops of the source whose partner is within 2 hops of
  the candidate;
- **granularity mismatch**: a neighbour of the source is already anchored to the candidate, so the
  candidate is probably a parent or child of the true match, not the match;
- **exclusivity**: the candidate is already the train partner of a different source.

Anchors are gold **train** pairs only, so no val or test pair is ever an anchor (the query's own pair
included). The module is used by the ``gold_alignment_rerank`` evaluator and by
``scripts/srga_rerank_*.py``; it has no pipeline state of its own. Beyond the top-K the first-stage
order is kept, and a query whose gold target is outside the top-K is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cga.core.errors import ConfigError
from cga.pipelines.evaluate.metrics import mrr_via_argsort, top_k_by_similarity

SCORE_FEATURES = ["cos_sapbert", "cos_a11", "cos_a12", "score_first", "gap_top1"]
ANCHOR_FEATURES = ["parent_agree", "child_agree", "hop2_agree"]
GRANULARITY_FEATURES = ["granularity_mismatch"]
EXCLUSIVITY_FEATURES = ["exclusive"]
GROUP_FEATURES: dict[str, list[str]] = {
    "scores": SCORE_FEATURES,
    "anchor": ANCHOR_FEATURES,
    "granularity": GRANULARITY_FEATURES,
    "exclusivity": EXCLUSIVITY_FEATURES,
}
# Score-source names a caller must supply similarity rows for.
SCORE_SOURCES = ("sapbert", "a11", "a12", "first")


def feature_names(groups: Iterable[str]) -> list[str]:
    """Return the feature names of ``groups`` in the canonical order.

    Raises:
        ConfigError: On an unknown group.
    """
    groups = list(groups)
    unknown = [g for g in groups if g not in GROUP_FEATURES]
    if unknown:
        raise ConfigError(f"unknown feature group(s) {unknown}; known: {list(GROUP_FEATURES)}")
    return [f for g, fs in GROUP_FEATURES.items() if g in groups for f in fs]


# --------------------------------------------------------------------------------------- anchors / graphs


@dataclass
class Anchors:
    """Known gold pairs used as cross-graph structural evidence.

    Attributes:
        partners: graph_1 node -> set of its anchored graph_2 nodes.
        sources_of: graph_2 node -> set of graph_1 nodes anchored to it.
    """

    partners: dict[str, set[str]] = field(default_factory=dict)
    sources_of: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "Anchors":
        """Build from ``(graph_1 id, graph_2 id)`` pairs."""
        a = cls()
        for n, d in pairs:
            a.partners.setdefault(n, set()).add(d)
            a.sources_of.setdefault(d, set()).add(n)
        return a

    def pairs(self) -> list[tuple[str, str]]:
        """Return the anchor pairs, sorted."""
        return sorted((n, d) for n, ds in self.partners.items() for d in ds)


def build_anchors(
    alignment_rows: Iterable[dict[str, Any]],
    role_of: dict[str, str],
    split: str = "train",
    shuffle_seed: int | None = None,
) -> Anchors:
    """Build anchors from the gold pairs of ``split`` (train by default; never val or test).

    Args:
        alignment_rows: Rows of ``alignments.parquet`` (``node_1_id``, ``node_2_id``, ``split``).
        role_of: node id -> ``graph_role``; pairs whose endpoints are not (graph_1, graph_2) nodes are dropped.
        split: Which gold split supplies anchors.
        shuffle_seed: When given, the graph_2 partners are permuted among the anchors (a cyclic shift
            of a random order): the endpoints stay, the correspondences are destroyed. This is the
            A24 control.

    Returns:
        The anchors.
    """
    pairs = sorted(
        {
            (r["node_1_id"], r["node_2_id"])
            for r in alignment_rows
            if r["split"] == split
            and role_of.get(r["node_1_id"]) == "graph_1"
            and role_of.get(r["node_2_id"]) == "graph_2"
        }
    )
    if shuffle_seed is not None and len(pairs) > 1:
        order = np.random.default_rng(shuffle_seed).permutation(len(pairs))
        shuffled = list(pairs)
        for j, i in enumerate(order):
            shuffled[i] = (pairs[i][0], pairs[order[(j + 1) % len(pairs)]][1])
        pairs = shuffled
    return Anchors.from_pairs(pairs)


class OntologyGraph:
    """Directed parent / child lookup and cached undirected k-hop neighbourhoods.

    Edges are stored child -> parent (``src_id`` is the child), as the ingest adapters write them.
    """

    def __init__(self, edge_rows: Iterable[dict[str, Any]]) -> None:
        self.parents: dict[str, set[str]] = {}
        self.children: dict[str, set[str]] = {}
        self._adj: dict[str, set[str]] = {}
        for e in edge_rows:
            s, d = e["src_id"], e["dst_id"]
            if s == d:
                continue
            self.parents.setdefault(s, set()).add(d)
            self.children.setdefault(d, set()).add(s)
            self._adj.setdefault(s, set()).add(d)
            self._adj.setdefault(d, set()).add(s)
        self._within: dict[tuple[str, int], frozenset[str]] = {}

    def within(self, node: str, hops: int) -> frozenset[str]:
        """Return the nodes at undirected distance ``1..hops`` from ``node`` (``node`` excluded)."""
        key = (node, hops)
        if key not in self._within:
            seen, frontier = {node}, {node}
            for _ in range(hops):
                frontier = {m for n in frontier for m in self._adj.get(n, ())} - seen
                seen |= frontier
            self._within[key] = frozenset(seen - {node})
        return self._within[key]


# --------------------------------------------------------------------------------------- features


def anchor_features(source: str, cand: str, anchors: Anchors, g1: OntologyGraph, g2: OntologyGraph) -> np.ndarray:
    """Return ``[parent_agree, child_agree, hop2_agree, granularity_mismatch, exclusive]`` for one (source, candidate).

    Args:
        source: Query graph_1 node.
        cand: Candidate graph_2 node.
        anchors: Known pairs. The query's own pair is not an anchor (it is val or test); in addition
            the source itself is never counted as its own neighbour.
        g1: Graph of the source.
        g2: Graph of the candidate.
    """
    p1, c1 = g1.parents.get(source, set()), g1.children.get(source, set())
    p2, c2 = g2.parents.get(cand, set()), g2.children.get(cand, set())
    part = anchors.partners

    def agree(neigh_src: set[str], neigh_cand: set[str]) -> int:
        return sum(1 for a in neigh_src if a in part and part[a] & neigh_cand)

    parent_agree = agree(p1, p2)
    child_agree = agree(c1, c2)

    near_src = [a for a in g1.within(source, 2) if a in part]
    near_cand = g2.within(cand, 2)
    hop2 = (sum(1 for a in near_src if part[a] & near_cand) / len(near_src)) if near_src else 0.0

    granularity = float(any(cand in part.get(a, ()) for a in p1 | c1))
    exclusive = float(any(s != source for s in anchors.sources_of.get(cand, ())))
    return np.array([parent_agree, child_agree, hop2, granularity, exclusive], dtype=np.float64)


@dataclass
class QueryFeatures:
    """One query's first-stage rank and, for the top-K, its feature matrix.

    Attributes:
        source: Query graph_1 node.
        target: Gold graph_2 node.
        first_rank: 1-based rank of the gold target in the full first-stage list.
        top_ids: The first stage's top-K candidate ids, best first.
        X: ``(len(top_ids), n_features)`` unstandardised features.
        gold_pos: Index of the gold target in ``top_ids``, or -1 when it is outside the top-K.
    """

    source: str
    target: str
    first_rank: int
    top_ids: list[str]
    X: np.ndarray
    gold_pos: int


def build_query_features(
    queries: list[tuple[str, str]],
    cand_ids: list[list[str]],
    cand_idx: list[np.ndarray | None],
    sims: dict[str, np.ndarray],
    source_row: dict[str, int],
    anchors: Anchors,
    g1: OntologyGraph,
    g2: OntologyGraph,
    groups: Iterable[str],
    top_k: int,
) -> list[QueryFeatures]:
    """Rank each query with the first stage and extract features for its top-K.

    Args:
        queries: ``(source, gold target)`` per query.
        cand_ids: Candidate ids per query (the evaluator's candidate lists).
        cand_idx: ``cand_ids`` as indices into the pool (``None`` = the whole pool).
        sims: Score source name -> ``(n_sources, n_pool)`` cosine matrix; needs :data:`SCORE_SOURCES`.
        source_row: source id -> row of ``sims``.
        anchors, g1, g2: Anchors and the two ontology graphs.
        groups: Feature groups to extract (see :data:`GROUP_FEATURES`).
        top_k: ``K``.

    Returns:
        One :class:`QueryFeatures` per query, in order.
    """
    groups = list(groups)
    names = feature_names(groups)
    missing = [s for s in SCORE_SOURCES if s not in sims]
    if missing:
        raise ConfigError(f"build_query_features needs score sources {list(SCORE_SOURCES)}, missing {missing}")
    use_struct = [g for g in ("anchor", "granularity", "exclusivity") if g in groups]
    struct_cols = {"parent_agree": 0, "child_agree": 1, "hop2_agree": 2, "granularity_mismatch": 3, "exclusive": 4}
    out: list[QueryFeatures] = []
    for qi, (source, target) in enumerate(queries):
        ids = cand_ids[qi]
        rows = {
            k: (m[source_row[source]] if cand_idx[qi] is None else m[source_row[source]][cand_idx[qi]])
            for k, m in sims.items()
        }
        first = rows["first"]
        first_rank = int(round(1.0 / mrr_via_argsort(first, ids, {target})))
        top_ids = top_k_by_similarity(first, ids, top_k)
        pos_of = {c: i for i, c in enumerate(ids)}
        sel = np.array([pos_of[c] for c in top_ids])
        top1 = float(first.max())
        cols: dict[str, np.ndarray] = {
            "cos_sapbert": rows["sapbert"][sel],
            "cos_a11": rows["a11"][sel],
            "cos_a12": rows["a12"][sel],
            "score_first": first[sel],
            "gap_top1": top1 - first[sel],
        }
        if use_struct:
            struct = np.array([anchor_features(source, c, anchors, g1, g2) for c in top_ids])
            for name, j in struct_cols.items():
                cols[name] = struct[:, j]
        X = np.stack([cols[n] for n in names], axis=1).astype(np.float64)
        out.append(
            QueryFeatures(
                source=source,
                target=target,
                first_rank=first_rank,
                top_ids=top_ids,
                X=X,
                gold_pos=top_ids.index(target) if target in top_ids else -1,
            )
        )
    return out


# --------------------------------------------------------------------------------------- model


def standardise(feats: list[QueryFeatures]) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-feature mean and std over every candidate of ``feats`` (std 0 -> 1)."""
    allx = np.concatenate([q.X for q in feats], axis=0)
    mean, std = allx.mean(axis=0), allx.std(axis=0)
    return mean, np.where(std < 1e-12, 1.0, std)


def fit_listwise(
    feats: list[QueryFeatures],
    top_k: int,
    l2: float,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Fit ``s_i = w . f_i`` by listwise softmax cross-entropy over each query's top-K.

    ``L = -mean_q log softmax(s)[gold] + l2 * ||w||^2`` over queries whose gold target is in the top-K
    (others carry no signal and are skipped). Features are standardised over the fitting queries.

    Args:
        feats: Queries with features computed at a top-K of at least ``top_k``.
        top_k: Candidates per query to use (the first ``top_k`` columns of each ``X``).
        l2: Ridge strength.
        max_iter: L-BFGS iterations.

    Returns:
        ``{"weights", "mean", "std", "top_k", "l2", "n_fit_queries"}`` as plain lists / numbers.
        With no usable query the weights are zero (the first-stage order is then kept).
    """
    import torch

    use = [q for q in feats if 0 <= q.gold_pos < top_k]
    sliced = [QueryFeatures(q.source, q.target, q.first_rank, q.top_ids[:top_k], q.X[:top_k], q.gold_pos) for q in feats]
    mean, std = standardise(sliced)
    n_feat = feats[0].X.shape[1] if feats else 0
    w = np.zeros(n_feat)
    if use:
        width = max(len(q.top_ids[:top_k]) for q in use)
        X = np.zeros((len(use), width, n_feat))
        mask = np.zeros((len(use), width), dtype=bool)
        gold = np.array([q.gold_pos for q in use])
        for i, q in enumerate(use):
            m = min(top_k, len(q.top_ids))
            X[i, :m] = (q.X[:m] - mean) / std
            mask[i, :m] = True
        Xt, mt, gt = torch.tensor(X), torch.tensor(mask), torch.tensor(gold)
        wt = torch.zeros(n_feat, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([wt], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            s = (Xt @ wt).masked_fill(~mt, float("-inf"))
            loss = -torch.log_softmax(s, dim=1)[torch.arange(len(gt)), gt].mean() + l2 * (wt**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        w = wt.detach().numpy().copy()
    return {
        "weights": w.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "top_k": int(top_k),
        "l2": float(l2),
        "n_fit_queries": len(use),
    }


def gold_rank_after_rerank(q: QueryFeatures, model: dict[str, Any]) -> int:
    """Return the 1-based rank of the gold target after re-ranking the top-K with ``model``.

    Candidates beyond the top-K keep the first-stage order after the re-ranked block, and a query
    whose gold target is outside the top-K keeps its first-stage rank. Ties inside the top-K keep the
    first-stage order.
    """
    k = min(model["top_k"], len(q.top_ids))
    if q.gold_pos < 0 or q.gold_pos >= k:
        return q.first_rank
    x = (q.X[:k] - np.asarray(model["mean"])) / np.asarray(model["std"])
    s = x @ np.asarray(model["weights"])
    g = s[q.gold_pos]
    return int(1 + np.sum(s > g) + np.sum(s[: q.gold_pos] == g))


def reranked_top_ids(q: QueryFeatures, model: dict[str, Any]) -> list[str]:
    """Return the first-stage top-K ids reordered by ``model`` (stable on ties)."""
    k = min(model["top_k"], len(q.top_ids))
    x = (q.X[:k] - np.asarray(model["mean"])) / np.asarray(model["std"])
    order = np.argsort(-(x @ np.asarray(model["weights"])), kind="stable")
    return [q.top_ids[i] for i in order] + q.top_ids[k:]


def save_reranker(path: Path, model: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write ``model`` plus ``meta`` (feature names, groups, first-stage ids, ...) as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, **model}, indent=2), encoding="utf-8")


def load_reranker(path: str | Path) -> dict[str, Any]:
    """Read a re-ranker JSON written by :func:`save_reranker`.

    Raises:
        ConfigError: If the file is missing or lacks the weight fields.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"re-ranker file not found: {p}")
    model = json.loads(p.read_text(encoding="utf-8"))
    for key in ("weights", "mean", "std", "top_k", "feature_names", "groups"):
        if key not in model:
            raise ConfigError(f"re-ranker file {p} lacks '{key}'")
    return model


# --------------------------------------------------------------------------------------- query building


def featurize_split(
    matrices: dict[str, Any],
    role_of: dict[str, str],
    alignment_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    split: str,
    mode: str,
    groups: Iterable[str],
    top_k: int,
    anchors: Anchors | None = None,
) -> list[QueryFeatures]:
    """Build the evaluator's queries for ``split`` / ``mode`` and extract re-ranker features.

    The query set and candidate lists are exactly those of ``gold_alignment`` (its own
    ``_build_queries``), so ranks agree with the evaluator's.

    Args:
        matrices: :data:`SCORE_SOURCES` name -> ``AlignedMatrix``; ``first`` is the first stage.
        role_of: node id -> ``graph_role``.
        alignment_rows: ``alignments.parquet`` rows (all splits; train supplies the anchors).
        edge_rows: ``edges.parquet`` rows, child -> parent.
        split: Gold split to query.
        mode: ``global`` or ``candidates``.
        groups: Feature groups.
        top_k: ``K``.
        anchors: Override the anchors (the shuffled control); default = train pairs.

    Returns:
        One :class:`QueryFeatures` per query.

    Raises:
        ConfigError: If a score source is missing or no pair of ``split`` is eligible.
    """
    from cga.pipelines.evaluate.evaluators.gold_alignment import GoldAlignmentEvaluator, _normalize

    missing = [s for s in SCORE_SOURCES if s not in matrices]
    if missing:
        raise ConfigError(f"featurize_split needs matrices {list(SCORE_SOURCES)}, missing {missing}")
    shared = set.intersection(*(set(m.node_ids) for m in matrices.values()))
    targets_of: dict[str, set[str]] = {}
    for r in alignment_rows:
        targets_of.setdefault(r["node_1_id"], set()).add(r["node_2_id"])
    pool_ids = sorted(n for n in shared if role_of.get(n) == "graph_2")
    index_of = {n: i for i, n in enumerate(pool_ids)}
    split_rows = [r for r in alignment_rows if r["split"] == split]
    pairs = sorted(
        {
            (r["node_1_id"], r["node_2_id"])
            for r in split_rows
            if r["node_1_id"] in shared
            and r["node_2_id"] in shared
            and role_of.get(r["node_1_id"]) == "graph_1"
            and role_of.get(r["node_2_id"]) == "graph_2"
        }
    )
    if not pairs:
        raise ConfigError(f"no eligible '{split}' pairs for the re-ranker")
    pool_by_pair = {(r["node_1_id"], r["node_2_id"]): r.get("candidates") for r in split_rows}
    queries, cand_ids, cand_idx, _ = GoldAlignmentEvaluator._build_queries(
        mode, pairs, pool_by_pair, targets_of, pool_ids, index_of, shared
    )
    sources = sorted({q[0] for q in queries})
    source_row = {s: i for i, s in enumerate(sources)}
    sims = {
        name: _normalize(m.select(sources).vectors) @ _normalize(m.select(pool_ids).vectors).T
        for name, m in matrices.items()
    }
    if anchors is None:
        anchors = build_anchors(alignment_rows, role_of)
    g1 = OntologyGraph(e for e in edge_rows if e["graph_role"] == "graph_1")
    g2 = OntologyGraph(e for e in edge_rows if e["graph_role"] == "graph_2")
    return build_query_features(queries, cand_ids, cand_idx, sims, source_row, anchors, g1, g2, groups, top_k)


# --------------------------------------------------------------------------------------- cross-validation

K_GRID = (10, 20, 50)
L2_GRID = (0.01, 0.1, 1.0)


def grouped_folds(feats: list[QueryFeatures], n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Split query indices into ``n_folds`` folds so every source node falls in exactly one fold."""
    sources = sorted({q.source for q in feats})
    order = np.random.default_rng(seed).permutation(len(sources))
    fold_of = {sources[i]: j % n_folds for j, i in enumerate(order)}
    qf = np.array([fold_of[q.source] for q in feats])
    return [np.flatnonzero(qf == f) for f in range(n_folds)]


def select_config(
    feats: list[QueryFeatures],
    n_folds: int = 5,
    k_grid: Iterable[int] = K_GRID,
    l2_grid: Iterable[float] = L2_GRID,
    seed: int = 0,
) -> tuple[int, float, dict[tuple[int, float], float]]:
    """Choose ``(K, l2)`` by grouped cross-validated MRR over ``feats``.

    Ties go to the smaller K, then the larger ``l2`` (the simpler model).

    Returns:
        ``(K, l2, {(K, l2): cv_mrr})``.
    """
    folds = grouped_folds(feats, n_folds, seed)
    scores: dict[tuple[int, float], float] = {}
    for k in k_grid:
        for l2 in l2_grid:
            rr = np.zeros(len(feats))
            for f, test_idx in enumerate(folds):
                train = [feats[i] for g, idx in enumerate(folds) if g != f for i in idx]
                model = fit_listwise(train, k, l2)
                rr[test_idx] = [1.0 / gold_rank_after_rerank(feats[i], model) for i in test_idx]
            scores[(k, l2)] = float(rr.mean())
    best = max(scores, key=lambda c: (round(scores[c], 12), -c[0], c[1]))
    return best[0], best[1], scores


def nested_cv_ranks(
    feats: list[QueryFeatures],
    n_outer: int = 5,
    n_inner: int = 4,
    seed: int = 0,
    k_grid: Iterable[int] = K_GRID,
    l2_grid: Iterable[float] = L2_GRID,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Out-of-fold gold ranks with ``(K, l2)`` re-selected inside every outer training fold.

    Returns:
        ``(ranks, chosen)``: the 1-based gold rank per query when its fold was held out, and the
        ``(K, l2)`` chosen in each outer fold.
    """
    folds = grouped_folds(feats, n_outer, seed)
    ranks = np.zeros(len(feats), dtype=int)
    chosen = []
    for f, test_idx in enumerate(folds):
        train = [feats[i] for g, idx in enumerate(folds) if g != f for i in idx]
        k, l2, _ = select_config(train, n_inner, k_grid, l2_grid, seed)
        model = fit_listwise(train, k, l2)
        ranks[test_idx] = [gold_rank_after_rerank(feats[i], model) for i in test_idx]
        chosen.append((k, l2))
    return ranks, chosen
