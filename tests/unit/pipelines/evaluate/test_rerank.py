"""Unit tests for the anchor-based re-ranker (Sprint 6.2) and the ``gold_alignment_rerank`` evaluator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cga.core.errors import ConfigError
from cga.data.aligned import AlignedMatrix
from cga.pipelines.evaluate.evaluators.gold_alignment import GoldAlignmentEvaluator
from cga.pipelines.evaluate.evaluators.gold_alignment_rerank import GoldAlignmentRerankEvaluator
from cga.pipelines.evaluate.rerank import (
    Anchors,
    OntologyGraph,
    QueryFeatures,
    anchor_features,
    build_anchors,
    feature_names,
    fit_listwise,
    gold_rank_after_rerank,
    grouped_folds,
    nested_cv_ranks,
    reranked_top_ids,
    select_config,
)
from cga.pipelines.evaluate.rows import MetricsDigest

ROLE = {n: "graph_1" for n in ["n", "a", "c", "z"]} | {n: "graph_2" for n in ["d", "x", "y", "e", "x2", "w"]}
# graph_1: a is the parent of n, c its child. graph_2: x is the parent of d, y its child, x2 the parent of e.
G1 = [{"src_id": "n", "dst_id": "a"}, {"src_id": "c", "dst_id": "n"}]
G2 = [{"src_id": "d", "dst_id": "x"}, {"src_id": "y", "dst_id": "d"}, {"src_id": "e", "dst_id": "x2"}]


def _align(n1: str, n2: str, split: str) -> dict:
    return {"node_1_id": n1, "node_2_id": n2, "split": split, "candidates": None}


ALIGN = [
    _align("a", "x", "train"),
    _align("c", "y", "train"),
    _align("a", "e", "train"),  # a neighbour of n already matched to e
    _align("z", "w", "train"),
    _align("n", "d", "val"),  # the query itself
    _align("z", "d", "test"),
]


def _graphs():
    return OntologyGraph(G1), OntologyGraph(G2)


def test_anchors_use_train_pairs_only_and_never_the_query_pair():
    anchors = build_anchors(ALIGN, ROLE)
    assert anchors.pairs() == [("a", "e"), ("a", "x"), ("c", "y"), ("z", "w")]
    assert ("n", "d") not in anchors.pairs()  # val
    assert ("z", "d") not in anchors.pairs()  # test
    assert "n" not in anchors.partners


def test_anchor_features_on_hand_built_fixture():
    anchors = build_anchors(ALIGN, ROLE)
    g1, g2 = _graphs()
    # [parent_agree, child_agree, hop2_agree, granularity_mismatch, exclusive]
    np.testing.assert_allclose(anchor_features("n", "d", anchors, g1, g2), [1, 1, 1.0, 0, 0])
    # e: a is already matched to e (granularity + exclusivity); parents x2 vs a's partners disagree.
    np.testing.assert_allclose(anchor_features("n", "e", anchors, g1, g2), [0, 0, 0.0, 1, 1])
    np.testing.assert_allclose(anchor_features("n", "w", anchors, g1, g2), [0, 0, 0.0, 0, 1])  # w is z's partner


def test_exclusivity_ignores_the_source_own_partner():
    anchors = Anchors.from_pairs([("n", "d")])
    g1, g2 = _graphs()
    assert anchor_features("n", "d", anchors, g1, g2)[4] == 0.0  # only the source itself is anchored to d


def test_shuffled_anchors_keep_endpoints_and_break_correspondence():
    rows = [_align(f"s{i}", f"t{i}", "train") for i in range(8)]
    role = {f"s{i}": "graph_1" for i in range(8)} | {f"t{i}": "graph_2" for i in range(8)}
    true = build_anchors(rows, role)
    shuf = build_anchors(rows, role, shuffle_seed=3)
    assert set(shuf.partners) == set(true.partners)
    assert set(shuf.sources_of) == set(true.sources_of)
    assert not set(shuf.pairs()) & set(true.pairs())
    assert build_anchors(rows, role, shuffle_seed=3).pairs() == shuf.pairs()  # deterministic


def test_feature_names_groups_and_unknown_group():
    assert len(feature_names(["scores", "anchor", "granularity", "exclusivity"])) == 10
    assert feature_names(["scores"]) == ["cos_sapbert", "cos_a11", "cos_a12", "score_first", "gap_top1"]
    with pytest.raises(ConfigError):
        feature_names(["bogus"])


def _qf(scores: list[float], gold_pos: int, first_rank: int | None = None) -> QueryFeatures:
    ids = [f"c{i}" for i in range(len(scores))]
    return QueryFeatures("s", ids[gold_pos] if gold_pos >= 0 else "far", first_rank or gold_pos + 1, ids,
                         np.array(scores, dtype=float).reshape(-1, 1), gold_pos)


def _model(weight: float, k: int) -> dict:
    return {"weights": [weight], "mean": [0.0], "std": [1.0], "top_k": k}


def test_rerank_reorders_inside_top_k_only():
    # Feature increases down the list; weight +1 reverses the top 3 and leaves ranks 4-5 alone.
    q = _qf([0.0, 1.0, 2.0, 3.0, 4.0], gold_pos=2)
    assert gold_rank_after_rerank(q, _model(1.0, 3)) == 1
    assert gold_rank_after_rerank(q, _model(-1.0, 3)) == 3
    assert reranked_top_ids(q, _model(1.0, 3)) == ["c2", "c1", "c0", "c3", "c4"]


def test_gold_outside_top_k_is_unchanged_and_ties_keep_first_stage_order():
    far = _qf([0.0, 1.0, 2.0, 3.0, 4.0], gold_pos=-1, first_rank=40)
    assert gold_rank_after_rerank(far, _model(5.0, 3)) == 40
    beyond = _qf([0.0, 1.0, 2.0, 3.0, 4.0], gold_pos=4)  # inside the list but beyond K=3
    assert gold_rank_after_rerank(beyond, _model(5.0, 3)) == 5
    tied = _qf([1.0, 1.0, 1.0], gold_pos=1)
    assert gold_rank_after_rerank(tied, _model(1.0, 3)) == 2


def _planted(n_queries: int, w_true: np.ndarray, seed: int = 0) -> list[QueryFeatures]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_queries):
        X = rng.normal(size=(10, len(w_true)))
        gold = int(np.argmax(X @ w_true + rng.gumbel(size=10)))
        out.append(QueryFeatures(f"s{i}", f"c{gold}", gold + 1, [f"c{j}" for j in range(10)], X, gold))
    return out


def test_listwise_fit_recovers_planted_weights_and_apply_reorders():
    w_true = np.array([2.0, -1.0, 0.0])
    feats = _planted(3000, w_true)
    model = fit_listwise(feats, top_k=10, l2=1e-6)
    w = np.array(model["weights"]) * 1.0
    # Standardised features here have ~unit std, so the fitted weights approximate the planted ones.
    assert np.corrcoef(w, w_true)[0, 1] > 0.99
    assert abs(w[2]) < 0.2
    mrr_first = np.mean([1.0 / q.first_rank for q in feats])
    mrr_fit = np.mean([1.0 / gold_rank_after_rerank(q, model) for q in feats])
    assert mrr_fit > mrr_first
    # The apply step orders by the model's score.
    q = feats[0]
    x = (q.X - np.array(model["mean"])) / np.array(model["std"])
    order = np.argsort(-(x @ w), kind="stable")
    assert reranked_top_ids(q, model) == [q.top_ids[i] for i in order]


def test_fit_with_no_usable_query_keeps_first_stage_order():
    feats = [_qf([0.0, 1.0, 2.0], gold_pos=-1, first_rank=9)]
    model = fit_listwise(feats, top_k=3, l2=0.1)
    assert model["weights"] == [0.0] and model["n_fit_queries"] == 0


def test_grouped_folds_keep_each_source_in_one_fold_and_cv_runs():
    feats = _planted(60, np.array([1.0, 0.5]))
    feats += [QueryFeatures(f.source, f.target, f.first_rank, f.top_ids, f.X, f.gold_pos) for f in feats[:10]]  # repeated sources
    folds = grouped_folds(feats, 5)
    assert sorted(i for f in folds for i in f) == list(range(len(feats)))
    for src in {f.source for f in feats}:
        assert len({k for k, idx in enumerate(folds) for i in idx if feats[i].source == src}) == 1
    k, l2, scores = select_config(feats, n_folds=3, k_grid=(5, 10), l2_grid=(0.1, 1.0))
    assert (k, l2) in scores and len(scores) == 4
    ranks, chosen = nested_cv_ranks(feats, n_outer=3, n_inner=2, k_grid=(5, 10), l2_grid=(0.1,))
    assert ranks.min() >= 1 and len(chosen) == 3


# --- evaluator --------------------------------------------------------------------------------------------


class _Table:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def to_pylist(self) -> list[dict]:
        return self._rows


class _Res:
    """Minimal EvalResources stand-in with 12 sources, 20 targets and four score sources."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        s_ids = [f"s{i}" for i in range(12)]
        t_ids = [f"t{i}" for i in range(20)]
        self.graph_role_by_node = {n: "graph_1" for n in s_ids} | {n: "graph_2" for n in t_ids}
        base = rng.normal(size=(32, 16))
        self.methods = {}
        for name, noise in [("sapbert", 1.0), ("a11", 0.5), ("a12", 0.5), ("first", 0.3)]:
            v = base + noise * rng.normal(size=base.shape)
            self.methods[name] = (AlignedMatrix(vectors=v.astype(np.float32), node_ids=s_ids + t_ids), f"run-{name}")
        # Source i is aligned to target i; 0-5 train, 6-8 val, 9-11 test.
        splits = ["train"] * 6 + ["val"] * 3 + ["test"] * 3
        rows = [_align(f"s{i}", f"t{i}", splits[i]) for i in range(12)]
        self.alignments = _Table(rows)
        self.edges = [{"src_id": f"s{i}", "dst_id": f"s{i + 1}", "graph_role": "graph_1"} for i in range(11)] + [
            {"src_id": f"t{i}", "dst_id": f"t{i + 1}", "graph_role": "graph_2"} for i in range(19)
        ]
        self.method_specs = {}
        self.digest = MetricsDigest()


def _write_model(tmp_path, weights, groups=("scores", "anchor", "granularity", "exclusivity"), k=8, shuffle=None):
    names = feature_names(groups)
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {"weights": weights, "mean": [0.0] * len(names), "std": [1.0] * len(names), "top_k": k,
             "feature_names": names, "groups": list(groups), "shuffle_seed": shuffle}
        )
    )
    return str(path)


def _params(path: str, **extra) -> dict:
    return {"name": "AX", "first_stage": "first", "score_methods": {"sapbert": "sapbert", "a11": "a11", "a12": "a12"},
            "reranker": path, "ks": [1, 5, 10], "split": "test", "modes": ["global"], **extra}


def _by_metric(rows) -> dict:
    return {(r.metric_name, r.param_value): r.metric_value for r in rows}


def test_zero_weight_reranker_equals_the_first_stage_gold_alignment(tmp_path):
    res = _Res()
    path = _write_model(tmp_path, [0.0] * 10)
    rows = GoldAlignmentRerankEvaluator().evaluate(res, "rr", _params(path))
    only_first = _Res()
    only_first.methods = {"first": only_first.methods["first"]}
    only_first.method_specs = {"first": type("S", (), {"label": "first", "kind": "gcl_run"})()}
    base = GoldAlignmentEvaluator().evaluate(only_first, "ga", {"ks": [1, 5, 10], "split": "test", "modes": ["global"]})
    assert _by_metric(rows) == pytest.approx(_by_metric(base))
    assert all(r.method_id == "AX" and r.evaluation_kind == "gold_alignment_rerank" for r in rows)
    assert rows[0].method_run_id == "run-first+rerank:AX"


def test_rerank_evaluator_runs_with_weights_and_shuffled_anchors(tmp_path):
    res = _Res()
    w = [0.1, 0.1, 0.1, 1.0, 0.0, 0.5, 0.5, 0.5, -1.0, -1.0]
    rows = GoldAlignmentRerankEvaluator().evaluate(res, "rr", _params(_write_model(tmp_path, w, shuffle=1)))
    vals = _by_metric(rows)
    assert 0.0 < vals[("mrr", None)] <= 1.0
    assert vals[("hits_at_k", 1)] <= vals[("hits_at_k", 5)] <= vals[("hits_at_k", 10)]


def test_rerank_evaluator_rejects_bad_params(tmp_path):
    res = _Res()
    path = _write_model(tmp_path, [0.0] * 10)
    with pytest.raises(ConfigError):
        GoldAlignmentRerankEvaluator().evaluate(res, "rr", _params(path, first_stage="nope"))
    with pytest.raises(ConfigError):
        GoldAlignmentRerankEvaluator().evaluate(res, "rr", {**_params(path), "ks": []})
    with pytest.raises(ConfigError):
        GoldAlignmentRerankEvaluator().evaluate(res, "rr", _params(str(tmp_path / "missing.json")))
