"""``gold_alignment`` evaluator: cross-graph retrieval of held-out gold pairs (Sprint 4).

Asks *given a graph_1 node, does the method rank its gold graph_2 partner
first*, for every pair of one gold alignment split (``test`` by default), and
reports Hits@K and MRR per method. It is the supervised counterpart of
``neighborhood``: the relevant set is a single gold target, not a view.

Eligibility mirrors ``neighborhood``: a pair is a query only when **both**
endpoints are present in **every** method, so all methods share one query set
and one candidate pool and their numbers are comparable.

Two retrieval modes, each emitted as its own ``view_id``:

``global``
    Candidates are every ``graph_2`` node in the method intersection, minus
    the *other* gold targets of the query's source (any split, or only
    ``sibling_filter_splits``). Some sources
    have several gold targets (1:n); without this filter a correct sibling
    ranked above the target would count as an error.
``candidates``
    Candidates are the pair's pre-sampled pool (``test.cands.tsv``) restricted
    to the method intersection, with the target always included. Pairs
    without a pool are skipped. This is the OAEI Bio-ML ranking protocol.

Ranking reuses :mod:`cga.pipelines.evaluate.metrics`, so ties break by node id
and results never depend on argsort's unstable order.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from cga.core.errors import ConfigError
from cga.pipelines.evaluate.evaluators import evaluators
from cga.pipelines.evaluate.metrics import hit_at_k, mrr_via_argsort, top_k_by_similarity
from cga.pipelines.evaluate.rows import MetricRow

_MODES = ("global", "candidates")
_SOURCE_ROLE = "graph_1"
_TARGET_ROLE = "graph_2"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def _query_set_sha256(pairs: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{n1}|{n2}" for n1, n2 in sorted(pairs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _alignment_rows(alignments: Any) -> list[dict[str, Any]]:
    """Return the alignments table as a list of row dicts (empty when absent)."""
    if alignments is None:
        return []
    return alignments.to_pylist()


@evaluators.register("gold_alignment")
class GoldAlignmentEvaluator:
    """Scores every configured method by Hits@K / MRR on one gold alignment split.

    Params:
        ks: Hits@K cutoffs, e.g. ``[1, 5, 10]``.
        split: Gold alignment split to query (default ``"test"``).
        modes: Subset of ``["global", "candidates"]`` (default both).
        sibling_filter_splits: Splits whose gold pairs feed the ``global`` mode's "other gold
            targets" filter (default: all splits).
    """

    def evaluate(self, res: Any, evaluation_id: str, params: dict[str, Any]) -> list[MetricRow]:
        """Emit ``hits_at_k`` (one row per k) and ``mrr`` rows per method and mode.

        Args:
            res: ``EvalResources``; uses ``methods``, ``method_specs``,
                ``graph_role_by_node``, ``alignments`` and ``digest``.
            evaluation_id: Config-supplied instance name.
            params: ``ks``, optional ``split``, ``modes`` and ``sibling_filter_splits``.

        Returns:
            The scored rows.

        Raises:
            ConfigError: On an empty/invalid ``ks`` or ``modes``, or when the
                selected split has no gold pairs.
        """
        ks: list[int] = list(params.get("ks") or [])
        split: str = params.get("split", "test")
        modes: list[str] = list(params.get("modes", _MODES))
        if not ks or any(k < 1 for k in ks):
            raise ConfigError(f"gold_alignment evaluation '{evaluation_id}' needs a non-empty ks of ints >= 1")
        unknown = [m for m in modes if m not in _MODES]
        if unknown or not modes:
            raise ConfigError(
                f"gold_alignment evaluation '{evaluation_id}' has invalid modes {modes}; known: {list(_MODES)}"
            )

        all_rows = _alignment_rows(getattr(res, "alignments", None))
        split_rows = [r for r in all_rows if r["split"] == split]
        if not split_rows:
            raise ConfigError(
                f"gold_alignment evaluation '{evaluation_id}': no '{split}' alignment pairs "
                "(does the dataset config enable `alignments`?)"
            )

        method_ids = sorted(res.methods.keys())
        node_sets = [set(res.methods[mid][0].node_ids) for mid in method_ids]
        shared = set.intersection(*node_sets) if node_sets else set()

        # Default (None) = every split. The unsupervised protocol restricts it to the scored split so
        # gold train pairs are never read.
        sibling_splits = params.get("sibling_filter_splits")
        if sibling_splits is not None and (
            not isinstance(sibling_splits, list) or not all(isinstance(s, str) for s in sibling_splits)
        ):
            raise ConfigError(
                f"gold_alignment evaluation '{evaluation_id}': sibling_filter_splits must be a list of split names"
            )
        targets_of: dict[str, set[str]] = {}
        for r in all_rows:
            if sibling_splits is not None and r["split"] not in sibling_splits:
                continue
            targets_of.setdefault(r["node_1_id"], set()).add(r["node_2_id"])

        role_of = res.graph_role_by_node
        pool_ids = sorted(n for n in shared if role_of.get(n) == _TARGET_ROLE)
        index_of = {nid: i for i, nid in enumerate(pool_ids)}

        # One query = one gold pair; dedupe so a repeated row counts once.
        seen: set[tuple[str, str]] = set()
        pairs: list[tuple[str, str]] = []
        n_missing = 0
        for r in split_rows:
            pair = (r["node_1_id"], r["node_2_id"])
            if pair in seen:
                continue
            seen.add(pair)
            if (
                pair[0] not in shared
                or pair[1] not in shared
                or role_of.get(pair[0]) != _SOURCE_ROLE
                or role_of.get(pair[1]) != _TARGET_ROLE
            ):
                n_missing += 1
                continue
            pairs.append(pair)
        if not pairs:
            raise ConfigError(
                f"gold_alignment evaluation '{evaluation_id}': none of the '{split}' pairs has both "
                "endpoints in every method"
            )
        pairs.sort()
        pool_by_pair = {(r["node_1_id"], r["node_2_id"]): r.get("candidates") for r in split_rows}

        rows: list[MetricRow] = []
        for mode in modes:
            queries, cand_ids, cand_idx, n_no_pool = self._build_queries(
                mode, pairs, pool_by_pair, targets_of, pool_ids, index_of, shared
            )
            n_queries = len(queries)
            max_k = max(ks)

            for method_id in method_ids:
                matrix, method_run_id = res.methods[method_id]
                spec = res.method_specs[method_id]
                pool_vecs = _normalize(matrix.select(pool_ids).vectors)
                # A source with several gold targets is several queries but one vector.
                sources = sorted({q[0] for q in queries})
                source_row = {nid: i for i, nid in enumerate(sources)}
                src_vecs = _normalize(matrix.select(sources).vectors)
                sim = src_vecs @ pool_vecs.T

                per_k: dict[int, list[float]] = {k: [] for k in ks}
                rr: list[float] = []
                for qi, (_, target) in enumerate(queries):
                    sim_row = sim[source_row[queries[qi][0]]]
                    row = sim_row if cand_idx[qi] is None else sim_row[cand_idx[qi]]
                    ids = cand_ids[qi]
                    relevant = {target}
                    top = top_k_by_similarity(row, ids, max_k)
                    rr.append(mrr_via_argsort(row, ids, relevant))
                    for k in ks:
                        per_k[k].append(hit_at_k(top[:k], relevant, k))

                common = dict(
                    evaluation_id=evaluation_id,
                    evaluation_kind="gold_alignment",
                    method_id=method_id,
                    method_label=spec.label,
                    method_kind=spec.kind,
                    method_run_id=method_run_id,
                    graph_role=_SOURCE_ROLE,
                    view_id=mode,
                    n_queries=n_queries,
                )
                for k in ks:
                    rows.append(
                        MetricRow(
                            **common,
                            param_name="k",
                            param_value=k,
                            metric_name="hits_at_k",
                            metric_value=float(np.mean(per_k[k])),
                        )
                    )
                rows.append(MetricRow(**common, metric_name="mrr", metric_value=float(np.mean(rr))))

            res.digest.add_scope_provenance(
                evaluation_id=evaluation_id,
                view_id=mode,
                graph_role=_SOURCE_ROLE,
                query_set_sha256=_query_set_sha256(queries),
                split=split,
                n_eligible_queries=n_queries,
                n_dropped_missing_in_a_method=n_missing,
                n_skipped_no_candidate_pool=n_no_pool,
            )
        return rows

    @staticmethod
    def _build_queries(
        mode: str,
        pairs: list[tuple[str, str]],
        pool_by_pair: dict[tuple[str, str], Any],
        targets_of: dict[str, set[str]],
        pool_ids: list[str],
        index_of: dict[str, int],
        shared: set[str],
    ) -> tuple[list[tuple[str, str]], list[list[str]], list[np.ndarray | None], int]:
        """Build per-query candidate id lists (method-independent, so built once per mode).

        Returns:
            ``(queries, candidate_ids, candidate_idx, n_skipped_no_pool)``;
            ``candidate_idx[i]`` indexes ``pool_ids`` (``None`` means all of it).
        """
        queries: list[tuple[str, str]] = []
        cand_ids: list[list[str]] = []
        cand_idx: list[np.ndarray | None] = []
        n_no_pool = 0
        for source, target in pairs:
            if mode == "global":
                siblings = {t for t in targets_of.get(source, ()) if t != target and t in index_of}
                if siblings:
                    ids = [t for t in pool_ids if t not in siblings]
                    cand_idx.append(np.fromiter((index_of[t] for t in ids), dtype=np.int64, count=len(ids)))
                else:
                    ids = pool_ids
                    cand_idx.append(None)
            else:
                pool = pool_by_pair.get((source, target))
                if not pool:
                    n_no_pool += 1
                    continue
                ids = sorted(({t for t in pool if t in index_of and t in shared} | {target}))
                cand_idx.append(np.fromiter((index_of[t] for t in ids), dtype=np.int64, count=len(ids)))
            queries.append((source, target))
            cand_ids.append(ids)
        return queries, cand_ids, cand_idx, n_no_pool
