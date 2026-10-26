# Sprint 1 Phase 1.9 — Comparative `evaluate` pipeline

## What was implemented

`evaluate` was reworked from scoring one model against two fixed reference views (Phase 1.8) into a
generic comparison pipeline: several **methods** (baseline embedding stores or trained models),
scored under one or more registered **evaluations**, producing one long-form `metrics.parquet` a
paper figure can pivot directly. Two evaluators ship: `neighborhood` (methods × views × `graph_role`
× `k`) and `cluster_quality` (methods × `graph_role` × `n_clusters`, silhouette). The design and
rationale are in `dev_docs/design_recommendation.md`; the target architecture (§5.5, §5.5a, §6.10,
§7.9–§7.11) was updated to match before this implementation began.

## How it was completed

- **`pipelines/gcl_train/pipeline.py::build_model_id`** — partitions `models/gcl/{dataset_id}/` by
  `model_id` (default `model.encoder`, i.e. `gcn`/`gat`), mirroring `embed`'s `embedder_id`. Also
  updated `cli._run_root` and the `gcl_train` manifest summary. This was the hard prerequisite: the
  old flat `LATEST` meant a GAT run's `LATEST` write silently clobbered a GCN run's, making a
  GCN-vs-GAT row in one comparison table impossible to express with `ref: latest`.
- **`pipelines/evaluate/rows.py`** — `MetricRow` (one `(evaluation, method, scope, sweep point,
  metric)` value), `MetricsDigest` (nested `evaluation → view → role → method → param → metric` view
  derived from a list of rows, plus per-scope provenance via `add_scope_provenance`), and
  `write_metrics` writing both `metrics.parquet` and `metrics.json` from the same `MetricsDigest` in
  one call so the two artifacts cannot disagree. `METRICS_SCHEMA`/`MetricRowRecord` added to
  `data/schemas.py` per §6.10.
- **`pipelines/evaluate/metrics.py`** (rewritten) — generalized from ranked-list-vs-single-nearest to
  set-based relevance: `normalized_precision_at_k` (`/min(k,|R|)`), `recall_at_k` (`/|R|`), set-based
  `hit_at_k` (any relevant in top-k) and `mrr` (first relevant's reciprocal rank). Added
  `top_k_by_similarity` and `mrr_via_argsort` as vectorized equivalents of the Python-loop versions,
  both proven to agree with the reference implementation across 50 random seeds including exact-tie
  cases. `top_k_by_similarity` deliberately does **not** use raw `np.argpartition` on similarity
  alone — that can select the wrong member of an exactly-tied group — so it partitions on a composite
  `(-sim, id_rank)` key via `np.lexsort` instead (still no Python-level tuple sort, the actual cost
  being avoided).
- **`pipelines/evaluate/relevance.py`** (new) — `RelevanceProvider` protocol (`prepare` /
  `defined_for` / `relevant`, returning `set[str]`, no `is_ranked` flag) with two registered
  providers: `HopRadiusRelevance` (radius-bounded `nx.single_source_shortest_path_length` on the
  *candidate-restricted* subgraph, so a node whose only real edge falls outside the candidate pool is
  correctly treated as an isolate within that scope) and `CosineTopKRelevance` (top-k over a declared
  reference embedding store, reusing `MethodSpec`/`load_method`).
- **`pipelines/evaluate/methods.py`** (new) — `MethodSpec.from_config` (validates `kind` ∈
  `{embedding_store, gcl_run}` and its required id field) and `load_method` (resolves through
  `ctx.resolve(f"method:{id}", ...)`, so every method and every view's reference lands in
  `manifest.inputs[]` with its concrete resolved run_id).
- **`pipelines/evaluate/evaluators/`** (new package) — `Evaluator` protocol and `evaluators` registry
  in `__init__.py`; `neighborhood.py` (the core loop: candidates = role ids ∩ the intersection of
  every method's node ids, `provider.prepare()` on that pool, queries = candidates ∩
  `provider.defined_for()` — intersecting the *pool*, not just the query set, is what keeps precision
  comparable when one method is missing nodes); `cluster_quality.py` (k-means + `silhouette_score`
  per method/role/`n_clusters`, deliberately built as the second evaluator to prove the registry is
  an actual extension point).
- **`pipelines/evaluate/pipeline.py`** (rewritten) — four steps, none aware of any evaluation kind:
  `load` (validate config eagerly, resolve `ingest` directly rather than walking a `gcl_train`
  manifest chain — with N methods there is no canonical chain — build the graph, `load_method` every
  configured method) → `evaluate` (registry lookup per `evaluations[]` entry, collect rows — the
  pipeline's only loop) → `visualize` (optional t-SNE per method) → `validate_and_manifest`
  (`metrics.parquet` + derived `metrics.json`, manifest, `LATEST`).
- **`configs/pipelines/evaluate.yaml`** — thin defaults (`inputs.ingest_run`, empty `methods`/
  `evaluations`). **`configs/experiments/neighborhood_compare.yaml`** (new) — the paper's comparison:
  4 methods (`deepwalk`, `secroberta`, `gcn`, `gat`) × 3 views (`structural_r1`, `structural_r2`,
  `textual`) + a `cluster_quality` evaluation, run via the existing `--experiment` flag (no CLI
  change; `deep_merge` replaces `methods`/`evaluations` lists wholesale).

## Files modified or created

- `src/cga/pipelines/gcl_train/pipeline.py` — `build_model_id`, `model_id`-partitioned `LATEST`,
  manifest summary
- `src/cga/cli.py` — `_run_root` resolves `gcl_train`'s `model_id` partition
- `src/cga/data/schemas.py` — `METRICS_SCHEMA`, `MetricRowRecord`
- `src/cga/pipelines/evaluate/rows.py`, `relevance.py`, `methods.py` (new)
- `src/cga/pipelines/evaluate/evaluators/__init__.py`, `neighborhood.py`, `cluster_quality.py` (new)
- `src/cga/pipelines/evaluate/metrics.py` (rewritten), `pipeline.py` (rewritten), `__init__.py`
  (updated to register evaluators)
- `src/cga/pipelines/evaluate/neighborhood.py` — deleted (superseded by `evaluators/neighborhood.py`)
- `configs/pipelines/evaluate.yaml` (rewritten), `configs/experiments/neighborhood_compare.yaml` (new)
- `tests/unit/pipelines/gcl_train/test_pipeline.py` — updated for the `model_id` output path
- `tests/unit/pipelines/evaluate/test_metrics.py` (rewritten), `test_relevance.py`, `test_methods.py`
  (new), `test_pipeline.py` (rewritten)
- `tests/unit/pipelines/evaluate/test_neighborhood.py` — deleted (its scenarios moved into
  `test_relevance.py`/`test_pipeline.py`)
- `dev_docs/sprints/sprint_1.md` — Phase 1.9 checkboxes marked complete with real numbers

## Challenges, ambiguity, assumptions

- **`np.argpartition` on raw similarity is unsound under ties.** The original plan proposed it for
  `top_k_by_similarity`'s hot path; testing with an exact 3-way tie showed it can pick the wrong
  member of a tied group before the id tie-break ever runs, silently dropping a candidate a correct
  ranking would have kept. Fixed by partitioning on `(-sim, id_rank)` via `np.lexsort` instead — a
  full `O(n log n)` sort of the *keys*, but still zero Python-level tuple comparisons, which was the
  actual cost the optimization targets.
- **`normalized_precision_at_k` and `recall_at_k` coincide when `k ≥ |R|`, not `k ≤ |R|`** — an error
  in my own first draft of the test suite, caught by running it. `min(k,|R|)` equals `|R|` once
  `k ≥ |R|`, at which point both metrics divide by `|R|`; for `k < |R|`, normalized precision divides
  by `k` (ordinary precision) while recall still divides by `|R|`, so they diverge there instead.
- **`query_set_sha256` was identical across every view in the real `attack_cwe` run** (not just
  across methods within one view). This is correct, not a bug: all 4 configured methods happen to
  cover the exact same node universe in this run, and none of `structural_r1`/`structural_r2`/
  `textual`'s providers excluded any node as undefined (no isolates fell out) — so every view landed
  on the same query set per role. A run with genuinely isolated nodes, or a method missing some
  node's embedding, would show the hashes diverge across views/methods as designed (covered by
  `test_evaluate_query_set_sha256_identical_across_methods`, which forces a missing-node case).
- **GCN/GAT trail the baselines badly in this verification run** (normalized-precision@20 ~0.11–0.15
  vs. DeepWalk's 0.93 on the structural view) — expected and not a defect: both dual-encoders were
  trained for only 5 epochs to keep verification fast, not for research-quality convergence. The
  paper run needs `gcl_train.train.epochs` at its real setting (400 per `configs/pipelines/
  gcl_train.yaml`) for both encoders before the comparison numbers mean anything.
- **A stale flat `LATEST` remains at `models/gcl/attack_cwe/LATEST`** from before this phase's
  migration (pointing at a pre-`model_id` run). It is inert — no code path reads it now that
  `gcl_train`/`evaluate` both resolve through `{model_id}/LATEST` — but it was left in place rather
  than deleted, since removing artifacts wasn't asked for.
- Per the design doc, `configs/pipelines/gcl_train.yaml` needs no new key: `model_id` defaults to
  `model.encoder`, so today's `gcn`/`gat` configs partition correctly with zero config change.

## CLI commands to test and run

```
uv run pytest tests/unit/pipelines/evaluate tests/unit/pipelines/gcl_train -q
uv run pytest tests/unit -q                                    # full suite, 325 passed

# One-time inputs for the comparison (no code change; already produced in this run)
uv run cga run embed --dataset attack_cwe --set embed.embedder.kind=hf_mean_pool \
    --set embed.embedder.model_ref=bert-base-uncased --set embed.embedder.label=bert_base
uv run cga run embed --dataset attack_cwe --set embed.embedder.kind=node2vec \
    --set embed.embedder.label=node2vec
uv run cga run gcl_train --dataset attack_cwe --set gcl_train.model.encoder=gcn
uv run cga run gcl_train --dataset attack_cwe --set gcl_train.model.encoder=gat

uv run cga run evaluate --dataset attack_cwe --experiment neighborhood_compare
uv run cga inspect artifacts/evaluations/attack_cwe/<run_id>
```

Verified end-to-end against real `attack_cwe` data: `metrics.parquet` has 424 rows (384 neighborhood
= 4 methods × 3 views × 2 roles × 4 ks, plus 40 cluster_quality = 4 methods × 2 roles × 5
`n_clusters`); `query_set_sha256` identical across all 4 methods within each (view, role);
`cga inspect` traces every method plus the textual reference back through `gcl_train`/
`training_data`/`embed`/`ingest` to raw file hashes; two consecutive runs produced a byte-identical
`metrics.parquet` (excluding the `run_id` column, which is expected to differ).
