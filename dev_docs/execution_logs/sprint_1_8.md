# Sprint 1 Phase 1.8 — Exploratory analysis in the `evaluate` pipeline

## What was implemented

A new `evaluate` pipeline (`src/cga/pipelines/evaluate/`) that scores a `gcl_train` run's learned
node embeddings against two label-free reference views — text-embedding similarity and
structural-embedding (DeepWalk/node2vec) similarity — using Precision@k, Hit@k, and MRR. This is
the neighborhood-preservation design in `target_architecture.md` §5.5a, not the alignment-ranking
`evaluate` pipeline (§5.5, Phase 1.9), which is unimplemented pending labeled-dataset work.

## How it was completed

- **`pipelines/evaluate/metrics.py`** — pure functions: `rank_by_similarity` (descending similarity,
  ties broken by node id for determinism), `top_k`, `precision_at_k`, `hit_at_k`, `mrr`. Retrieval
  direction is fixed: the ranking always comes from the learned embedding, the relevant set always
  comes from the reference view's own top-k, never the reverse.
- **`pipelines/evaluate/neighborhood.py`** — `score_view(learned_matrix, reference_matrix,
  graph_role_by_node, role, ks)` restricts comparisons to one `graph_role` at a time (never
  `attack`↔`cwe`), intersects node ids present in both the learned and reference matrix (a node
  missing from a reference store, e.g. no text, is silently excluded from that view — matching
  `training_data.similarity`'s existing eligibility convention), and averages each metric over every
  eligible query node.
- **`pipelines/evaluate/visualize.py`** — `plot_tsne_by_role`, a straight port of legacy
  `1_tsne_plot.py::tsne_plot1`'s t-SNE scatter colored by `graph_role`. Writes a PNG only; returns
  nothing, so it can never be mistaken for a metric.
- **`pipelines/evaluate/pipeline.py`** — four steps: `load` (resolve the `gcl_train` run via
  `ArtifactRef`, then follow its manifest's `training_data` input, then that manifest's
  `text_embeddings`/`graph_embeddings`/`ingest` inputs — nothing is passed in by hand, matching
  `gcl_train.LoadStep`'s existing pattern) → `neighborhood` (score_view for each configured view ×
  role) → `visualize` (skipped unless `evaluate.visualize.enabled`) → `validate_and_manifest`
  (writes `metrics.json`, records manifest outputs, writes `LATEST`).
- **`configs/pipelines/evaluate.yaml`** — `inputs.gcl_train_run`, `neighborhood.{ks, views}`,
  `visualize.{enabled, perplexity}`.
- Wired into `cga.pipelines.__init__` (registers on import) and `cli._run_root` (routes to
  `artifacts/evaluations/{dataset_id}/{run_id}`, matching §5.5/§5.5a).

## Files modified or created

- `src/cga/pipelines/evaluate/__init__.py`, `metrics.py`, `neighborhood.py`, `visualize.py`,
  `pipeline.py` (new)
- `src/cga/pipelines/__init__.py` — registers `evaluate`
- `src/cga/cli.py` — `_run_root` entry for `evaluate`
- `configs/pipelines/evaluate.yaml` (new)
- `tests/unit/pipelines/evaluate/__init__.py`, `test_metrics.py`, `test_neighborhood.py`,
  `test_pipeline.py` (new)
- `dev_docs/sprints/sprint_1.md` — Phase 1.8 checkboxes marked complete with the real verify numbers
- `dev_docs/target_architecture.md` — §5.5a already described this design (added when the phase was
  planned); no changes needed during implementation

## Challenges, ambiguity, assumptions

- The legacy script's `precision@k`/`hit@k` conflated "hit" with "overlap count," and computed
  Spearman correlation as a third, separate metric (`isSrc` branch). None of that was ported
  verbatim — this implementation defines Hit@k and MRR against the reference's single nearest
  neighbor specifically (standard IR definitions), and Precision@k as top-k set overlap, per the
  metric definitions agreed in the Phase 1.8 planning pass.
- `n_queries` is reported per `(view, role, k)` even though it is currently role/view-invariant
  within a fixed universe (identical eligible-id sets) — kept per-k in case a future `k`-dependent
  eligibility rule is added, and because the pipeline's `metrics.json` structure already nests by
  `k`.
- A `graph_role` with fewer than 2 eligible nodes (e.g. a tiny test fixture, or a role reduced to
  zero after reference-store intersection) scores as `n_queries: 0`, all metrics `0.0` — this is
  distinct from an error, since it can legitimately happen for a small dataset.

## CLI commands to test and run

```
uv run pytest tests/unit/pipelines/evaluate -q
uv run pytest tests/unit -q                          # full suite, 201 passed
uv run cga run evaluate --dataset attack_cwe          # against the existing Phase 1.7 gcl_train run
uv run cga run evaluate --dataset attack_cwe --set evaluate.visualize.enabled=true
uv run cga inspect artifacts/evaluations/attack_cwe/<run_id>
```

Verified end-to-end against the real `attack_cwe` `gcl_train` run (`node_embeddings` shape
`(1136, 128)`): text/graph_1 hit@1 = 0.217, MRR = 0.336; structural/graph_2 hit@1 = 0.274,
MRR = 0.430; metrics increase monotonically with `k` within each view/role as expected. Two
consecutive runs produced a byte-identical `metrics.json`.
