# Sprint 1 — Implement the target architecture

## Objective

Build the dataset-agnostic cross-graph alignment system specified in
[target_architecture.md](../target_architecture.md), replacing the legacy notebook-derived scripts.
End state: `cga run ingest|embed|training_data|gcl_train|evaluate --dataset attack_cwe` runs the full
pipeline end-to-end, `ncit_doid` runs through ingestion, and every phase is covered by unit tests.

Legacy code is moved aside in Phase 1.0 and kept intact for reference. Regression testing against
legacy outputs is **out of scope for this sprint** — it will be planned separately once the
architecture move is complete.

## Conventions for every phase

- Google-style docstrings on every module, class, and public function (per `CLAUDE.md`).
- Unit tests live in `tests/unit/`, mirroring the package layout.
- Each phase ends with an entry in `dev_docs/execution_logs/sprint1.md`.
- Run tests with `uv run pytest tests/unit -q`.

---

## Phase 1.0 — Move legacy code and data aside

- [x] `git mv` legacy source into `legacy/`: `config.py`, `data_preprocessing/`, `LLM_embeddings/`,
      `GCL/`, `run.sh`.
- [x] Move (filesystem, not git — these are git-ignored) `datasets/` → `legacy/datasets/` and
      `model_outputs/` → `legacy/model_outputs/`.
- [x] Add `legacy/README.md`: one paragraph stating this is frozen pre-refactor code kept for
      reference, pointing at `dev_docs/existing_architecture.md`.
- [x] Update `.gitignore`: ignore `legacy/datasets/`, `legacy/model_outputs/`, and the new
      `data/`, `artifacts/`, `models/` roots.
- [x] Create the new empty tree: `src/cga/`, `configs/`, `tests/unit/`, `tests/integration/`,
      `data/raw/`, `artifacts/`, `models/`.
- [x] Copy raw source files into the new layout: `legacy/datasets/ics_attack/attack.json` and
      `legacy/datasets/cwe/cwec_v4.12.xml` → `data/raw/attack_cwe/`;
      `benchmark/OAEI-Bio-ML/ncit-doid/` → `data/raw/ncit_doid/`.
- [x] Verify: `python -c "import config"` fails from repo root (legacy is isolated), and the four
      raw files exist under `data/raw/`.

---

## Phase 1.1 — Package skeleton and `core/`

- [x] Update `pyproject.toml`: `src` layout, `[project.scripts] cga = "cga.cli:main"`, add
      `pyarrow`, `pydantic>=2`, `pyyaml`, `pytest` (dev group). Run `uv sync`.
- [x] `core/errors.py` — `ConfigError`, `MissingInputError`, `ArtifactValidationError`.
- [x] `core/registry.py` — generic `Registry` class (`register(name)` decorator, `get`, `names`);
      one instance each for pipelines, adapters, embedders, samplers, losses.
- [x] `core/config.py` — YAML deep-merge layering (base → dataset → pipeline → experiment →
      `--config` → `--set`), `${paths.x}` / `${env:VAR}` interpolation, `config_hash` (sha256 of
      resolved config).
- [x] `core/io.py` — `read_parquet` / `write_parquet` (schema-enforcing), `read_npy` / `write_npy`,
      `read_json` / `write_json`, `sha256_file`.
- [x] `core/seeds.py` — `seed_everything(seed)` covering `random`, `numpy`, `torch`, `PYTHONHASHSEED`.
- [x] `core/logging.py` — console + per-run file handler, pipeline/step name in the format string.
- [x] `core/manifest.py` — `Manifest` model per §3.6, `record_input` / `record_output` with hashing,
      `write()`.
- [x] `core/artifacts.py` — `ArtifactRef` and resolution of `latest` / explicit `run_id` / path,
      reading and writing `LATEST` files.
- [x] `core/run.py` — `RunContext` (run_id generation per §3.5, run_dir, seed, device, logger,
      manifest, `resolve()`).
- [x] `core/pipeline.py` — `Step` protocol and `Pipeline` base class: validate config, run steps in
      order, write manifest, mark `status` success/failed.
- [x] `configs/base.yaml` per §7.2.
- [x] Tests: config layering and override precedence; `${}` interpolation; `config_hash` stability;
      registry duplicate-name error; `seed_everything` reproducibility; `ArtifactRef` resolution of
      `latest`.

---

## Phase 1.2 — Canonical data layer

- [x] `data/ids.py` — `make_node_id(graph_key, source_id)`, `parse_node_id`, validation (no colon in
      `graph_key`).
- [x] `data/schemas.py` — pyarrow schemas + pydantic row models for `nodes`, `edges`, `texts`,
      `alignments`, `examples`, `status`, `index`, `universe` (§6.1–6.6).
- [x] `data/graph_store.py` — read/write the four ingest tables with pk-uniqueness and FK checks
      (`edges.{src,dst} ⊂ nodes`, same `graph_role`; `alignments` endpoints in opposite roles).
- [x] `data/aligned.py` — `AlignedMatrix` (`load` with validation, `row_of`, `select`, `save`) and
      `NodeUniverse` (ordered `node_id` + `graph_role`, `from_parquet`, `save`).
- [x] `data/embedding_store.py` — shard append, `status.parquet` upsert, `materialize()` →
      `vectors.npy` + `index.parquet`, `pending_nodes()` for resume.
- [x] `data/example_store.py` — write/read `examples.parquet` with FK checks against a universe.
- [x] Tests: `node_id` round-trip and rejection of bad input; schema violation raises on write;
      FK violation raises; `AlignedMatrix.select` reorders correctly and raises on a missing id;
      `EmbeddingStore` resume leaves `ok` rows untouched and re-runs `failed` ones;
      materialize excludes non-`ok` rows and keeps `index` aligned.

---

## Phase 1.3 — Adapter layer and `ingest` pipeline

- [x] `adapters/base.py` — `GraphSpec`, `DatasetAdapter` ABC (§4.1), `@register_adapter`.
- [x] `adapters/attack_cwe/` — `stix_parser.py`, `cwe_parser.py`, `adapter.py` per §4.2.
- [x] `tests/regression/test_attack_cwe_legacy_graph.py` — compare `stix_parser`/`cwe_parser` edge
      output against `legacy/datasets/{ics_attack/attack_graph.json, cwe/cwec_v4.12_graph.json}`
      `links` (as unordered source/target pairs).
- [x] `adapters/ncit_doid/` — `owl_parser.py` (streaming `owl:Class` + `rdfs:subClassOf`,
      predicate-map-driven text extraction), `refs_parser.py` (`refs_equiv/*.tsv`, candidate-pool
      parsing without `eval`), `adapter.py` per §4.3.
- [x] `pipelines/ingest/pipeline.py` — the six steps in §5.1, writing the four tables + `stats.json`.
- [x] `configs/pipelines/ingest.yaml`, `configs/datasets/attack_cwe.yaml`,
      `configs/datasets/ncit_doid.yaml` per §7.3–7.4.
- [x] `cli.py` — `cga list`, `cga run <pipeline>`, `cga config show`, `cga validate`, `cga inspect`,
      `--dry-run`.
- [x] Tests: each adapter against committed source snippets in `tests/unit/adapters/fixtures/` (a few
      STIX objects, a few `owl:Class` blocks, a few TSV rows) asserting emitted canonical records;
      cross-graph edge raises; duplicate `node_id` raises.
- [x] Verify: `cga run ingest --dataset attack_cwe` → **1,217 nodes / 1,922 edges** (not the legacy
      1,136/2,078 — §5.1's node-set policy keeps *all* parsed entities rather than only edge-reachable
      ones, so this is the architecturally-correct count against the committed `ics-attack.json` /
      `cwec_v4.12.xml`, confirmed against `tests/regression/`); `cga run ingest --dataset ncit_doid` →
      24,227 nodes / 18,752 edges, 4,686 alignments populated with train (1,406) / test (3,280) splits.

---

## Phase 1.4 — `embed` pipeline

- [x] `pipelines/embed/text_embedders.py` — `hf_mean_pool`: tokenize, fixed-size chunking, forward
      pass, mean-pool last hidden layer, mean across chunks.
- [x] `pipelines/embed/graph_embedders.py` — `deepwalk` (uniform walks) and `node2vec` (2nd-order
      biased walks, p/q) over `edges.parquet`, via `gensim.Word2Vec`; port from
      `legacy/GCL/get_graph_features/2_node2vec_deepwalk.py`.
- [x] `pipelines/embed/pipeline.py` — `generate` (per-node status recording, `on_error: continue`,
      shard append) → `materialize` → `validate`; `--resume` support.
- [x] `configs/pipelines/embed.yaml` per §7.5, including the structural-embedder variant.
- [x] Tests: chunking splits a long text as expected; a node with empty text gets `missing_text` and
      does not shift `index` rows; a raising embedder records `failed` and continues;
      `embedder_id` construction; walk generation is seeded and reproducible.
- [x] Verify: text and deepwalk stores both produced for `attack_cwe`, `status` all `ok`,
      `vectors.npy` shapes `(1136, 768)` and `(1136, 128)` (actual: 1,217 total nodes per Phase 1.3's
      node-set policy; deepwalk → 1,137 ok / 80 skipped, shape `(1137, 128)`; hf_mean_pool → see
      `dev_docs/execution_logs/sprint_1_4.md` for the real run's ok/missing_text/empty_text counts).

---

## Phase 1.5 — `training_data` pipeline

- [x] `pipelines/training_data/similarity.py` — cosine matrices, cross-graph (text) and within-graph
      (structural), driven by `graph_role`, computed only over `held_out == False` universe rows.
- [x] `pipelines/training_data/samplers.py` — `rank_topk`, `percentile`, `supervised`; all registered
      and seeded.
- [x] `pipelines/training_data/pipeline.py` — `build_universe` (intersection policy; also computes
      `held_out` from `alignments.parquet`'s `test` split per the corrected §5.3 step 1) →
      `similarity` → `sample` → `split` → `validate_and_manifest` (includes the leakage assertion
      that no `test`-split alignment node appears in `examples.parquet`).
- [x] `configs/pipelines/training_data.yaml` per §7.6 as numbered then (now §7.7), including
      `held_out_splits` — that key is removed again in Phase 1.5a.
- [x] Tests: `build_universe` excludes non-`ok` nodes from either store, and correctly flags
      `held_out` from alignment `test` rows; each sampler on a small hand-built similarity matrix
      produces the expected pairs; `rank_topk`/`percentile` never draw a `held_out` node as
      anchor/positive/negative; sampler output is reproducible under a fixed seed; `supervised`
      sampler no-ops when `alignments.parquet` is empty; `validate_and_manifest` raises if a
      test-alignment node id appears anywhere in `examples.parquet`.
- [x] Verify: `cga run training_data --dataset attack_cwe` produces `universe.parquet` (1,136 rows)
      and a non-empty `examples.parquet` whose ids all resolve in the universe.

> **Superseded in part by Phase 1.5a.** As shipped, `held_out` was derived *only* from
> `alignments.parquet`, so `attack_cwe` (no alignments) had no held-out nodes and no test split at
> all. Phase 1.5a moves the partition into `ingest` and makes it uniform across datasets.

---

## Phase 1.6 — Per-node `train`/`val`/`test` split

Motivated by the gap above: `attack_cwe` had no held-out set, and `ncit_doid`'s node-level
eligibility was derived from alignment pairs rather than from a node partition. Architecture
amended in §5.1.1, §5.3, §6.5, §6.7, §7.5, §7.7, §8.4.

**Policy** (decided with the developer): proportional per graph, **identical for every dataset** —
each `graph_role` independently partitioned ~80/10/10 by a seeded shuffle. A dataset's own
alignment split does **not** override the node partition. Both `val` and `test` nodes are excluded
from similarity/sampling.

- [x] `data/schemas.py` — `SPLITS_SCHEMA` / `SplitRecord` per §6.5 (`node_id` pk, `graph_role`,
      `split`); add `split` to `UNIVERSE_SCHEMA` alongside the existing `held_out`.
- [x] `data/graph_store.py` — `write_splits` / `read_splits` with FK check against `nodes.parquet`
      and a completeness check (exactly one row per node).
- [x] `pipelines/ingest/splitter.py` — seeded proportional per-`graph_role` partition; validates
      `fractions` sums to 1.0; deterministic given `(nodes, fractions, seed)`.
- [x] `pipelines/ingest/pipeline.py` — new `split_nodes` step (§5.1 step 6) writing
      `splits.parquet`; per-split-per-graph counts into `stats.json`.
- [x] `configs/pipelines/ingest.yaml` — `split.fractions` / `split.seed` per §7.5.
- [x] `pipelines/training_data/pipeline.py` — `build_universe` reads `splits.parquet`, carries
      `split` onto `universe.parquet`, and computes
      `held_out = split in {val, test} or node in alignment test pair` (§5.3 step 1); `supervised`
      sampler restricted to `train`-split endpoints; renamed the old `split` step to
      `assign_example_split`, inheriting each example's split from its anchor.
- [x] `configs/pipelines/training_data.yaml` — drop `held_out_splits` and the `split` block (both
      now owned by `ingest`).
- [x] Tests: fractions produce the expected per-graph counts and sum to the node total; the split is
      reproducible under a fixed seed and changes with a different one; every node appears exactly
      once; `build_universe` marks `val`/`test` nodes `held_out`; `ncit_doid`-shaped fixture with an
      OAEI `test` alignment pair whose endpoints landed in our `train` split still marks them
      `held_out` (the union rule); examples never reference a `held_out` node.
- [x] Verify: re-run `ingest` + `training_data` for `attack_cwe`; `splits.parquet` shows ~80/10/10
      within **both** `attack` and `cwe`; `examples.parquet` anchors are all `train`-split; record
      the new example count (it will drop from 34,080, since ~20% of anchors are now held out).

---

## Phase 1.7 — `gcl_train` pipeline

- [x] `pipelines/gcl_train/models.py` — `DualEncoder` with GCN and GAT towers (port from
      `legacy/GCL/models/GNN_dual_loss2.py`).
- [x] `pipelines/gcl_train/losses.py` — `triplet`, `infonce`, registered. `jsd` was deliberately
      dropped in this port (decided with the developer): the legacy JSD loss treats a softmax over an
      L2-normalized embedding's dimensions as a probability distribution, which has no principled
      interpretation, and the architecture's `jsd` mention was not worth carrying that approximation
      forward for.
- [x] `pipelines/gcl_train/trainer.py` — full-batch loop, `alpha·L_text + (1−alpha)·L_graph`,
      `metrics.jsonl` per epoch, checkpoints embedding `universe_sha256`.
- [x] `pipelines/gcl_train/pipeline.py` — `load` (select every matrix onto the universe, remap ids →
      rows) → `train` → `export` → `manifest`. No evaluation step: that is Phase 1.8.
- [x] `configs/pipelines/gcl_train.yaml` per §7.8 (`monitor` block disabled by default, `loss.name`
      restricted to `triplet | infonce`) and `configs/experiments/attack_cwe_secroberta_gcn.yaml` per
      §7.10.
- [x] Tests: each loss against hand-computed values; `load` raises when an example id is outside the
      universe; checkpoint refuses to load against a mismatched `universe_sha256`.
- [x] Verify: `cga run gcl_train --dataset attack_cwe --set gcl_train.train.epochs=5` produces
      `node_embeddings/` shape `(1136, 128)` and `metrics.jsonl` with 5 rows (loss decreased
      monotonically 0.85 → 0.46 across the 5 epochs, confirming the loop is actually learning).

---

## Phase 1.8 — Exploratory analysis in the `evaluate` pipeline

Neighborhood-preservation analysis for datasets without alignment labels (§5.5a), ported from
`legacy/GCL/gcl_analysis/2_precision@k_text_graph.py`. Query is always the learned embedding;
relevant items are the reference top-k (text or structural), scored per view, per `k`, within
`graph_role`. Legacy `3_top_pairs_text_graph.py` is not ported.

- [x] `pipelines/evaluate/metrics.py` — `precision_at_k`, `hit_at_k`, `mrr`, `rank_by_similarity`
      (deterministic tie-break by node id), `top_k` as pure functions over ranked id lists.
- [x] `pipelines/evaluate/neighborhood.py` — `score_view`: reference top-k from the text and
      structural embedding stores, restricted to same-`graph_role` pairs and nodes eligible in both
      the learned and reference matrix.
- [x] `pipelines/evaluate/visualize.py` — optional t-SNE scatter colored by `graph_role`, ported from
      `1_tsne_plot.py::tsne_plot1`; visualization only, never a reported metric.
- [x] `pipelines/evaluate/pipeline.py` — `load` (resolve the `gcl_train` run and follow its manifest
      chain to `training_data` → `ingest`/text/structural embedding stores) → `neighborhood` →
      `visualize` (optional) → `validate_and_manifest`, writing
      `artifacts/evaluations/{dataset_id}/{run_id}/metrics.json` (+ `plots/` if enabled).
- [x] `configs/pipelines/evaluate.yaml` — `inputs.gcl_train_run`, `neighborhood` block (`ks`,
      `views: [text, structural]`), `visualize.enabled`.
- [x] Tests: each metric on a hand-computed toy case (`tests/unit/pipelines/evaluate/test_metrics.py`);
      cross-`graph_role` exclusion and reference-eligibility filtering
      (`test_neighborhood.py`); end-to-end pipeline test against a fabricated `gcl_train` run,
      including an unknown-view `ConfigError` and byte-identical metrics across two runs
      (`test_pipeline.py`).
- [x] Verify: `cga run evaluate --dataset attack_cwe` on the real Phase 1.7 run (`node_embeddings`
      shape `(1136, 128)`) writes precision@k/hit@k/MRR for both views across `ks: [1, 5, 10, 20]`
      (e.g. text/graph_1 hit@1 = 0.217, structural/graph_2 hit@1 = 0.274); re-running produced a
      byte-identical `metrics.json`.

---

## Phase 1.9 — Comparative `evaluate` pipeline

Rework `evaluate` from scoring one model into comparing **several embedding methods side by side**
under one or more registered **evaluations** (§5.5, §5.5a, §6.10; rationale in
`dev_docs/design_recommendation.md`). A *method* is any per-node `AlignedMatrix` — a baseline
embedding store or a trained model's export — so baselines and models are scored by the same code on
the same node universe. This produces the paper's comparison table and makes a new evaluation a new
file rather than a pipeline change. Phase 1.8's `metrics.py` and `visualize.py` carry over;
`neighborhood.py` is replaced.

- [x] `pipelines/gcl_train/pipeline.py` + `cli._run_root` — partition output by `model_id` (default
      `model.encoder`) → `models/gcl/{dataset_id}/{model_id}/{run_id}/`, so `gcn` and `gat` keep
      separate `LATEST` pointers. Prerequisite: today's flat `LATEST` makes a GCN-vs-GAT comparison
      unexpressible with `ref: latest`. Existing runs become invisible under the new layout; re-run
      both trainings, no compatibility shim.
- [x] `pipelines/evaluate/rows.py` — `MetricRow` + long-form `metrics.parquet` writer per §6.10
      (`METRICS_SCHEMA` in `data/schemas.py`), plus the nested `metrics.json` digest **derived from
      the same rows** so the two cannot drift.
- [x] `pipelines/evaluate/metrics.py` — generalize to relevance **sets**: `normalized_precision_at_k`
      (`|L_k ∩ R| / min(k,|R|)`), `recall_at_k` (`|L_k ∩ R| / |R|`), set-based `hit_at_k` and `mrr`.
      Both precision columns are emitted so neither is mislabeled when `k > |R|`.
- [x] `pipelines/evaluate/methods.py` — `MethodSpec` + `load_method` resolving `embedding_store`
      (`artifacts/embeddings/{dataset}/{embedder_id}/`) and `gcl_run`
      (`models/gcl/{dataset}/{model_id}/` → `node_embeddings/`). Each lands in `manifest.inputs[]` as
      `method:<id>`; a view's reference block reuses the same spec and loader.
- [x] `pipelines/evaluate/evaluators/neighborhood.py` — owns the methods × views × `graph_role` × `k`
      loop, with a `relevance_providers` registry for views: `hop_radius` (relevant = all nodes within
      radius `r`, radius-bounded BFS on the fly, isolates fall out via an empty set) and
      `cosine_top_k` (top-k over a declared reference store). Eligibility intersects the **candidate
      pool** across methods, not just the query set, so precision stays comparable; `query_set_sha256`
      records that the methods shared a query set.
- [x] `pipelines/evaluate/evaluators/cluster_quality.py` — methods × `graph_role` × `n_clusters`,
      silhouette. Closes the §8.4 diagnostic promise and is the second implementation that proves the
      evaluator layer is a layer.
- [x] `pipelines/evaluate/pipeline.py` — rewrite to four kind-agnostic steps: `load` (validate config,
      resolve `ingest_run`, build the graph, `load_method` every method and reference) → `evaluate`
      (registry lookup per `evaluations[]` entry, collect rows) → `visualize` (optional, per method) →
      `validate_and_manifest`. Deleted `neighborhood.py`. Each evaluator validates its own opaque
      `params` with its own pydantic model.
- [x] `configs/pipelines/evaluate.yaml` — thin defaults per §7.9 (`inputs.ingest_run`, empty
      `methods`/`evaluations`, `visualize`); `configs/experiments/neighborhood_compare.yaml` per
      §7.11 carries the paper's methods and evaluations.
- [x] Produced the missing comparison inputs (no code change): a generic
      `bert-base-uncased` text store (`hf_mean_pool-bert_base`), a `node2vec` graph store
      (`node2vec-node2vec`), and `gcn`/`gat` training runs under the new `model_id` layout.
- [x] Tests: the `min(k,|R|)` denominator (`|R|=2, k=5` → 1.0); both precision columns coincide when
      `k ≥ |R|` and diverge when `k < |R|`; `hop_radius` on a path graph (`r=1 ⊂ r=2`, isolate excluded,
      a node whose only neighbor is off-pool treated as isolated); one query set and one
      `query_set_sha256` across methods when a method is missing nodes; duplicate `method_id`/
      `evaluation_id` → `ConfigError`; unknown evaluation `kind`/relevance `provider` → `ConfigError`;
      an experiment overlay fully replaces the default `methods` list; `gcn` and `gat` runs do not
      clobber each other's `LATEST`; `metrics.parquet` matches `METRICS_SCHEMA`; determinism across two
      runs. 202 new/changed tests across `test_metrics.py`, `test_relevance.py`, `test_methods.py`,
      `test_pipeline.py`; full suite 325 passed.
- [x] Verify: `cga run evaluate --dataset attack_cwe --experiment neighborhood_compare` on the real
      `attack_cwe` data (4 methods: `deepwalk`, `secroberta`, `gcn`, `gat`; 3 views: `structural_r1`,
      `structural_r2`, `textual`) writes `metrics.parquet` with 424 rows (384 neighborhood + 40
      `cluster_quality`, matching `4 × 3 × 2 × 4` and `4 × 2 × 5`); `query_set_sha256` identical across
      all 4 methods within each (view, role) (e.g. `graph_1`: 203 queries, `graph_2`: 933 queries);
      `deepwalk` leads the structural views (0.93 normalized-precision@20 vs. GCN/GAT's ~0.11–0.15
      after only 5 training epochs) and `secroberta` leads the textual view (MRR 0.64 vs. GCN/GAT's
      ~0.45), as expected for undertrained dual-encoders; `cga inspect` lists every method plus the
      textual reference (`method:ref:textual`) as manifest inputs, tracing back through `gcl_train` →
      `training_data` → `embed`/`ingest` to raw file hashes; re-running produced a byte-identical
      `metrics.parquet` (excluding the `run_id` column).

---

## Phase 1.10 — `plot` pipeline

A thin sixth pipeline turning an `evaluate` run's long-form `metrics.parquet` into figures (§5.6,
§7.12). Separate from `evaluate` so re-plotting never re-scores, and so one evaluation run can be
plotted several ways. A **figure spec** selects a slice of §6.10's rows and maps columns to plot
roles (`x`/`hue`/`facet`); it names no evaluation kind, so precision-vs-`k` and silhouette-vs-
`n_clusters` are the same renderer against different filters. `evaluate.visualize` (t-SNE) stays put:
it projects an embedding matrix, not metric rows.

- [x] `pipelines/plot/renderers.py` — `renderers` registry + a `FigureSpec` pydantic model
      (`figure_id`, `kind`, `select`, `x`, `hue`, `facet`, `params`). Ship `line` first; add `bar` as
      the second implementation that proves the registry is an extension point.
- [x] `pipelines/plot/pipeline.py` — three steps: `load` (resolve `inputs.evaluate_run` via
      `ctx.resolve`, read `metrics.parquet`, apply each figure's `select` and raise `ConfigError` on
      an empty selection — a blank PNG is not an acceptable output) → `render` (registry lookup per
      `figures[]` entry, one PNG each) → `validate_and_manifest`. Register as `"plot"`.
- [x] `cli._run_root` — `plot` → `artifacts/plots/{dataset_id}`.
- [x] `configs/pipelines/plot.yaml` per §7.12 (`inputs.evaluate_run: latest`, empty `figures`,
      `format`); `configs/plots/neighborhood_compare.yaml` with the paper's two figures
      (normalized precision@k faceted by view × role; silhouette vs `n_clusters` faceted by role).
- [x] Tests: `select` picks the expected rows from a toy `metrics.parquet`; empty selection →
      `ConfigError`; unknown `kind` → `ConfigError`; duplicate `figure_id` → `ConfigError`;
      one PNG per configured figure with the expected filenames; faceting on a null column
      (`view_id` for `cluster_quality`) does not crash; end-to-end against a fabricated `evaluate`
      run, including the manifest recording it as an input. 14 new tests across
      `test_renderers.py`, `test_pipeline.py`; full suite 339 passed.
- [x] Verify: `cga run plot --dataset attack_cwe --config configs/plots/neighborhood_compare.yaml`
      against the real Phase 1.9 evaluate run writes both figures under
      `artifacts/plots/attack_cwe/{run_id}/figures/`; `cga inspect` traces back to the `evaluate` run
      and its methods; re-running the same spec is a no-op on the numbers (plots only).

### Phase 1.10a — Per-run training loss curves

Added after the initial `plot` pipeline: with all 8 `gcl_train` model_id combinations
(`{gcn,gat}-{secroberta,securebert2}-{deepwalk,node2vec}`) now trained to 400 epochs, each run's
`metrics.jsonl` (§6.8: one `{epoch, loss, loss_graph, loss_text, lr, elapsed_s}` object per epoch)
needs a loss-vs-epoch figure. This is **not** a `metrics.parquet`-driven figure — `gcl_train`'s
per-epoch losses are training diagnostics, not evaluation metrics, and were deliberately kept out of
`evaluate`'s output (§6.8: "Authoritative metrics live in the `evaluate` run's `metrics.json`, not
here"). Per the developer: one figure **per run**, not a cross-run comparison — every `model_id`
under the dataset gets its own loss-curve PNG, discovered automatically rather than named in config.

- [x] `pipelines/plot/loss_curves.py` — `read_loss_curve(run_dir)` loading `metrics.jsonl` into a
      small typed structure (epoch, loss, loss_graph, loss_text, lr); `render_loss_curve(rows,
      output_path, *, title)` drawing three lines (combined/graph/text loss) vs. epoch on one axes,
      log-scale y-axis (legacy losses span >2 orders of magnitude over 400 epochs). Pure function of
      the loaded rows, independent of `renderers.py`'s `FigureSpec`/`select` machinery, since there is
      no `metrics.parquet` row to select here.
- [x] `pipelines/plot/pipeline.py` — new `discover_gcl_runs(gcl_dataset_dir)` helper: lists every
      `model_id` directory under `models/gcl/{dataset_id}/`, resolves each one's `LATEST` (skipping a
      `model_id` with no successful run or whose run has no `metrics.jsonl`), and returns
      `{model_id: run_dir}`. New `LossCurvesStep` (runs after `render`, before
      `validate_and_manifest`): for each discovered run, records it as a manifest input
      (`gcl_run:{model_id}`) via `ctx.resolve` and writes `figures/loss_curve_{model_id}.png`. Gated
      by `plot.loss_curves.enabled` (default `false`, so existing figure-only configs are unaffected).
      `LoadStep` now tolerates an empty `figures` list when `loss_curves.enabled` is true, since a
      loss-curves-only run needs no `evaluate` run at all.
- [x] `configs/pipelines/plot.yaml` — add `loss_curves: {enabled: false}` block.
      `configs/plots/neighborhood_compare.yaml` — set `loss_curves.enabled: true` so the paper run
      also gets every discovered `gcl_train` loss curve alongside the two `metrics.parquet` figures.
- [x] Tests: `read_loss_curve` parses a hand-written `metrics.jsonl` fixture into the expected rows;
      `discover_gcl_runs` finds every `model_id` with a `LATEST`, skips one without a successful run
      or without `metrics.jsonl` (doesn't raise), and is unaffected by the stale flat
      `models/gcl/{dataset}/LATEST` noted in Phase 1.9's execution log; `render_loss_curve` writes one
      PNG with three series; end-to-end pipeline tests with `loss_curves.enabled: true` and fabricated
      `gcl_train` runs assert one `loss_curve_{model_id}.png` and one `gcl_run:{model_id}` manifest
      input per run, that loss-curves-only (`figures: []`) doesn't raise and needs no `evaluate_run`
      input, and that a `model_id` without a successful run is skipped without failing the pipeline;
      `loss_curves.enabled: false` (the default) produces no loss-curve output and no new manifest
      inputs, so existing behavior is unchanged. 12 new tests across `test_loss_curves.py`,
      `test_pipeline.py`; full suite 351 passed.
- [x] Verify: `cga run plot --dataset attack_cwe --config configs/plots/neighborhood_compare.yaml`
      against the real trained `gcl_train` runs (400 epochs each) writes one
      `figures/loss_curve_{model_id}.png` per discovered `model_id` — the 8
      `{gcn,gat}-{secroberta,securebert2}-{deepwalk,node2vec}` combinations plus 2 older standalone
      `gcn`/`gat` verification runs from Phase 1.9, 10 total — each showing combined/graph/text loss
      decreasing over 400 epochs; `cga inspect` lists all 10 `gcl_run:{model_id}` inputs alongside the
      existing `evaluate_run` input, each traceable to its own `training_data`/`embed`/`ingest`
      chain.
