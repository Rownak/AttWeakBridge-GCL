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

<!-- Full alignment `evaluate` pipeline (Pipeline 5, §5.5/§8.4 — Hits@k/MRR over alignment pairs,
     silhouette/similarity-distribution diagnostics) is deferred: out of scope for the paper, and
     `attack_cwe` has no alignments to rank against anyway. Revisit post-paper. -->

## Phase 1.8 — Exploratory analysis in the `evaluate` pipeline

Not the alignment-metrics pipeline from §5.5/§8.4 (deferred — see note above). This phase is a
lighter, paper-focused rewrite of `legacy/GCL/gcl_analysis/2_precision@k_text_graph.py`: does the
learned embedding preserve the neighborhoods that text similarity and graph/structural similarity
already define? `1_tsne_plot.py`'s t-SNE plots are ported as optional visualization only, never a
metric. `3_top_pairs_text_graph.py` is skipped entirely — out of scope.

Two independent reference views, each scored the same way:

- **Text preservation** — reference neighborhood = top-k by cosine similarity in the text-embedding
  space (e.g. SecBERT/GPT text embeddings). Query = learned node embedding. For each node, does the
  learned embedding's top-k retrieve the text reference's top-k?
- **Structural preservation** — reference neighborhood = top-k by cosine similarity in the
  graph/structural-embedding space (DeepWalk, node2vec) or shortest-path-derived similarity. Same
  retrieval question, structural reference instead of text.

Relevance label for both views: node `j` is "relevant" to query node `i` iff `j` is in `i`'s
reference top-k (text or structural, per view). Retrieval direction is always *learned embedding →
reference top-k*, never the reverse — this measures whether the learned space preserves an existing
neighborhood, not whether the reference recovers the learned one.

Metrics, computed per view (text, structural), each restricted to same-`graph_role` comparisons
(`attack`↔`attack`, `cwe`↔`cwe`) matching the legacy split, and reported per `k`:

- **Precision@k** — `|top_k(learned) ∩ top_k(reference)| / k`, averaged over nodes. Interpretation:
  fraction of the learned neighborhood that agrees with the reference neighborhood.
- **Hit@k** — 1 if the reference's single nearest neighbor (k=1 relevant item) appears in the
  learned top-k, else 0, averaged over nodes. Interpretation: how often the single most-relevant
  reference node is retrievable at all within budget k.
- **MRR** — mean of `1/rank` of the reference's nearest neighbor within the learned ranking,
  averaged over nodes. Interpretation: how highly the single most-relevant reference node is ranked,
  not just whether it made the cutoff.

- [ ] `pipelines/evaluate/metrics.py` — `precision_at_k`, `hit_at_k`, `mrr`, all taking a learned
      similarity matrix and a reference similarity matrix (or reference top-k indices) plus `ks`;
      pure functions, no I/O.
- [ ] `pipelines/evaluate/reference_neighborhoods.py` — build text and structural reference
      similarity from the `embed` pipeline's stores (no separate reference model to train).
- [ ] `pipelines/evaluate/visualize.py` — optional t-SNE scatter of learned embeddings colored by
      `graph_role`, ported from `1_tsne_plot.py::tsne_plot1`; explicitly not used for metrics.
- [ ] `pipelines/evaluate/pipeline.py` — `load` (learned embeddings via `gcl_train` manifest chain,
      plus text/structural embedding stores) → `score` (metrics.py over both views, all `ks`) →
      `visualize` (optional) → `validate_and_manifest`, writing
      `artifacts/evaluations/{dataset_id}/{run_id}/metrics.json` (+ `plots/` if enabled).
- [ ] `configs/pipelines/evaluate.yaml` — `ks`, `views: [text, structural]`, `visualize.enabled`.
- [ ] Tests: `precision_at_k`/`hit_at_k`/`mrr` on hand-built toy similarity matrices with known
      answers; same-`graph_role` restriction excludes cross-role pairs; `score` is order-independent
      w.r.t. how reference top-k ties are broken (documented, not silently nondeterministic).
- [ ] Verify: `cga run evaluate --dataset attack_cwe` on the Phase 1.7 run writes `metrics.json` with
      precision@k/hit@k/MRR for both text and structural views across configured `ks`.

---

## Phase 1.9 — Finalize

- [ ] `pipelines/embed/finetune.py` — MLM/CLM continued pretraining → `models/llm/{slug}/{run_id}/`,
      wired as the optional first `embed` step.
- [ ] `scripts/check_env.py` (port from legacy) and `scripts/run_all.sh` chaining the five pipelines
      for one experiment config.
- [ ] `tests/integration/` — each pipeline end-to-end on `attack_cwe` with `epochs: 2`, asserting
      manifests, schemas, shapes, and referential integrity.
- [ ] `README.md` — install, dataset setup, the five commands, adding a new dataset (§10).
- [ ] Full run: `scripts/run_all.sh` on `attack_cwe`, then `cga inspect` on the evaluation run to
      confirm the manifest chain reaches back through gcl_train and ingest to raw file hashes.
