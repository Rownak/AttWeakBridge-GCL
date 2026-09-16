# Sprint 1 — Implement the target architecture

## Objective

Build the dataset-agnostic cross-graph alignment system specified in
[target_architecture.md](../target_architecture.md), replacing the legacy notebook-derived scripts.
End state: `cga run ingest|embed|training_data|gcl_train --dataset attack_cwe` runs the full pipeline
end-to-end, `ncit_doid` runs through ingestion, and every phase is covered by unit tests.

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

- [ ] `git mv` legacy source into `legacy/`: `config.py`, `data_preprocessing/`, `LLM_embeddings/`,
      `GCL/`, `run.sh`.
- [ ] Move (filesystem, not git — these are git-ignored) `datasets/` → `legacy/datasets/` and
      `model_outputs/` → `legacy/model_outputs/`.
- [ ] Add `legacy/README.md`: one paragraph stating this is frozen pre-refactor code kept for
      reference, pointing at `dev_docs/existing_architecture.md`.
- [ ] Update `.gitignore`: ignore `legacy/datasets/`, `legacy/model_outputs/`, and the new
      `data/`, `artifacts/`, `models/` roots.
- [ ] Create the new empty tree: `src/cga/`, `configs/`, `tests/unit/`, `tests/integration/`,
      `data/raw/`, `artifacts/`, `models/`.
- [ ] Copy raw source files into the new layout: `legacy/datasets/ics_attack/attack.json` and
      `legacy/datasets/cwe/cwec_v4.12.xml` → `data/raw/attack_cwe/`;
      `benchmark/OAEI-Bio-ML/ncit-doid/` → `data/raw/ncit_doid/`.
- [ ] Verify: `python -c "import config"` fails from repo root (legacy is isolated), and the four
      raw files exist under `data/raw/`.

---

## Phase 1.1 — Package skeleton and `core/`

- [ ] Update `pyproject.toml`: `src` layout, `[project.scripts] cga = "cga.cli:main"`, add
      `pyarrow`, `pydantic>=2`, `pyyaml`, `pytest` (dev group). Run `uv sync`.
- [ ] `core/errors.py` — `ConfigError`, `MissingInputError`, `ArtifactValidationError`.
- [ ] `core/registry.py` — generic `Registry` class (`register(name)` decorator, `get`, `names`);
      one instance each for pipelines, adapters, embedders, samplers, losses.
- [ ] `core/config.py` — YAML deep-merge layering (base → dataset → pipeline → experiment →
      `--config` → `--set`), `${paths.x}` / `${env:VAR}` interpolation, `config_hash` (sha256 of
      resolved config).
- [ ] `core/io.py` — `read_parquet` / `write_parquet` (schema-enforcing), `read_npy` / `write_npy`,
      `read_json` / `write_json`, `sha256_file`.
- [ ] `core/seeds.py` — `seed_everything(seed)` covering `random`, `numpy`, `torch`, `PYTHONHASHSEED`.
- [ ] `core/logging.py` — console + per-run file handler, pipeline/step name in the format string.
- [ ] `core/manifest.py` — `Manifest` model per §3.6, `record_input` / `record_output` with hashing,
      `write()`.
- [ ] `core/artifacts.py` — `ArtifactRef` and resolution of `latest` / explicit `run_id` / path,
      reading and writing `LATEST` files.
- [ ] `core/run.py` — `RunContext` (run_id generation per §3.5, run_dir, seed, device, logger,
      manifest, `resolve()`).
- [ ] `core/pipeline.py` — `Step` protocol and `Pipeline` base class: validate config, run steps in
      order, write manifest, mark `status` success/failed.
- [ ] `configs/base.yaml` per §7.2.
- [ ] Tests: config layering and override precedence; `${}` interpolation; `config_hash` stability;
      registry duplicate-name error; `seed_everything` reproducibility; `ArtifactRef` resolution of
      `latest`.

---

## Phase 1.2 — Canonical data layer

- [ ] `data/ids.py` — `make_node_id(graph_key, source_id)`, `parse_node_id`, validation (no colon in
      `graph_key`).
- [ ] `data/schemas.py` — pyarrow schemas + pydantic row models for `nodes`, `edges`, `texts`,
      `alignments`, `examples`, `status`, `index`, `universe` (§6.1–6.6).
- [ ] `data/graph_store.py` — read/write the four ingest tables with pk-uniqueness and FK checks
      (`edges.{src,dst} ⊂ nodes`, same `graph_role`; `alignments` endpoints in opposite roles).
- [ ] `data/aligned.py` — `AlignedMatrix` (`load` with validation, `row_of`, `select`, `save`) and
      `NodeUniverse` (ordered `node_id` + `graph_role`, `from_parquet`, `save`).
- [ ] `data/embedding_store.py` — shard append, `status.parquet` upsert, `materialize()` →
      `vectors.npy` + `index.parquet`, `pending_nodes()` for resume.
- [ ] `data/example_store.py` — write/read `examples.parquet` with FK checks against a universe.
- [ ] Tests: `node_id` round-trip and rejection of bad input; schema violation raises on write;
      FK violation raises; `AlignedMatrix.select` reorders correctly and raises on a missing id;
      `EmbeddingStore` resume leaves `ok` rows untouched and re-runs `failed` ones;
      materialize excludes non-`ok` rows and keeps `index` aligned.

---

## Phase 1.3 — Adapter layer and `ingest` pipeline

- [ ] `adapters/base.py` — `GraphSpec`, `DatasetAdapter` ABC (§4.1), `@register_adapter`.
- [ ] `adapters/attack_cwe/` — `stix_parser.py`, `cwe_parser.py`, `adapter.py` per §4.2.
- [ ] `adapters/ncit_doid/` — `owl_parser.py` (streaming `owl:Class` + `rdfs:subClassOf`,
      predicate-map-driven text extraction), `refs_parser.py` (`refs_equiv/*.tsv`, candidate-pool
      parsing without `eval`), `adapter.py` per §4.3.
- [ ] `pipelines/ingest/pipeline.py` — the six steps in §5.1, writing the four tables + `stats.json`.
- [ ] `configs/pipelines/ingest.yaml`, `configs/datasets/attack_cwe.yaml`,
      `configs/datasets/ncit_doid.yaml` per §7.3–7.4.
- [ ] `cli.py` — `cga list`, `cga run <pipeline>`, `cga config show`, `cga validate`, `cga inspect`,
      `--dry-run`.
- [ ] Tests: each adapter against committed source snippets in `tests/unit/fixtures/` (a few STIX
      objects, a few `owl:Class` blocks, a few TSV rows) asserting emitted canonical records;
      cross-graph edge raises; duplicate `node_id` raises.
- [ ] Verify: `cga run ingest --dataset attack_cwe` → 1,136 nodes / 2,078 edges;
      `cga run ingest --dataset ncit_doid` → alignments populated with train/test splits.

---

## Phase 1.4 — `embed` pipeline

- [ ] `pipelines/embed/text_embedders.py` — `hf_mean_pool`: tokenize, fixed-size chunking, forward
      pass, mean-pool last hidden layer, mean across chunks.
- [ ] `pipelines/embed/graph_embedders.py` — `deepwalk` (uniform walks) and `node2vec` (2nd-order
      biased walks, p/q) over `edges.parquet`, via `gensim.Word2Vec`; port from
      `legacy/GCL/get_graph_features/2_node2vec_deepwalk.py`.
- [ ] `pipelines/embed/pipeline.py` — `generate` (per-node status recording, `on_error: continue`,
      shard append) → `materialize` → `validate`; `--resume` support.
- [ ] `configs/pipelines/embed.yaml` per §7.5, including the structural-embedder variant.
- [ ] Tests: chunking splits a long text as expected; a node with empty text gets `missing_text` and
      does not shift `index` rows; a raising embedder records `failed` and continues;
      `embedder_id` construction; walk generation is seeded and reproducible.
- [ ] Verify: text and deepwalk stores both produced for `attack_cwe`, `status` all `ok`,
      `vectors.npy` shapes `(1136, 768)` and `(1136, 128)`.

---

## Phase 1.5 — `training_data` pipeline

- [ ] `pipelines/training_data/similarity.py` — cosine matrices, cross-graph (text) and within-graph
      (structural), driven by `graph_role`.
- [ ] `pipelines/training_data/samplers.py` — `rank_topk`, `percentile`, `supervised`; all registered
      and seeded.
- [ ] `pipelines/training_data/pipeline.py` — `build_universe` (intersection policy) → `similarity`
      → `sample` → `split` → `validate_and_manifest`.
- [ ] `configs/pipelines/training_data.yaml` per §7.6.
- [ ] Tests: `build_universe` excludes non-`ok` nodes from either store; each sampler on a small
      hand-built similarity matrix produces the expected pairs; sampler output is reproducible under
      a fixed seed; `supervised` sampler no-ops when `alignments.parquet` is empty;
      `split: from_alignments` keeps test-alignment nodes out of train examples.
- [ ] Verify: `cga run training_data --dataset attack_cwe` produces `universe.parquet` (1,136 rows)
      and a non-empty `examples.parquet` whose ids all resolve in the universe.

---

## Phase 1.6 — `gcl_train` pipeline and evaluation

- [ ] `pipelines/gcl_train/models.py` — `DualEncoder` with GCN and GAT towers (port from
      `legacy/GCL/models/GNN_dual_loss2.py`).
- [ ] `pipelines/gcl_train/losses.py` — `triplet`, `jsd`, `infonce`, registered.
- [ ] `pipelines/gcl_train/trainer.py` — full-batch loop, `alpha·L_text + (1−alpha)·L_graph`,
      `metrics.jsonl` per epoch, checkpoints embedding `universe_sha256`.
- [ ] `pipelines/gcl_train/evaluate.py` — Hits@k / MRR when alignments exist; silhouette over
      `graph_role` + cross-graph similarity stats when they do not.
- [ ] `pipelines/gcl_train/pipeline.py` — `load` (select every matrix onto the universe, remap ids →
      rows) → `train` → `export` → `evaluate` → `manifest`.
- [ ] `configs/pipelines/gcl_train.yaml` and
      `configs/experiments/attack_cwe_secroberta_gcn.yaml` per §7.7–7.8.
- [ ] Tests: each loss against hand-computed values; `load` raises when an example id is outside the
      universe; checkpoint refuses to load against a mismatched `universe_sha256`;
      Hits@k on a toy ranking; evaluate no-ops cleanly with zero alignments.
- [ ] Verify: `cga run gcl_train --dataset attack_cwe --set gcl_train.train.epochs=5` produces
      `node_embeddings/` shape `(1136, 128)`, `metrics.jsonl` with 5 rows, and `eval/metrics.json`.

---

## Phase 1.7 — Finalize

- [ ] `pipelines/embed/finetune.py` — MLM/CLM continued pretraining → `models/llm/{slug}/{run_id}/`,
      wired as the optional first `embed` step.
- [ ] `scripts/check_env.py` (port from legacy) and `scripts/run_all.sh` chaining the four pipelines
      for one experiment config.
- [ ] `tests/integration/` — each pipeline end-to-end on `attack_cwe` with `epochs: 2`, asserting
      manifests, schemas, shapes, and referential integrity.
- [ ] `README.md` — install, dataset setup, the four commands, adding a new dataset (§10).
- [ ] Full run: `scripts/run_all.sh` on `attack_cwe`, then `cga inspect` on the GCL run to confirm
      the manifest chain reaches back to raw file hashes.
