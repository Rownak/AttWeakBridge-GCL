# Sprint 1 — Execution Log

## Phase 1.0 — Target architecture design

### What was implemented

Wrote [dev_docs/target_architecture.md](dev_docs/target_architecture.md): the proposed
production-style architecture for the incremental refactor, using
[existing_architecture.md](dev_docs/existing_architecture.md) as the as-is reference. Covers the
repository layout, four pipelines (`ingest`, `embed`, `training_data`, `gcl_train`) behind a
registry and an `awb` CLI, the identity/storage strategy, artifact formats and path conventions,
manifests, YAML config layering with examples, data schemas, the end-to-end data flow, a six-phase
refactoring plan, and the seams reserved for MLflow / Hydra / DVC / PySpark.

### Approach / key decisions

- **Canonical `node_id = "{source}:{source_id}"`** (STIX id for ATT&CK, numeric id for CWE) replaces
  the positional `doc_id_to_emb_id` scheme. Every table that refers to a node carries this string
  column; dense `.npy` arrays are only ever written/read together with an `index.parquet` sidecar
  (`AlignedMatrix`), and positions are computed exactly once, in `training_data.build_universe`.
- **Embedding store = shards + `status.parquet` + materialized vectors.** Per-node status
  (`ok / missing_text / empty_text / failed / skipped`) with attempt counts makes missing text and
  failed generation explicit and makes `--resume` re-run only non-`ok` rows without shifting anyone.
- **Structural embeddings (deepwalk/node2vec) live in the `embed` pipeline**, not in graph
  preparation or training-data creation — they are node→vector functions with the same store
  contract as text embedders. This deviates from the literal "text embedding generation" wording in
  the brief; flipping it later only changes registry/config placement, not the store contract.
- **Ingestion keeps every parsed node**; the legacy "drop nodes without a description" decision
  moves downstream (status table + universe policy) so exclusions are recorded with a reason.
- **Immutable run directories + `LATEST` pointers + `manifest.json`** per run give versioning and
  traceability without DVC/MLflow; `data/`, `artifacts/`, `models/` are separate roots because they
  will belong to different tools later.
- **Parquet for records, `.npy`+index for dense arrays, no pickle between pipelines** — chosen for
  PySpark/DVC compatibility and cross-Python-version stability.
- Config layering (`base → dataset → pipeline → experiment → --config → --set`) is a single pure
  function, deliberately mirroring Hydra's composition so the migration is a swap, not a rewrite.
- Legacy behaviors that are arguably bugs (no special tokens in chunking, `ChildOf`-only CWE edges,
  unseeded shuffles) are exposed as explicit config flags with the legacy value as default, so
  parity with the Sprint-0 baseline can be established first and behavior changed deliberately.
- Hop-distance features and the `text_hop` path are excluded from phase 1 (no live consumer in the
  legacy tree) and listed as optional.

### Files modified or created

- `dev_docs/target_architecture.md` (new)
- `dev_docs/execution_logs/sprint1.md` (new — this file)

### Challenges / assumptions / limitations

- **Sprint numbering assumed.** The design starts the refactor on branch
  `v0.3_yaml_based_pipelines`; it is logged here as Sprint 1 / Phase 1.0. Rename if the sprint plan
  numbers it differently.
- Design only — no code, configs, or tests were created. Phase 1 of §11 is the first implementation
  step.
- The plan adds three dependencies (`pyarrow`, `pydantic>=2`, `pyyaml`) not yet in `pyproject.toml`.
- Evaluation (precision@k etc.) is specified as optional because no evaluation code exists in the
  working tree to port; it needs a ground-truth mapping artifact that ingestion does not currently
  produce.
- The Gradio demo is explicitly out of scope for this design and would need re-homing on the new
  artifact stores in a later phase.

### CLI commands to test / run this phase

Documentation only — nothing to execute. Once Phase 1 lands, the intended smoke test is:

```bash
uv sync
uv run awb list
uv run awb run ingest --dataset ics_attack --dry-run
uv run awb run ingest --dataset ics_attack
uv run awb validate data/processed/ics_attack-cwec_v4.12/<run_id>
```

---

## Phase 1.1 — Generalize the architecture to arbitrary graph pairs

### What was implemented

Revised [dev_docs/target_architecture.md](dev_docs/target_architecture.md) so the design describes a
general **cross-graph alignment** system over two text-attributed graphs, rather than an ATT&CK–CWE
specific pipeline. ATT&CK–CWE becomes one dataset; OAEI Bio-ML `ncit-doid` is designed in as the
second, per [oaei_bioml_ncit_doid_dataset_structure.md](dev_docs/oaei_bioml_ncit_doid_dataset_structure.md).

### Approach / key decisions

- **Adapter layer (new §4).** Added a `DatasetAdapter` ABC with five iterators (`iter_nodes`,
  `iter_edges`, `iter_texts`, `iter_alignments`, `source_fingerprint`) plus `validate_sources()`,
  registered by `dataset_id` in the same registry used for pipelines. `src/cga/adapters/` is stated
  in the document as the only dataset-specific code in the repo. Adding a dataset = one adapter
  package + one `configs/datasets/<name>.yaml` (§10), touching no pipeline or core module.
- **`graph_role` replaces domain names.** Nodes carry `graph_key` (`attack`, `ncit`, …) for
  provenance and `graph_role` (`graph_1`/`graph_2`) for logic. Generic code branches only on
  `graph_role`, so "cross-domain text similarity / within-domain graph similarity" became
  "cross-graph / within-graph" with no dataset knowledge. This also removes the legacy
  "graph_1 rows come first" ordering assumption from `universe.parquet`.
- **`alignments.parquet` is a new canonical table** (§6.4) carrying gold pairs, source-provided
  splits, and ranking candidate pools. It is empty for `attack_cwe` and populated for `ncit_doid`;
  consumers must handle zero rows. This makes evaluation (§8.4) generic: Hits@k/MRR when alignments
  exist, unsupervised diagnostics when they do not. A `supervised` sampler was added alongside
  `rank_topk`/`percentile` to use alignment pairs as positives where available.
- **Per-ontology predicate maps live in YAML**, not code — NCIT uses `P90`/`P97`/`P325` while DOID
  uses `oboInOwl:hasExactSynonym`/`IAO_0000115`, so text variants are composed from abstract roles
  (`label`, `synonym`, `definition`) that the dataset config maps to concrete predicates. Both
  dataset YAMLs therefore have the same shape and the generic pipelines read only shared keys.
- **NCIT logical axioms deferred.** Only direct `rdfs:subClassOf` edges are parsed initially
  (16,030 NCIT / 9,388 DOID edges is sufficient structure); `owl:equivalentClass`/`intersectionOf`/
  `Restriction` interpretation and `refs_subs` subsumption alignment moved to the backlog.
- **Layout changes per instruction**: raw data is per dataset *pair* (`data/raw/attack_cwe/`,
  `data/raw/ncit_doid/`); dataset configs are pair-centric (`attack_cwe.yaml`, `ncit_doid.yaml`);
  `fixtures/tiny` and all Sprint-0/tiny-data workflow removed — `attack_cwe` (1,136 nodes) is the
  development and integration-test dataset directly.
- **Simplification.** Removed the phased refactoring plan, the legacy-export/parity-test bridge, the
  `[E]`/`[O]` tagging scheme (replaced by a single §12 backlog), similarity-matrix caching, the
  diagnostic similarity columns on `examples.parquet`, `report`/histogram steps, artifact GC,
  early stopping, and JSON logging. Package renamed `attweakbridge` → `cga`, CLI `awb` → `cga`,
  to match the now dataset-neutral scope.
- Preserved unchanged: canonical `node_id` identity, Parquet + `AlignedMatrix`/index-sidecar
  storage, the embedding store with per-node status and resume, immutable run dirs with `LATEST`,
  manifests and the traceability chain, config layering, and the MLflow/Hydra/DVC/PySpark seams.

### Files modified or created

- `dev_docs/target_architecture.md` (rewritten)
- `dev_docs/execution_logs/sprint1.md` (this section appended)

### Challenges / assumptions / limitations

- **Package/CLI rename assumed** (`cga`). If the repo name or distribution name should stay
  `attweakbridge`, this is a find-and-replace in the document before implementation starts.
- **`node_id` for ontologies uses the IRI fragment** (`ncit:C3224`, `doid:DOID_4362`) rather than
  the full IRI, with the full IRI kept as `external_id`. This assumes fragments are unique within
  each ontology — true for the sampled classes, but worth verifying against the full files during
  adapter implementation. `id_style: fragment` is a config knob so the alternative is available.
- **`ncit_doid` is not used in the test suite** (≈15.8k + 8.5k classes, 55 MB NCIT OWL); only
  committed source snippets exercise its adapter in unit tests. Full-size ingestion performance is
  unverified, hence the `pruning.mode` backlog item.
- The `alignments.parquet` `candidates` column assumes `test.cands.tsv` parses to a list of target
  IRIs; the file stores them as a Python-literal tuple string, which the adapter must parse
  explicitly rather than `eval`.
- Design only — no code, configs, or tests were created.

### CLI commands to test / run this phase

Documentation only — nothing to execute. The intended smoke test once ingestion exists:

```bash
uv run cga list adapters
uv run cga run ingest --dataset attack_cwe --dry-run
uv run cga run ingest --dataset attack_cwe
uv run cga validate data/processed/attack_cwe/<run_id>
uv run cga run ingest --dataset ncit_doid
```
