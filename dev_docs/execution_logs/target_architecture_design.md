# Target architecture design

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

---

## Phase 1.2 — Split `evaluate` into its own pipeline; constrain `core/`

### What was implemented

Two changes to [target_architecture.md](dev_docs/target_architecture.md), both prompted by an
external ML research engineer's review of the design:

1. **Evaluation promoted from a `gcl_train` step to Pipeline 5** (`evaluate`), with its own package,
   config file, artifact root, and manifest.
2. **A "`core/` stays boring" constraint added to §2**, bounding what the framework layer may grow
   into.

Phase numbering in [sprint_1.md](dev_docs/sprints/sprint_1.md) shifted: the old Phase 1.6
(`gcl_train` + evaluation) became Phase 1.6 (`gcl_train` only) and a new Phase 1.7 (`evaluate`);
Finalize moved 1.7 → 1.8.

### Approach / key decisions

- **Accepted the reviewer's evaluate split, for stronger reasons than they gave.** Their argument was
  convenience (evaluating saved models independently). The operative reasons here are (a) with
  `evaluate` inside `gcl_train`, changing `eval.ks` changes `gcl_train`'s `config_hash` and
  invalidates a completed 400-epoch run — a cost that exists today, not just after tooling adoption;
  (b) MLflow evaluates a *registered model version*, which is a separate run by construction;
  (c) DVC gets a re-runnable evaluation stage that does not dirty the training stage.
- **`evaluate` resolves its own inputs by walking the manifest chain.** It takes a `gcl_train` run
  ref, reads that run's manifest, and follows `inputs[]` back to the `ingest` run for
  `nodes.parquet` / `alignments.parquet`. Nothing is passed by hand, so an evaluation cannot be run
  against mismatched sources — the same guard `training_data` → `gcl_train` already uses.
- **Kept in-loop metrics, renamed to `gcl_train.monitor`** (default off). It calls the same
  `ranking.py` and appends to `metrics.jsonl` for watching a run in progress, but only Pipeline 5
  writes reported numbers. Naming it `monitor` rather than `eval` makes the split unambiguous.
- **Rejected the reviewer's other simplifications** (dropping `registry.py`, `manifest.py`,
  `pipeline.py`, `io.py`). Each is a planned integration surface: the registry is what makes §10's
  "add a dataset without touching core" promise true and already backs five component families; the
  manifest is both the traceability guarantee and the MLflow mirror point; declared step inputs and
  outputs are what map onto `dvc.yaml` stages.
- **Converted their valid underlying concern into an explicit constraint instead.** The reviewer
  correctly identified a failure mode (`core/` drifting into a home-grown Kedro) but aimed it at the
  wrong files. §2 now forbids the specific things that would constitute that drift — DAG/dependency
  resolution, conditional or parallel step scheduling, config-driven step composition, a plugin
  lifecycle beyond import-time registration, and generic wrappers in `io.py` — while keeping the
  modules themselves.

### Files modified

- `dev_docs/target_architecture.md` — §2 tree (`pipelines/evaluate/`, `configs/pipelines/evaluate.yaml`,
  `artifacts/evaluations/`), new "`core/` stays boring" block, §5 preamble (four → five pipelines,
  rationale), §5.4 trimmed, new §5.5, §6.7 and §6.8 metric ownership, new §7.8 with §7.9 renumbered,
  §8.4 rewritten as Pipeline 5, §8.6 eval tests, §9 flow diagram, §11 MLflow and DVC rows.
- `dev_docs/sprints/sprint_1.md` — Objective, Phase 1.6 narrowed, new Phase 1.7, Finalize → 1.8.

### Assumptions / limitations

- `evaluate` writes to `artifacts/evaluations/{dataset_id}/{run_id}/` rather than under the model
  directory, so a model can be scored many times without mutating an immutable run dir. The link
  runs one way: the evaluation manifest names the `gcl_train` run it scored.
- `attack_cwe` has no alignments, so the ranking path stays unexercised on real data until
  `ncit_doid` reaches Phase 1.7 or ground truth is supplied. Phase 1.7 tests cover it with a toy
  ranking of known answer, and cover the zero-alignment no-op explicitly.
- The reviewer's recommendation was assessed against the stated MLflow/Hydra/DVC trajectory. Were
  the project to stay hand-run indefinitely, their case for dropping the framework modules would be
  considerably stronger.

### CLI commands

No code yet — documentation only. The commands this phase adds to the eventual surface:

```bash
cga run evaluate --dataset attack_cwe                       # score the latest gcl_train run
cga run evaluate --dataset attack_cwe --set evaluate.inputs.gcl_run=<run_id>
cga inspect artifacts/evaluations/attack_cwe/<run_id>       # walk back to raw file hashes
```
