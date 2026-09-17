# Target Architecture — Cross-Graph Alignment by Graph Contrastive Learning

**Scope.** A clean, YAML-configured architecture for the refactored system. The legacy
implementation described in [existing_architecture.md](existing_architecture.md) is the starting
point, but the target system is **dataset-agnostic**: it solves the general problem of aligning two
text-attributed graphs, and ATT&CK–CWE is one dataset among several.

This iteration deliberately excludes MLflow, Hydra, DVC, and PySpark; §11 records the seams that
keep them addable later without moving files or changing pipeline boundaries. A small backlog of
deferred items is in §12. Refactoring phases are planned separately.

---

## 1. Problem setting

The system operates on a **graph pair**: two independent, text-attributed graphs with no edges
between them.

| Concept | Meaning |
|---|---|
| `graph_1`, `graph_2` | The two graphs. Roles are fixed per dataset (`graph_1` = source/left, `graph_2` = target/right) but carry no semantics beyond that. |
| Node | An entity in one graph, with a stable identifier from its source data. |
| Intra-graph edge | A typed relation between two nodes **of the same graph**. No cross-graph edges are ingested. |
| Node text | One or more text variants per node (name, description, synonyms, concatenations thereof). |
| Node metadata | Non-text attributes retained for filtering, display, and evaluation. |
| Alignment pair | A ground-truth correspondence between a `graph_1` node and a `graph_2` node, used for evaluation (and optionally supervision). Optional — a dataset may have none. |

The pipeline learns one embedding space in which nodes from both graphs are directly comparable,
using text embeddings, structural embeddings, and a contrastive objective over anchor/positive/
negative examples.

### 1.1 Datasets in scope

| `dataset_id` | `graph_1` | `graph_2` | Native format | Alignment ground truth |
|---|---|---|---|---|
| `attack_cwe` | MITRE ATT&CK (`attack`) | MITRE CWE (`cwe`) | STIX JSON bundle; CWE XML catalog | none currently |
| `ncit_doid` | NCIT (`ncit`) | DOID (`doid`) | OWL/XML ontologies | `refs_equiv/{train,test}.tsv`, `test.cands.tsv` |

Everything downstream of ingestion sees only the canonical representation in §3 and is identical for
both datasets. Adding a third dataset means writing one adapter and one YAML file (§4).

### 1.2 Design goals

Derived from the legacy properties in `existing_architecture.md` §10:

| Legacy property | Target property |
|---|---|
| Identity is an implicit array row (`doc_id_to_emb_id`); a dropped text shifts every downstream array | Every stored vector, edge, and training example carries a **stable canonical `node_id`**. Arrays are never consumed without their sidecar index. |
| Hardcoded to one dataset; ATT&CK/CWE names throughout the pipeline code | One canonical schema; all dataset-specific logic isolated in registered adapters. |
| No orchestration; ~13 scripts run by hand from specific directories | Five pipelines behind one CLI, a pipeline registry, explicit upstream-run references. |
| Global mutable `config.py`; hyperparameters hardcoded in five files | Layered YAML (`base` → `dataset` → `pipeline` → `experiment` → CLI overrides), validated on load, snapshotted into every run. |
| Silent truncation on exception; stale artifacts consumed unnoticed | Per-node status tables, artifact validation on read, hashes in manifests, fail-fast on mismatch. |
| No seeds, no provenance | One seed per run applied to all RNGs; a `manifest.json` per run linking config, upstream run ids, code revision, input hashes. |
| No tests | Unit tests for stores, schemas, adapters, samplers; integration tests per pipeline. |

Non-goals: distributed execution, experiment UI, remote storage, model serving.

---

## 2. Repository structure

```
<repo>/
├── pyproject.toml                  # deps + [project.scripts] cga = cga.cli:main
├── uv.lock / .python-version
├── README.md
├── CLAUDE.md
│
├── configs/                        # all YAML config; nothing else is configuration
│   ├── base.yaml                   #   paths, logging, seed, device
│   ├── datasets/                   #   one file per DATASET PAIR
│   │   ├── attack_cwe.yaml
│   │   └── ncit_doid.yaml
│   ├── pipelines/                  #   defaults per registered pipeline
│   │   ├── ingest.yaml
│   │   ├── embed.yaml
│   │   ├── training_data.yaml
│   │   ├── gcl_train.yaml
│   │   └── evaluate.yaml
│   └── experiments/                #   named overlays spanning pipelines
│       └── attack_cwe_secroberta_gcn.yaml
│
├── src/cga/                        # installable package (src layout)
│   ├── __init__.py
│   ├── cli.py                      # `cga` entry point: run / list / validate / inspect / config
│   ├── core/                       # framework-level, pipeline- and dataset-agnostic
│   │   ├── config.py               #   YAML layering, dotted overrides, pydantic validation
│   │   ├── logging.py
│   │   ├── registry.py             #   generic name→class registry (pipelines, adapters, embedders,
│   │   │                           #   samplers, losses all use it)
│   │   ├── pipeline.py             #   Pipeline base class + Step protocol
│   │   ├── run.py                  #   RunContext: run_id, dirs, seed, logger, manifest
│   │   ├── manifest.py
│   │   ├── artifacts.py            #   ArtifactRef resolution (latest | run_id | path)
│   │   ├── io.py                   #   parquet / npy / json helpers with sha256
│   │   ├── seeds.py
│   │   └── errors.py
│   ├── data/                       # canonical schemas and stores — the identity strategy
│   │   ├── ids.py                  #   node_id construction/parsing
│   │   ├── schemas.py              #   pyarrow schemas + pydantic row models
│   │   ├── graph_store.py          #   nodes / edges / texts / alignments tables
│   │   ├── embedding_store.py      #   vectors + index + per-node status
│   │   ├── example_store.py        #   training examples keyed by node_id
│   │   └── aligned.py              #   AlignedMatrix + NodeUniverse
│   ├── adapters/                   # >>> the only dataset-specific code in the repo <<<
│   │   ├── base.py                 #   DatasetAdapter ABC + @register_adapter
│   │   ├── attack_cwe/
│   │   │   ├── adapter.py          #     AttackCweAdapter
│   │   │   ├── stix_parser.py      #     ATT&CK STIX JSON → canonical records
│   │   │   └── cwe_parser.py       #     CWE XML → canonical records
│   │   └── ncit_doid/
│   │       ├── adapter.py          #     NcitDoidAdapter
│   │       ├── owl_parser.py       #     OWL/XML → classes, subClassOf edges, annotations
│   │       └── refs_parser.py      #     refs_equiv/*.tsv → alignment pairs
│   ├── pipelines/
│   │   ├── ingest/pipeline.py      #   Pipeline 1 — runs the selected adapter, normalizes, validates
│   │   ├── embed/
│   │   │   ├── pipeline.py         #   Pipeline 2
│   │   │   ├── finetune.py         #     LM continued pretraining (optional step)
│   │   │   ├── text_embedders.py   #     HF encoder → per-node vector
│   │   │   └── graph_embedders.py  #     deepwalk / node2vec → per-node vector
│   │   ├── training_data/
│   │   │   ├── pipeline.py         #   Pipeline 3
│   │   │   ├── similarity.py
│   │   │   └── samplers.py
│   │   ├── gcl_train/
│   │   │   ├── pipeline.py         #   Pipeline 4
│   │   │   ├── models.py           #     DualEncoder, GCN/GAT towers
│   │   │   ├── losses.py           #     triplet, jsd, infonce
│   │   │   └── trainer.py
│   │   └── evaluate/
│   │       ├── pipeline.py         #   Pipeline 5
│   │       ├── ranking.py          #     Hits@k / MRR against alignment pairs
│   │       └── diagnostics.py      #     unsupervised checks when alignments are absent
│   └── py.typed
│
├── scripts/
│   ├── check_env.py
│   └── run_all.sh                  # chains the five pipelines for one experiment
│
├── data/
│   ├── raw/                        # immutable source data, one dir per DATASET PAIR
│   │   ├── attack_cwe/
│   │   │   ├── attack.json
│   │   │   └── cwec_v4.12.xml
│   │   └── ncit_doid/
│   │       ├── ncit.owl
│   │       ├── doid.owl
│   │       └── refs_equiv/{train,test,full}.tsv
│   └── processed/{dataset_id}/{run_id}/        # Pipeline 1 output
│
├── artifacts/
│   ├── embeddings/{dataset_id}/{embedder_id}/{run_id}/
│   ├── training_data/{dataset_id}/{run_id}/
│   └── evaluations/{dataset_id}/{run_id}/
│
├── models/
│   ├── llm/{base_model_slug}/{run_id}/
│   └── gcl/{dataset_id}/{run_id}/
│
├── tests/
│   ├── unit/                       # ids, config, stores, adapters, samplers, losses
│   └── integration/                # each pipeline end-to-end on attack_cwe
│
└── dev_docs/
```

**Why `data/` `artifacts/` `models/` are separate roots.** Different lifecycles and different future
owners: `data/` is what DVC will version, `models/` is what an MLflow registry will log, `artifacts/`
is derived and garbage-collectable by run id.

**`core/` stays boring.** Its modules exist to be the integration surface for MLflow, Hydra, and DVC
later (§11) — not to become an execution framework. Concretely, `core/` **must not** grow:

- a DAG or dependency resolver between steps — `Pipeline.run()` is `for step in self.steps: step(ctx)`,
  in declaration order, nothing more;
- conditional, parallel, or retry-based step scheduling;
- config-driven step composition (steps are Python attributes of a pipeline class, not YAML entries);
- a plugin lifecycle beyond `registry.register(name)` at import time;
- generic read/write wrappers in `io.py` — it holds schema-enforced Parquet/npy/JSON helpers and
  `sha256_file`, because hashing must happen in one place (§8.3), and nothing else.

The declared inputs/outputs on each step earn their keep by mapping onto `dvc.yaml` stages; anything
beyond that declaration is orchestration this project does not do. If a `core/` module starts looking
like Kedro, the abstraction has overshot and should be pushed back down into the pipeline that needed it.

**Testing data.** `attack_cwe` (1,136 nodes, 2,078 edges) is small enough to serve as the development
and integration-test dataset directly; no separate fixture subset is maintained. `ncit_doid` is
large (≈15.8k + 8.5k classes) and is not used in the test suite.

---

## 3. Canonical representation

Every adapter produces exactly these four tables. Nothing downstream knows what a dataset is.

### 3.1 Node identity

```
node_id = "{graph_key}:{source_id}"
```

`graph_key` is the dataset's name for one side (`attack`, `cwe`, `ncit`, `doid`); `source_id` is the
identifier the source data itself provides.

| Dataset | `graph_1` example | `graph_2` example |
|---|---|---|
| `attack_cwe` | `attack:attack-pattern--0a2a5b45-…` (STIX id) | `cwe:79` |
| `ncit_doid` | `ncit:C3224` (IRI fragment) | `doid:DOID_4362` |

Rules:

- Assigned **once**, by the adapter, from the source record's own identifier — never from a counter,
  loop index, or file order.
- A `graph_role` column (`graph_1` / `graph_2`) is carried on every node, so downstream code does
  domain-aware logic (cross-graph vs. within-graph) without knowing the graph names.
- `external_id` (`T0800`, `CWE-79`, the full IRI) is metadata for display and evaluation, not
  identity — source vocabularies can revoke/replace human-facing ids while the internal id is stable.
- `dataset_id` is the pair name (`attack_cwe`, `ncit_doid`) and namespaces every artifact directory.

### 3.2 Tabular storage: Parquet with declared schemas

All record-oriented artifacts are **Parquet** with a pyarrow schema declared in `data/schemas.py`,
enforced on write and read. Chosen over JSON/pickle for typed columns, cross-Python-version
stability, column-selective reads, and native PySpark/DVC/pandas compatibility. Small
human-readable summaries (`manifest.json`, `stats.json`) stay JSON. **No pickle crosses a pipeline
boundary.**

### 3.3 Dense arrays: `AlignedMatrix` = `.npy` + index sidecar

Vectors stay `.npy` float32 (what torch/gensim/sklearn consume), but an `.npy` is **never** written
or read without its index:

```
vectors.npy       # (n, d) float32
index.parquet     # row:int32, node_id:str — exactly n rows, row = 0..n-1, node_id unique
manifest.json     # shape, dtype, sha256 of both, producer run id
```

```python
m = AlignedMatrix.load(dir)      # validates n_rows == len(index), uniqueness, hashes
m.row_of("cwe:79")               # -> int
m.select(universe.node_ids)      # -> reordered copy; raises on any missing id
```

**`NodeUniverse`** is the in-process replacement for the legacy `attack_weak_range.json`: an ordered
list of `node_id`s a training run operates on, with each node's `graph_role`. Every array a training
run touches is `select()`-ed onto the same universe **by id** before a tensor exists, so two stores
with different row counts can no longer silently mis-align.

### 3.4 Embedding store: vectors + status, retryable

One store per `(dataset_id, embedder_id, run_id)`:

```
artifacts/embeddings/{dataset_id}/{embedder_id}/{run_id}/
├── manifest.json
├── status.parquet          # one row per input node — the ground truth of what happened
├── shards/part-*.parquet   # incremental append-only writes during generation
├── vectors.npy             # materialized from shards: status == "ok" rows only
└── index.parquet
```

`status ∈ {ok, missing_text, empty_text, failed, skipped}` with `attempts`, `error`,
`input_sha256`. This is what makes the store robust:

- A node with no usable text is recorded as `missing_text`, not dropped — downstream knows *why*.
- A failed forward pass is caught **per node**, logged, recorded as `failed`, and generation
  continues. The legacy `try/except: break` truncation is gone.
- `cga run embed --resume <run_id>` re-processes only non-`ok` rows, increments `attempts`, appends
  a shard, re-materializes. Row identity is `node_id`, so retries never shift positions.
- `input_sha256` ties each vector to the exact text (or edge set) it came from; a changed upstream
  artifact makes the store detectably stale.

Graph embedders use the same layout; nodes isolated from the graph get `status = skipped`.

### 3.5 Versioning: immutable run directories + `LATEST`

- Every invocation gets `run_id = {YYYYMMDD}T{HHMMSS}Z-{config_hash[:8]}` and writes only into its
  own directory. Directories are never overwritten; re-running produces a sibling.
- Each lineage directory holds a `LATEST` file naming the most recent **successful** run. Downstream
  configs reference upstream artifacts as `latest` or an explicit run_id; `latest` is resolved at
  start and the concrete id is what the manifest records.
- `config_hash` is the sha256 of the fully-resolved config, so identical runs are visibly siblings.

### 3.6 Manifests

One `manifest.json` per run directory:

```json
{
  "manifest_version": 1,
  "pipeline": "embed", "pipeline_version": "0.3.0",
  "run_id": "20260916T142233Z-3f9a1c2e", "status": "success",
  "started_at": "...", "finished_at": "...", "duration_s": 812.4,
  "dataset_id": "attack_cwe", "seed": 42,
  "config_hash": "3f9a1c2e…", "config_resolved_path": "config.resolved.yaml",
  "code": {"git_commit": "72b17f4", "git_dirty": false, "package_version": "0.3.0"},
  "env": {"python": "3.13.2", "torch": "2.6.0+cu124", "device": "cuda:0"},
  "inputs":  [{"name": "texts", "pipeline": "ingest", "run_id": "20260916T1201…",
               "path": "data/processed/attack_cwe/…/texts.parquet", "sha256": "…"}],
  "outputs": [{"name": "vectors", "path": "vectors.npy", "sha256": "…",
               "shape": [1136, 768], "dtype": "float32"}],
  "summary": {"n_ok": 1136, "n_failed": 0, "n_missing_text": 0}
}
```

Traceability from a result back to source data is the chain of `inputs[].run_id` links:
gcl run → training_data run → embedding runs → ingest run → raw file hashes. `cga inspect` walks it.

---

## 4. Dataset adapter layer

The single place dataset-specific knowledge lives.

### 4.1 Interface

```python
# cga/adapters/base.py

@dataclass(frozen=True)
class GraphSpec:
    key: str                  # "attack", "ncit", …
    role: str                 # "graph_1" | "graph_2"
    display_name: str

class DatasetAdapter(ABC):
    """Reads one dataset's native files and yields canonical records."""

    name: str                       # registry key == dataset_id, e.g. "ncit_doid"
    ConfigModel: type[BaseModel]    # adapter-specific config (file paths, predicate maps, …)
    graphs: tuple[GraphSpec, GraphSpec]

    def validate_sources(self) -> None: ...
        # raise MissingInputError with actionable paths before any parsing

    @abstractmethod
    def iter_nodes(self) -> Iterator[NodeRecord]: ...

    @abstractmethod
    def iter_edges(self) -> Iterator[EdgeRecord]: ...
        # intra-graph only; the pipeline rejects any edge crossing graph_role

    @abstractmethod
    def iter_texts(self) -> Iterator[TextRecord]: ...
        # one row per (node_id, text_variant)

    def iter_alignments(self) -> Iterator[AlignmentRecord]:
        return iter(())         # default: dataset has no ground truth

    def source_fingerprint(self) -> dict[str, str]:
        # {relative_path: sha256} for every raw file read — recorded in the manifest
```

Registration mirrors the pipeline registry:

```python
@register_adapter("ncit_doid")
class NcitDoidAdapter(DatasetAdapter): ...
```

`cga/adapters/__init__.py` imports each package so registration happens on import. **Adding a
dataset = one adapter package + one `configs/datasets/<name>.yaml`.** No file outside
`cga/adapters/` and `configs/datasets/` changes.

### 4.2 `attack_cwe` adapter

- `stix_parser.py` — walks the STIX bundle's `objects`; nodes are objects with an `id`; edges are
  `type == "relationship"` objects mapped to `(source_ref, target_ref, relationship_type)`.
  Dangling endpoints are counted and reported, not silently dropped.
- `cwe_parser.py` — `xml.etree` under the CWE-7/XHTML namespaces; nodes are `Weakness` elements;
  edges are `Related_Weaknesses` entries filtered by `nature` (default `[ChildOf]`, configurable).
- Text variants: `description` (ATT&CK `Description`; CWE `Name + Description +
  Extended_Description`) and `full_metadata` (adds `x_mitre_detection` / the five extra CWE fields).
- No alignment pairs. Evaluation for this dataset is unsupervised only (§8.4) until a
  ground-truth mapping is supplied.

### 4.3 `ncit_doid` adapter

- `owl_parser.py` — streaming OWL/XML parse of each ontology. Nodes are `owl:Class` elements keyed
  by `rdf:about`; edges are direct `rdfs:subClassOf rdf:resource` pointers. NCIT's
  `owl:equivalentClass` / `intersectionOf` / `Restriction` logical axioms are **not** interpreted in
  the initial design — both ontologies expose plain `subClassOf` hierarchies (16,030 NCIT / 9,388
  DOID edges) which is sufficient structure. Restriction-derived edges are a backlog item (§12).
- **Per-ontology predicate map**, supplied in the dataset YAML rather than hardcoded, because the
  two ontologies use different annotation vocabularies:

  | Text role | NCIT | DOID |
  |---|---|---|
  | label | `rdfs:label` | `rdfs:label` |
  | synonym | `P90` | `oboInOwl:hasExactSynonym` |
  | definition | `P97`, `P325` | `IAO_0000115` (not universal) |

  Text variants are then assembled generically from the mapped roles: `label`,
  `label_definition`, `label_synonyms_definition`.
- `refs_parser.py` — `refs_equiv/{train,test}.tsv` (`SrcEntity`, `TgtEntity`, `Score`) →
  `AlignmentRecord(node_1_id, node_2_id, relation="equivalence", split, score)`.
  `test.cands.tsv` candidate pools are parsed into a separate optional artifact used only by
  ranking evaluation. `refs_subs/*` (subsumption) is out of scope.
- Classes absent from the pruned aligned subset are retained; a `pruning` config option can restrict
  ingestion to nodes reachable from the alignment set if the full ontologies prove unwieldy.

---

## 5. Pipelines

Five pipelines, each registered by name, each a sequence of steps that declare the artifacts they
read and write so the base class can validate inputs before work starts and write the manifest at
the end. **Only Pipeline 1 is dataset-aware**, and only through the adapter registry.

Pipelines 1–4 form the training chain; Pipeline 5 (`evaluate`) is separate because it consumes a
finished model's exported embeddings rather than the trainer's internals. Keeping it out of
`gcl_train` means changing `eval.ks` does not change `gcl_train`'s `config_hash` and therefore does
not invalidate a completed training run — and it is what lets MLflow evaluate a registered model
version, and DVC re-run evaluation as its own stage, without retraining.

### 5.1 `ingest` — dataset adapter → canonical graph artifacts

| | |
|---|---|
| **Inputs** | `data/raw/{dataset_id}/…` as declared by the adapter's config |
| **Config** | `dataset.adapter` (registry key), adapter-specific block, text-variant selection, edge relation filters, `split` fractions/seed |
| **Outputs** → `data/processed/{dataset_id}/{run_id}/` | `nodes.parquet`, `edges.parquet`, `texts.parquet`, `alignments.parquet` (empty if none), `splits.parquet`, `stats.json`, `manifest.json` |

Steps:

1. `resolve_adapter` — look up `dataset.adapter` in the registry, build it from the dataset config,
   call `validate_sources()`, record `source_fingerprint()` as manifest inputs.
2. `ingest_nodes` — drain `iter_nodes()`; assign `graph_role` from the adapter's `GraphSpec`s;
   enforce `node_id` uniqueness.
3. `ingest_edges` — drain `iter_edges()`; assert both endpoints exist and share a `graph_role`
   (cross-graph edges are a hard error); dedupe; count dropped danglers into `stats.json`.
4. `ingest_texts` — drain `iter_texts()`; enforce `(node_id, text_variant)` uniqueness; hash each
   text.
5. `ingest_alignments` — drain `iter_alignments()`; assert endpoints exist and are in opposite
   graphs.
6. `split_nodes` — assign every node a `split ∈ {train, val, test}` → `splits.parquet` (§6.5).
7. `validate_and_manifest` — schema/FK/uniqueness checks, `stats.json` (per-graph node and edge
   counts, text coverage, degree summary, alignment counts per split, node counts per split per
   graph), `LATEST`.

Node-set policy: **all** parsed entities are written. The legacy "drop description-less nodes"
filter moves downstream (embedding status + `NodeUniverse`), so exclusions are recorded with a
reason instead of silently shifting indices.

#### 5.1.1 Node splitting

Split membership is a property of the **dataset**, not of a training run: the same partition must
be used by every `training_data` run built on this ingest run, and by the `evaluate` run that
eventually scores a model trained from it. Computing it once here — rather than redrawing it inside
`training_data` — is what makes that guarantee mechanical rather than conventional, and is why
`splits.parquet` sits beside the other canonical tables in `data/processed/`.

The policy is **proportional per graph**, identical for every dataset:

- For each `graph_role` **independently**, nodes are shuffled with `seed` and partitioned into
  `train`/`val`/`test` by `split.fractions` (default `[0.8, 0.1, 0.1]`). Each graph therefore
  contributes the same ~80/10/10 proportion, so neither split is skewed toward the larger graph.
- Assignment is per **node**, not per alignment pair. A dataset that ships its own split
  (`ncit_doid`'s `refs_equiv/{train,test}.tsv`) does **not** override this — that source split stays
  recorded in `alignments.parquet.split` and is consumed separately (see below), keeping the node
  partition uniform across datasets.
- Seeded and deterministic: the same ingest config and seed produce the same partition, and the
  partition is hashed into the manifest so a downstream run can detect it changed.

`val` and `test` nodes are both **held out** from contrastive example construction (§5.3); only
`train` nodes form the sampling pool. `splits.parquet` never removes a node from `nodes.parquet` —
held-out nodes still get embeddings and still appear in `universe.parquet`, because `gcl_train`
needs their features to encode the graph and `evaluate` needs them to score.

**Interaction with dataset-provided alignment splits.** These are two independent exclusions and
both are enforced. A node is held out from sampling if it is in our `val`/`test` partition **or** it
appears in an `alignments.parquet` row whose `split` is `test`. For `attack_cwe` (no alignments) the
second clause is vacuous. For `ncit_doid` it keeps the benchmark's own test pairs genuinely unseen,
so reported Hits@k/MRR stay comparable to published OAEI Bio-ML numbers even though our node
partition was drawn independently of theirs.

### 5.2 `embed` — text and structural embedding generation

Text and graph embedders are both node→vector functions with identical storage requirements, so
they share one pipeline and one store contract; `embedder.kind` selects the implementation.

| | |
|---|---|
| **Inputs** | `texts.parquet` (text embedders) or `edges.parquet` + `nodes.parquet` (graph embedders) from an `ingest` run; optionally a fine-tuned model |
| **Config** | `embedder.kind ∈ {hf_mean_pool, deepwalk, node2vec}`, model ref, chunking/pooling, walk params, batch size, `resume` |
| **Outputs** | embedding store (§3.4) at `artifacts/embeddings/{dataset_id}/{embedder_id}/{run_id}/`; fine-tuned LMs at `models/llm/{base_model_slug}/{run_id}/` |

Steps: `finetune` (optional) → `generate` (per-node, status-recording, shard-appending) →
`materialize` (shards → `vectors.npy` + `index.parquet`) → `validate`.

`embedder_id` = `{kind}-{model_slug}[-ft:{finetune_run_id[:8]}]`, e.g. `hf-pt_SecRoBERTa`,
`deepwalk-d128-w30-n200`.

Nothing here references the dataset beyond `dataset_id` in the path.

### 5.3 `training_data` — contrastive example creation

| | |
|---|---|
| **Inputs** | one text embedding store + one graph embedding store (by `ArtifactRef`), `nodes.parquet` for `graph_role`, `splits.parquet` for the node partition, `alignments.parquet` if present |
| **Config** | sampler and parameters, universe policy, seed |
| **Outputs** → `artifacts/training_data/{dataset_id}/{run_id}/` | `universe.parquet`, `examples.parquet`, `stats.json`, `manifest.json` |

Steps:

1. `build_universe` — nodes with `status == ok` in **every** required store (policy `intersection`,
   default) → `universe.parquet` (`row`, `node_id`, `graph_role`, `split`, `held_out`). This replaces
   the legacy "graph_1 rows come first" assumption with explicit columns. `split` is carried over
   verbatim from the ingest run's `splits.parquet` (§5.1.1) — this pipeline never draws a partition
   of its own. `held_out` is the derived sampling-eligibility flag:

   ```
   held_out = (split in {val, test}) or (node appears in an alignments.parquet row with split == test)
   ```

   Both clauses are enforced; the second is vacuous for datasets without alignments. `held_out` is
   computed here, **before** similarity or sampling ever run, because that is the only point in the
   pipeline where eligibility can be decided without having already leaked the node into a
   similarity matrix.
2. `similarity` — cosine matrices: **cross-graph** in text space, **within-graph** in structural
   space, computed **only over `held_out == False` rows**. Defined by `graph_role`, not by dataset.
   A held-out node's embedding is still available (it stays in `universe.parquet` for `gcl_train`
   and `evaluate`), it is simply never a candidate row/column when `rank_topk`/`percentile` build
   their similarity-ranked neighbor lists — otherwise a node reserved for evaluating the trained
   model could be selected as an unsupervised anchor/positive/negative during training, contaminating
   the very embeddings that evaluation later scores.
3. `sample` — a registered sampler emitting rows of the examples schema (§6.6), drawing only from the
   `held_out == False` pool for `rank_topk`/`percentile`:
   - `rank_topk` — top-k/bottom-k by argsort, pairs zipped by rank (legacy sample 10)
   - `percentile` — percentile thresholds with randomized pairing (legacy sample 11), seeded
   - `supervised` — positives taken from `alignments.parquet` where the dataset provides them,
     restricted to pairs whose endpoints are both in the `train` node split and not in a
     `test`-split alignment row
4. `assign_example_split` — every example inherits the `split` of its **anchor** node, so the node
   partition and the example partition can never disagree. In practice this makes all unsupervised
   examples `train` (steps 1–3 already excluded `val`/`test` anchors from the sampling pool);
   the step exists so the invariant is asserted rather than assumed, and so a future sampler that
   deliberately emits held-out-anchor examples (e.g. for a val-loss curve) labels them correctly
   without changing the schema.
5. `validate_and_manifest` — in addition to the usual schema/FK checks, assert that no `held_out`
   node id appears in any of `examples.{anchor_id, text_pos_id, text_neg_id, graph_pos_id,
   graph_neg_id}` — the leakage check that makes step 1's exclusion enforced, not just intended.

### 5.4 `gcl_train` — graph contrastive learning

| | |
|---|---|
| **Inputs** | `examples.parquet` + `universe.parquet` from a `training_data` run; the two embedding stores resolved **via that run's manifest** (so they cannot be mismatched); `edges.parquet`; `alignments.parquet` when in-loop monitoring is on |
| **Config** | encoder (`gcn`/`gat`, dims, heads), loss (`triplet`/`jsd`/`infonce`, margin, alpha), optimizer, epochs, checkpointing, optional in-loop monitoring |
| **Outputs** → `models/gcl/{dataset_id}/{run_id}/` | `config.resolved.yaml`, `manifest.json`, `checkpoints/`, `metrics.jsonl`, `node_embeddings/` (AlignedMatrix) |

Steps:

1. `load` — resolve refs; build `NodeUniverse`; `select()` every `AlignedMatrix` onto it; remap
   `edges.parquet` and `examples.parquet` node_ids → universe rows, raising on any id outside the
   universe; build tensors.
2. `train` — full-batch loop, `alpha·L_text + (1−alpha)·L_graph`, seeded; per-epoch
   `{epoch, loss, loss_text, loss_graph, lr, elapsed_s}` appended to `metrics.jsonl`; checkpoint
   every `checkpoint_every` epochs and at the end.
3. `export` — final per-node embeddings as an `AlignedMatrix` keyed by `node_id`.
4. `manifest`, `LATEST`.

`monitor.enabled` optionally runs the same ranking code every `monitor.every` epochs and appends
`{"epoch": …, "eval": {…}}` rows to `metrics.jsonl`, for watching a run in progress. It is a
convenience, not the reported result: authoritative metrics come from Pipeline 5, which is the only
thing that writes evaluation results.

### 5.5 `evaluate` — alignment metrics and diagnostics

| | |
|---|---|
| **Inputs** | `node_embeddings/` (AlignedMatrix) from a `gcl_train` run; `nodes.parquet` for `graph_role`, `splits.parquet`, and `alignments.parquet`, all resolved **via that run's manifest chain** |
| **Config** | `gcl_run` ref, `ks`, candidate-pool usage, diagnostic toggles |
| **Outputs** → `artifacts/evaluations/{dataset_id}/{run_id}/` | `metrics.json`, `rankings.parquet` (optional per-pair detail), `manifest.json` |

Steps:

1. `load` — resolve the `gcl_train` run, read its manifest, follow `inputs[]` back to the `ingest`
   run that produced `nodes.parquet` / `splits.parquet` / `alignments.parquet`, and load the exported
   embeddings. Nothing is passed in by hand, so an evaluation cannot be run against mismatched sources.
2. `rank` — with alignments present, score each test pair (§8.4); skipped when there are none.
3. `diagnose` — unsupervised checks (§8.4), always run; reported over all nodes and over the
   `test` split separately.
4. `validate_and_manifest`.

Because it reads only the exported `AlignedMatrix`, this pipeline evaluates any saved model — an
older run, a checkpoint re-export, or eventually a model pulled from an MLflow registry — without
touching `gcl_train`.

---

## 6. Data schemas

pyarrow types; `pk` = primary key, enforced on write.

### 6.1 `nodes.parquet`

| Field | Type | Notes |
|---|---|---|
| `node_id` pk | string | `{graph_key}:{source_id}` |
| `graph_key` | string | `attack`, `cwe`, `ncit`, `doid` |
| `graph_role` | string | `graph_1` \| `graph_2` — what generic code branches on |
| `source_id` | string | identifier as it appears in the source |
| `external_id` | string | display/eval id (`T0800`, `CWE-79`, full IRI) |
| `name` | string | canonical label |
| `node_type` | string \| null | adapter-defined (`attack-pattern`, `owl:Class`, …) |
| `metadata` | string (JSON) | adapter-specific extras, kept opaque downstream |
| `has_text` | bool | any non-empty text variant |
| `source_version` | string | catalog/ontology version the record came from |
| `raw_sha256` | string | hash of the source record, for change detection |

### 6.2 `edges.parquet`

| Field | Type | Notes |
|---|---|---|
| `edge_id` pk | string | `sha1(src\|dst\|relation)` |
| `src_id`, `dst_id` | string | FK → nodes; **must share `graph_role`** |
| `graph_role` | string | denormalized for cheap per-graph filtering |
| `relation` | string | `ChildOf`, `uses`, `subClassOf`, … |
| `directed` | bool | |
| `weight` | float32 | default 1.0 |

### 6.3 `texts.parquet`

| Field | Type | Notes |
|---|---|---|
| `node_id` | string | composite pk with `text_variant` |
| `text_variant` | string | `description`, `full_metadata`, `label_definition`, … |
| `text` | string | may be `""`; never null |
| `text_sha256` | string | |
| `n_chars` | int32 | |

### 6.4 `alignments.parquet`

| Field | Type | Notes |
|---|---|---|
| `alignment_id` pk | string | `sha1(node_1_id\|node_2_id\|relation)` |
| `node_1_id` | string | FK → nodes, `graph_role == graph_1` |
| `node_2_id` | string | FK → nodes, `graph_role == graph_2` |
| `relation` | string | `equivalence` (initial scope) |
| `score` | float32 | 1.0 for gold |
| `split` | string | `train` \| `test` \| `unassigned` — **the source's own** pair split, not ours |
| `candidates` | list\<string\> \| null | ranking candidate pool (`test.cands.tsv`) |

Empty for datasets without ground truth; consumers must handle zero rows.

This `split` is a property of an **alignment pair** as the benchmark shipped it, and is distinct
from the per-**node** partition in `splits.parquet` (§6.5). Both are honored: §5.3 step 1 holds a
node out of sampling if either marks it held out.

### 6.5 `splits.parquet`

| Field | Type | Notes |
|---|---|---|
| `node_id` pk | string | FK → nodes; exactly one row per node in `nodes.parquet` |
| `graph_role` | string | denormalized; the partition is drawn per graph (§5.1.1) |
| `split` | string | `train` \| `val` \| `test` |

Written by `ingest.split_nodes` for **every** dataset, including those with no alignments. Produced
by a seeded proportional draw per `graph_role`, so each graph contributes the same train/val/test
proportions.

### 6.6 `examples.parquet`

| Field | Type | Notes |
|---|---|---|
| `example_id` pk | string | `sha1` of the five ids + sampler |
| `anchor_id` | string | FK → universe |
| `text_pos_id`, `text_neg_id` | string | **cross-graph** pair (text view) |
| `graph_pos_id`, `graph_neg_id` | string | **within-graph** pair (structural view) |
| `sampler` | string | `rank_topk` \| `percentile` \| `supervised` |
| `split` | string | `train` \| `val` \| `test` — inherited from the anchor node's `splits.parquet` row |

### 6.7 Embedding store tables

`status.parquet`: `node_id` pk, `status`, `attempts:int16`, `error:string|null`,
`input_sha256:string|null`, `n_chunks:int16|null`, `updated_at:timestamp`.
`index.parquet`: `row:int32` pk, `node_id:string` unique.
`shards/part-*.parquet`: `node_id`, `vector: list<float32>`, `input_sha256`, `n_chunks`.
`universe.parquet`: `row:int32` pk, `node_id:string` unique, `graph_role:string`,
`split:string` (carried from `splits.parquet`), `held_out:bool` (derived — `True` when
`split in {val, test}` **or** the node appears in a `test`-split `alignments.parquet` row;
see §5.3 step 1).

### 6.8 `metrics.jsonl`

One object per epoch: `{"epoch": 12, "loss": 0.41, "loss_text": 0.39, "loss_graph": 0.43, "lr": 1e-3, "elapsed_s": 0.8}`; in-loop monitoring (`gcl_train.monitor`) appends `{"epoch": …, "eval": {…}}`.
Authoritative metrics live in the `evaluate` run's `metrics.json`, not here.

### 6.9 Model manifests

`models/llm/…` adds `base_model` (+revision), `objective` (`mlm`/`clm`), `text_variant`,
`texts_run_id`, epochs, effective batch size, final loss.
`models/gcl/…` adds `training_data_run_id`, `text_embedding_run_id`, `graph_embedding_run_id`,
`universe_sha256`, `n_nodes`, `n_examples`, `final_loss`. Evaluation metrics live in the `evaluate`
run that references this model, not here. Checkpoints embed
`universe_sha256` so a checkpoint refuses to load against a different node ordering.

---

## 7. Configuration

### 7.1 Layering

```
configs/base.yaml
  + configs/datasets/{--dataset}.yaml
  + configs/pipelines/{pipeline}.yaml
  + configs/experiments/{--experiment}.yaml   (the section matching this pipeline)
  + each --config file, in order
  + each --set key.path=value
  → pydantic validation → config.resolved.yaml (+ sha256 → config_hash)
```

Deep merge, later layers win, lists replace. Interpolation limited to `${paths.<key>}` and
`${env:VAR}`. Resolution is one pure function in `core/config.py` — deliberately Hydra's composition
model minus its runtime.

### 7.2 `configs/base.yaml`

```yaml
project: cga
seed: 42
device: auto              # auto | cpu | cuda:0

paths:
  raw: data/raw
  processed: data/processed
  artifacts: artifacts
  models: models

logging:
  level: INFO
  file: true              # also write {run_dir}/run.log

run:
  name: null              # optional human alias; run_id is always generated
```

### 7.3 `configs/datasets/attack_cwe.yaml`

```yaml
dataset:
  id: attack_cwe
  adapter: attack_cwe
  root: ${paths.raw}/attack_cwe
  graphs:
    graph_1: {key: attack, display_name: "MITRE ATT&CK"}
    graph_2: {key: cwe,    display_name: "MITRE CWE"}
  adapter_config:
    attack:
      file: attack.json
      edge_from_relationships: true
      include_types: null            # null = every object with an id
    cwe:
      file: cwec_v4.12.xml
      version: cwec_v4.12
      edge_relations: [ChildOf]
  text_variants:
    description:   {attack: [description],
                    cwe: [name, description, extended_description]}
    full_metadata: {attack: [description, detection],
                    cwe: [name, description, extended_description, modes_of_introduction,
                          common_consequences, detection_methods, potential_mitigations,
                          demonstrative_examples]}
  alignments: null                   # no ground truth available
```

### 7.4 `configs/datasets/ncit_doid.yaml`

```yaml
dataset:
  id: ncit_doid
  adapter: ncit_doid
  root: ${paths.raw}/ncit_doid
  graphs:
    graph_1: {key: ncit, display_name: "NCI Thesaurus"}
    graph_2: {key: doid, display_name: "Disease Ontology"}
  adapter_config:
    ncit:
      file: ncit.owl
      id_style: fragment             # IRI fragment after '#'
      edge_predicates: [rdfs:subClassOf]
      predicate_map:                 # text role -> ontology-specific predicates
        label:      [rdfs:label]
        synonym:    [P90]
        definition: [P97, P325]
    doid:
      file: doid.owl
      id_style: fragment             # IRI fragment after the last '/'
      edge_predicates: [rdfs:subClassOf]
      predicate_map:
        label:      [rdfs:label]
        synonym:    ["oboInOwl:hasExactSynonym"]
        definition: [IAO_0000115]
    pruning:
      mode: none                     # none | alignment_reachable
  text_variants:
    label:                     [label]
    label_definition:          [label, definition]
    label_synonyms_definition: [label, synonym, definition]
  alignments:
    equivalence:
      train: refs_equiv/train.tsv
      test:  refs_equiv/test.tsv
      candidates: refs_equiv/test.cands.tsv
```

Note both dataset files have the **same shape**: identity, adapter key, per-graph adapter config,
text-variant composition, alignment sources. The generic pipelines read only the shared keys.

### 7.5 `configs/pipelines/ingest.yaml`

```yaml
ingest:
  text_variant_selection: null     # null = ingest every variant declared in dataset.text_variants
  split:
    fractions: [0.8, 0.1, 0.1]     # train / val / test, per graph_role (§5.1.1)
    seed: null                     # null = use the run's global seed
```

`split.fractions` must sum to 1.0 and is applied to **each graph independently**, so both graphs
contribute the same proportions. The same values apply to every dataset — a dataset's own
alignment-level split (`ncit_doid`) does not override this node partition; it is enforced
separately in §5.3 step 1.

### 7.6 `configs/pipelines/embed.yaml`

```yaml
embed:
  inputs:
    ingest_run: latest
    text_variant: description
  embedder:
    kind: hf_mean_pool              # hf_mean_pool | deepwalk | node2vec
    model_ref: jackaduma/SecRoBERTa # hub id, or models/llm/<slug>/<run_id>
    label: pt_SecRoBERTa
    max_length: 512
    chunking: fixed
    add_special_tokens: false       # legacy behavior; flip deliberately, not silently
    pooling: mean_last_hidden
    batch_size: 8
  finetune:
    enabled: false
    objective: auto                 # mlm for encoders, clm for gpt2-family
    epochs: 10
    lr: 1.0e-4
    per_device_batch_size: 2
    grad_accum: 4
    mlm_probability: 0.15
  on_error: continue                # continue (record failed) | abort
  resume: null
```

Structural variant (a second yaml or `--set` overrides):

```yaml
embed:
  embedder:
    kind: deepwalk
    label: deepwalk
    dimensions: 128
    walk_length: 30
    num_walks: 200
    window: 5
    epochs: 10
    workers: 1                      # >1 makes gensim non-deterministic
    p: 1.0                          # node2vec only
    q: 1.0
```

### 7.7 `configs/pipelines/training_data.yaml`

```yaml
training_data:
  inputs:
    ingest_run: latest              # supplies nodes, splits, alignments
    text_embeddings:  {embedder_id: hf-pt_SecRoBERTa, run: latest}
    graph_embeddings: {embedder_id: deepwalk-d128-w30-n200, run: latest}
  universe_policy: intersection     # intersection | text_only
  sampler:
    name: rank_topk                 # rank_topk | percentile | supervised
    k: 30
    top_pct: 0.04                   # percentile sampler
    bottom_pct: 0.04
```

No split configuration lives here: the node partition is drawn once by `ingest` (§7.5) and read
from `splits.parquet`, and each example inherits its anchor's split (§5.3 step 4).

### 7.8 `configs/pipelines/gcl_train.yaml`

```yaml
gcl_train:
  inputs:
    training_data_run: latest       # embeddings + universe resolved from its manifest
  model:
    encoder: gcn                    # gcn | gat
    hidden_dim: 128
    out_dim: 128
    gat_heads: [8, 1]
    dropout: 0.0
  loss:
    name: triplet                   # triplet | jsd | infonce
    margin: 1.0
    alpha: 0.5                      # text-view weight
    temperature: 0.1                # infonce only
  optim: {name: adam, lr: 1.0e-3, weight_decay: 0.0}
  train:
    epochs: 400
    checkpoint_every: 50
    log_every: 10
  monitor:
    enabled: false                  # in-loop progress metrics only, appended to metrics.jsonl;
    every: 50                       # authoritative results come from `cga run evaluate`
    ks: [1, 5, 10]
```

### 7.9 `configs/pipelines/evaluate.yaml`

```yaml
evaluate:
  inputs:
    gcl_run: latest                 # nodes/alignments resolved from its manifest chain
  ranking:
    enabled: true                   # no-ops when the dataset has no alignments
    ks: [1, 5, 10]
    use_candidate_pool: true        # rank within test.cands.tsv when the dataset supplies it
    also_full_target_set: true      # and over every graph_2 node
  diagnostics:
    silhouette: true                # over graph_role
    similarity_distribution: true
    neighbor_samples: 50            # nearest cross-graph neighbours dumped for inspection
  write_rankings: false             # per-pair detail to rankings.parquet
```

### 7.10 `configs/experiments/attack_cwe_secroberta_gcn.yaml`

```yaml
# Each top-level key overlays the matching pipeline config.
embed:
  embedder: {kind: hf_mean_pool, model_ref: jackaduma/SecRoBERTa, label: pt_SecRoBERTa}
training_data:
  sampler: {name: rank_topk, k: 30}
gcl_train:
  model: {encoder: gcn}
  loss:  {name: triplet, margin: 1.0, alpha: 0.5}
evaluate:
  ranking: {ks: [1, 5, 10]}
```

---

## 8. Cross-cutting components

### 8.1 CLI

```
cga list [pipelines|adapters|embedders|samplers]
cga run <pipeline> --dataset <id> [--experiment <name>] [--config f.yaml ...]
                   [--set key.path=value ...] [--run-name NAME] [--resume RUN_ID] [--dry-run]
cga validate <artifact_dir>        # schema + hash + alignment checks
cga inspect  <artifact_dir>        # print manifest and walk the upstream chain
cga config show <pipeline> [...same layering flags]
```

`--dry-run` resolves config and inputs, prints the planned run directory and input hashes, and
exits — cheap insurance before a long embedding job.

### 8.2 RunContext

Created by the CLI, passed to every step: `run_id`, `run_dir`, `dataset_id`, resolved immutable
`config`, `seed`, `device`, a `logger` writing to console and `{run_dir}/run.log`, and a `manifest`
accumulator (`record_input` / `record_output`, hashing on record). `ctx.resolve(ArtifactRef)` turns
`latest`/run_id refs into concrete directories **and** records them as inputs — one call, so a
pipeline cannot read an artifact it failed to declare.

### 8.3 Validation

- **Config**: pydantic v2 per pipeline and per adapter, `extra="forbid"` so typos fail loudly; enums
  for every choice list; cross-field checks.
- **On write**: schema enforcement, pk uniqueness, FK checks (`edges.{src,dst} ⊂ nodes`,
  `examples.*_id ⊂ universe`, `alignments.node_*_id ⊂ nodes` with correct roles), sha256 recorded.
- **On read**: manifest present and `status == success`; hashes match; `AlignedMatrix` row count ==
  index count; checkpoint `universe_sha256` matches.
- `cga validate` runs the read-side checks standalone.

### 8.4 Evaluation

Pipeline 5 (§5.5), entirely generic — it reads an exported `AlignedMatrix`, `nodes.parquet` for
`graph_role`, `splits.parquet` for the node partition, and `alignments.parquet`, and knows nothing
about any dataset:

- **With alignments** (`ncit_doid`): for each test pair, rank `graph_2` candidates by cosine
  similarity to the `graph_1` node — over the full `graph_2` node set, and over the supplied
  candidate pool when present. Reports Hits@k and MRR. Uses `alignments.parquet` only; knows nothing
  about ontologies.
- **Without alignments** (`attack_cwe` today): unsupervised diagnostics only — silhouette score over
  `graph_role`, cross-graph similarity distribution, nearest-cross-graph-neighbor samples dumped for
  manual inspection.

Diagnostics are reported **both** over all nodes and restricted to the `test` split, so a dataset
with no alignment ground truth still gets a number computed on nodes the contrastive objective
never saw (§5.1.1). Both write `artifacts/evaluations/{dataset_id}/{run_id}/metrics.json`; the
ranking block is absent when `alignments.parquet` is empty. `gcl_train.monitor` may run the same
ranking code in-loop for progress visibility, but only this pipeline writes the reported numbers.

### 8.5 Logging, reproducibility, error handling

- stdlib `logging`, configured once by the CLI; console + per-run file handler; pipeline/step names
  in the format string; progress every `log_every` items rather than progress bars.
- `seed_everything(seed)` at run start (`random`, `numpy`, `torch` CPU/CUDA, `PYTHONHASHSEED`).
  Known residual non-determinism is **declared** in the manifest (`env.deterministic: false` with a
  reason: gensim `workers > 1`, non-deterministic GAT kernels) rather than hidden.
- Typed exceptions (`ConfigError`, `MissingInputError`, `ArtifactValidationError`); the CLI catches,
  logs, marks the manifest `status: failed`, exits non-zero. Per-item failures inside
  `embed.generate` are recorded, not raised (`on_error: continue`).

### 8.6 Testing

- `tests/unit/` — `node_id` construction/parsing; config layering and overrides;
  `AlignedMatrix.select` (missing id, reorder); `EmbeddingStore` resume semantics (failed row
  re-runs, `ok` row untouched); each sampler on a toy matrix; each loss against hand-computed
  values; **each adapter against a handful of committed source snippets** (a few STIX objects, a few
  `owl:Class` blocks, a few TSV rows) asserting the canonical records they produce; Hits@k / MRR on a
  toy ranking and the zero-alignment no-op path.
- `tests/integration/` — each pipeline end-to-end on `attack_cwe` with `epochs: 2`, asserting
  manifests, schemas, shapes, and referential integrity.

---

## 9. End-to-end data flow

```
data/raw/{dataset_id}/…            (attack.json + cwec.xml  |  ncit.owl + doid.owl + refs_equiv/)
                    │
                    ▼
        ┌───────── cga run ingest --dataset <id> ──────────┐
        │  resolve_adapter → registry[dataset.adapter]      │   << only dataset-aware step
        │  data/processed/{dataset_id}/{run}/               │
        │    nodes.parquet       (node_id, graph_role, …)   │
        │    edges.parquet       (intra-graph only)         │
        │    texts.parquet       (node_id, variant, text)   │
        │    alignments.parquet  (may be empty)             │
        │    splits.parquet      (train/val/test per node)  │
        └───────┬───────────────────────────┬──────────────┘
          texts │                           │ edges + nodes
                ▼                           ▼
   cga run embed (hf_mean_pool)     cga run embed (deepwalk)
   [optional finetune → models/llm/]
   artifacts/embeddings/{ds}/hf-…/{run}/     artifacts/embeddings/{ds}/deepwalk-…/{run}/
     status.parquet vectors.npy index.parquet  status.parquet vectors.npy index.parquet
                │                           │
                └─────────────┬─────────────┘
                              ▼
                   cga run training_data      ← also reads splits.parquet
              artifacts/training_data/{ds}/{run}/
                universe.parquet   ← NodeUniverse (id ↔ row, graph_role, split, held_out)
                examples.parquet   ← anchor/pos/neg by node_id, train-split anchors only
                manifest.json      ← records both embedding run_ids
                              │
                              ▼
                     cga run gcl_train
                models/gcl/{ds}/{run}/
                  config.resolved.yaml, manifest.json
                  checkpoints/, metrics.jsonl
                  node_embeddings/{vectors.npy, index.parquet}   ← research output
                              │
                              ▼
                     cga run evaluate
            artifacts/evaluations/{ds}/{run}/
              metrics.json        ← Hits@k / MRR, or unsupervised diagnostics
              rankings.parquet    ← optional per-pair detail
              manifest.json       ← records the gcl_train run it scored
```

Identity flows as `node_id` strings through every arrow. Positions are computed exactly once, in
`training_data.build_universe`, written down (`universe.parquet`), and re-validated by
`gcl_train.load` before any tensor exists. Everything to the right of `ingest` is dataset-agnostic.

---

## 10. Adding a new dataset

1. Create `src/cga/adapters/<name>/` with an adapter implementing the five iterators and
   `@register_adapter("<name>")`.
2. Create `configs/datasets/<name>.yaml` following the shape in §7.3–7.4.
3. Place source files under `data/raw/<name>/`.
4. Add a unit test asserting the adapter's canonical output on committed source snippets.
5. Run `cga run ingest --dataset <name>`, then the remaining four pipelines unchanged.

No change to `cga/core/`, `cga/data/`, or any of the five pipelines.

---

## 11. Seams for later tooling

| Tool | What it will own | Why the current design fits |
|---|---|---|
| **MLflow** | experiment tracking, model registry | `RunContext` is the single place a run starts and ends — an `MlflowTracker` mirrors `manifest.json` params and `metrics.jsonl` rows; `models/llm` and `models/gcl` become logged artifacts, and the standalone `evaluate` pipeline (§5.5) already scores a model from its exported embeddings alone, which is how a registered model version is evaluated. No pipeline code changes. |
| **Hydra** | config composition, sweeps | `configs/` is already grouped (`datasets/`, `pipelines/`, `experiments/`) with the same override semantics; `core/config.resolve()` is the only function to swap. Interpolation is kept to what OmegaConf supports. |
| **DVC** | data/artifact versioning, stage DAG | `data/`, `artifacts/`, `models/` are separate roots with immutable run dirs; each pipeline's declared inputs/outputs map 1:1 onto `dvc.yaml` stages (`cga run <p>` is the `cmd`). `LATEST` files become DVC-tracked. Evaluation being its own stage means re-scoring does not invalidate a completed training stage. |
| **PySpark** | scaling similarity/sampling to full ontologies | Every record artifact is Parquet with a declared schema and string keys — readable by Spark with no conversion; `similarity` and `sample` are pure functions over tables, swappable behind the same `Step` interface. |

Guardrails that keep these open: no pickle between pipelines; no experiment parameters encoded in
filenames; one config-resolution function; `RunContext` as the only tracking hook; all node
references by `node_id`.

---

## 12. Backlog

Deliberately excluded from the initial implementation; none require architectural change to add.

- **OWL logical axioms** — deriving edges from NCIT `owl:equivalentClass` / `intersectionOf` /
  `Restriction` blocks. Plain `subClassOf` is sufficient structure to start.
- **Subsumption alignment** (`refs_subs/*`) — a second `relation` value in `alignments.parquet`.
- **Ontology pruning** — `pruning.mode: alignment_reachable` for `ncit_doid` if full-size ingestion
  proves too slow.
- **Hop-distance features** — the legacy `text_hop` feature and its similarity matrices; no live
  consumer exists.
- **Artifact GC** — `cga gc --keep-latest N --keep-referenced`, once disk pressure appears.
- **Early stopping / LR scheduling** in `gcl_train`.
- **Structured JSON logs**, source-data download helper, richer similarity-report plots.
- **Demo UI** — the legacy Gradio app, re-homed on the new artifact stores.
