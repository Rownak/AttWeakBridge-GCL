# `evaluate` architecture — design recommendation

> **Status:** recommendation. Reviews `evaluate_redesign_plan.md` against `target_architecture.md`
> §5.5, §5.5a, §8.4. Supersedes the plan's Steps 5–7 and 9; keeps its Steps 1–4 and 8 nearly intact.

## Verdict

**Yes — add an `evaluations` layer above views, and yes to long-form metric rows.** But the reason is
stronger than "silhouette might be nice later," and it changes where the layer belongs.

`target_architecture.md` §8.4 **already commits** `evaluate` to three distinct evaluation shapes:

| §8.4 commitment | Loop shape | Ground truth |
|---|---|---|
| Alignment ranking (`ncit_doid`) | pairs × ks | `alignments.parquet` |
| Neighborhood preservation (§5.5a) | methods × views × roles × ks | derived reference |
| Silhouette over `graph_role`, similarity distributions | methods × roles × (k-clusters) | none |

The redesign plan's extension point — `relevance_providers`, a registry of ground-truth *set*
providers — only varies the ground truth of **one** of these three shapes. The other two cannot be
expressed as a `RelevanceProvider` at all: silhouette has no query and no relevant set; alignment
ranking is cross-role and pair-driven. So the plan, as written, would land a `pipeline.py` that has
to hardcode a second and third step anyway — exactly the rigidity its Step 7 rewrite exists to
remove. The abstraction is one level too low.

The fix is small: keep the plan's design *inside* a neighborhood evaluator, and make the evaluator
itself the registered unit.

---

## What should remain

Everything outside `evaluate/`. No change is recommended to:

- **`RunContext` / `Manifest` / `ArtifactRef`** — `record_input`/`record_output`, `LATEST`,
  `cga inspect` lineage. Already correct and already the MLOps backbone §11 depends on.
- **`AlignedMatrix`, `NodeUniverse`, Parquet-with-declared-schemas** (§3). The method axis needs
  nothing new here.
- **The `Registry` class** (`core/registry.py`). Generic, decorator-based, `ConfigError` lists known
  names. Add registry instances, never a new mechanism.
- **Config layering** (§7.1) and the `--experiment` flag. `deep_merge` replacing lists wholesale is
  exactly right for overlay-supplied `methods`/`evaluations` lists.
- **Pipeline-5-is-separate** (§5). Keeping `evaluate` out of `gcl_train` so `ks` changes don't
  invalidate a training run stays the single best decision in the current design.

From the redesign plan, keep with minor edits:

- **Step 1** (set-based metrics, both `normalized_precision_at_k` and `recall_at_k`) — correct.
- **Step 3** (`RelevanceProvider`, `HopRadiusRelevance`, `CosineTopKRelevance`) — correct, but it
  becomes an internal detail of the neighborhood evaluator, not a pipeline-level concept.
- **Step 4** (`MethodSpec` / `load_method`, references reuse `MethodSpec`) — correct and central.
  Promote it: it is shared by *all* evaluators, so it belongs beside the pipeline, not inside
  neighborhood.
- **Step 8** (`model_id` partition of `models/gcl/`) — a hard prerequisite, independent, do it first.
- Dropping the `pooling` step; `bert-base-uncased` as the generic reference.

---

## What should change

### 1. The registered unit is an *evaluator*, not a relevance provider

```python
class Evaluator(Protocol):
    """One evaluation method: owns its own loop and emits tidy metric rows."""
    def evaluate(self, res: EvalResources, params: dict) -> list[MetricRow]: ...
```

`EvalResources` is loaded once by the pipeline and passed to every evaluator: the loaded methods
(`dict[method_id, AlignedMatrix]`), their specs, `nodes`/`edges`/`splits`, the `nx.Graph`, and
`ctx`. Evaluators are `RunContext`-free apart from resolving their own extra references through
`load_method`, so each unit-tests in isolation.

Three evaluators cover §8.4 completely:

| `evaluations[].kind` | Owns | Replaces |
|---|---|---|
| `neighborhood` | methods × views × roles × ks; `RelevanceProvider` registry internally | plan Steps 5–7 core |
| `cluster_quality` | methods × roles × `n_clusters`; silhouette | §8.4 promise, unbuilt |
| `alignment_ranking` | test pairs × ks; Hits@k, MRR | §5.5 step 2, Phase 1.9 |

`relevance_providers` still exists — one level down, scoped to `neighborhood`. That is where it was
always the right abstraction.

### 2. Long-form metric rows, one schema

The plan's `SUMMARY_SCHEMA` is neighborhood-specific: fixed `k`,
`normalized_precision_at_k`, `hit_at_k`, `mrr` columns. Silhouette has none of those, so a wide
schema forces either null-padding that grows with every evaluation or one parquet per evaluator
(and no single table to pivot). Use long form:

```
metrics.parquet  — one row per (evaluation, method, scope, metric)
```

| column | type | notes |
|---|---|---|
| `dataset_id`, `run_id` | string | |
| `evaluation_id`, `evaluation_kind` | string | `evaluation_id` is the config-supplied instance name |
| `method_id`, `method_label`, `method_kind`, `method_run_id` | string | `method_run_id` = lineage per row |
| `graph_role` | string | nullable — cross-role evaluations set `null` |
| `view_id` | string | nullable — `null` for `cluster_quality` |
| `param_name`, `param_value` | string, int32 | the sweep axis: `("k", 10)`, `("n_clusters", 8)`; nullable |
| `metric_name` | string | `normalized_precision_at_k`, `silhouette`, `mrr`, … |
| `metric_value` | float64 | |
| `n_queries` | int32 | nullable |

One `MetricRow` dataclass, one schema, one writer. A new evaluator adds **zero** schema change —
this is the property that makes "add a new evaluation easily" actually true, and it is why long form
beats wide here despite being slightly less pleasant to eyeball.

Pivoting stays one line: `df.pivot_table(index="param_value", columns="method_label",
values="metric_value")` after filtering `evaluation_id` + `metric_name`.

Keep `metrics.json` as a **human-readable digest** derived from the same rows (nested
`evaluation → view → role → method → param`), plus per-`(evaluation, scope)` provenance:
`query_set_sha256`, `n_excluded_undefined`, `mean_relevant_set_size`. Parquet is the machine
contract; JSON is for reading and for `cga inspect`. Do not let the two be computed independently —
derive JSON from the rows so they cannot drift.

### 3. Hop distance computed on the fly, not as an artifact

Radius-bounded BFS (`cutoff=2`) touches the 2-hop ball (~50–200 nodes), not the graph — ~600 small
BFS per role, well under a second. More importantly, the relevance set is defined over the
**method-intersected candidate pool**, which changes with the method list; a precomputed matrix
would be keyed to one universe and silently wrong for another. On-the-fly is both cheaper and more
correct. (If unbounded distances are ever needed, that reverses — it becomes an `ingest`-adjacent
artifact. Noted in the backlog, not built.)

### 4. `evaluate` stops walking the `gcl_train` manifest chain

With N methods there is no canonical chain, and a generic-text reference is in no training lineage.
Resolve `ingest` directly from config; each method and each reference resolves itself via
`load_method` and lands in `manifest.inputs[]` as `method:<id>` / `reference:<view_id>`. Lineage
gets *stronger*, not weaker: every row carries `method_run_id`.

### 5. Drop `rankings.parquet` from §5.5's promise

~290k rows of re-derivable data no figure consumes. Mark **not implemented**; a documented omission
beats an undelivered promise. Per-pair detail, if ever needed, is an `alignment_ranking` option.

---

## Pipeline structure

Four steps, none of which know any evaluation kind:

1. **`load`** — validate config eagerly (duplicate ids, unknown `kind`/`provider`, empty sweeps);
   resolve `ingest`; read `nodes`/`edges`/`splits`; build the graph; `load_method` every method and
   every evaluator-declared reference. Fail before any compute.
2. **`evaluate`** — for each configured evaluation: `registry.get(kind)` → `evaluate(res, params)`
   → extend `rows`. The only loop in the pipeline.
3. **`visualize`** — optional, per method, `plots/tsne_{method_id}.png`. Returns nothing; can never
   be mistaken for a metric.
4. **`validate_and_manifest`** — write `metrics.parquet` (schema-enforced) + derived `metrics.json`,
   record outputs, log the pivot, write `LATEST`.

## Artifact layout

```
artifacts/evaluations/{dataset_id}/{run_id}/
    metrics.parquet      # long-form rows — the machine contract
    metrics.json         # nested digest derived from the rows
    manifest.json        # inputs: ingest + method:<id> + reference:<view_id>
    plots/tsne_{method_id}.png
models/gcl/{dataset_id}/{model_id}/{run_id}/   # Step 8: model_id partition
```

Unchanged from §5.5 apart from `metrics.parquet` replacing `rankings.parquet`, and the `model_id`
level. No experiment parameters in filenames (§11 guardrail) — `evaluation_id` lives in the rows.

## Configuration model

`configs/pipelines/evaluate.yaml` — thin, no paper specifics:

```yaml
evaluate:
  inputs: {ingest_run: latest}
  methods: []
  evaluations: []
  visualize: {enabled: false, perplexity: 25.0, methods: []}
```

`configs/experiments/neighborhood_compare.yaml` — a named evaluation via the **existing**
`--experiment` flag (no CLI change):

```yaml
evaluate:
  methods:
    - {method_id: deepwalk,   label: "DeepWalk",          kind: embedding_store, embedder_id: deepwalk-deepwalk,          ref: latest}
    - {method_id: node2vec,   label: "node2vec",          kind: embedding_store, embedder_id: node2vec-node2vec,          ref: latest}
    - {method_id: secroberta, label: "SecRoBERTa",        kind: embedding_store, embedder_id: hf_mean_pool-pt_SecRoBERTa, ref: latest}
    - {method_id: gcn,        label: "Dual-encoder GCN",  kind: gcl_run,         model_id: gcn,                           ref: latest}
    - {method_id: gat,        label: "Dual-encoder GAT",  kind: gcl_run,         model_id: gat,                           ref: latest}
  evaluations:
    - evaluation_id: neigh
      kind: neighborhood
      params:
        ks: [1, 5, 10, 20]
        roles: [graph_1, graph_2]
        views:
          - {view_id: structural_r1, provider: hop_radius,   params: {radius: 1}}
          - {view_id: structural_r2, provider: hop_radius,   params: {radius: 2}}
          - {view_id: textual,       provider: cosine_top_k,
             params: {top_k: 10, reference: {kind: embedding_store, embedder_id: hf_mean_pool-bert_base, ref: latest}}}
    - evaluation_id: clusters
      kind: cluster_quality
      params: {roles: [graph_1, graph_2], n_clusters: [2, 4, 8, 16, 32]}
  visualize: {enabled: true, methods: [gcn, gat]}
```

Two rules keep this additive:

- **`params` is opaque.** Handed verbatim to the evaluator's factory, never inspected by the
  pipeline. This is the single property that makes a new evaluator a new file.
- **Validation lives with the evaluator**, one pydantic model per kind (§8.3, `extra="forbid"`), so
  typos fail at `load` with actionable messages and the pipeline holds no per-kind knowledge.

`evaluations` being a **list of instances** (not a dict keyed by kind) means the same kind can appear
twice with different params — `neighborhood` at `top_k: 10` and `top_k: 25` in one run, distinguished
by `evaluation_id`. That is the sensitivity check the redesign plan lists as an open question, for
free.

---

## Adding things with minimal change

| Task | Change required |
|---|---|
| **New baseline / model** | One line in `methods:`. Zero code. (`gcl_run` needs its `model_id`.) |
| **New metric in an existing evaluator** | Append `MetricRow`s with a new `metric_name`. Zero schema change, zero config change. |
| **New evaluation method** | One file in `pipelines/evaluate/evaluators/`, `@evaluators.register("...")`, one pydantic params model, one test. Zero change to `pipeline.py`, schemas, or CLI. |
| **New named evaluation** | One file in `configs/experiments/`. Zero code. |
| **New dataset** | Unchanged from §10 — adapter + dataset config + raw files. `evaluate` is dataset-agnostic: `alignment_ranking` is simply not listed for a dataset without alignments, instead of being a runtime `if` inside the pipeline. This is a genuine improvement over §5.5 step 2's "skipped when there are none." |

## Sequencing

1. **Step 8** (`model_id` partition) — independent prerequisite; GCN/GAT clobber each other today.
2. `MetricRow` + `METRICS_SCHEMA` + long-form writer + derived JSON digest.
3. Set-based `metrics.py` (plan Step 1) and `methods.py` (plan Step 4, promoted to shared).
4. `neighborhood` evaluator with the `relevance_providers` registry inside it (plan Steps 3, 5).
5. `pipeline.py` rewrite against the `Evaluator` protocol.
6. `cluster_quality` — the proof the layer works, and it closes §8.4's silhouette promise.
7. Docs: `target_architecture.md` §5.5/§5.5a/§8.4, sprint plan, execution log.

Steps 2–5 are the redesign plan's scope re-cut; step 6 is the new capability and should be written
by a different person or on a different day than step 5, because a layer that only ever has one
implementation is not a layer.

## Costs, stated plainly

- **Long form is less readable raw** than a wide table. Mitigated by the JSON digest and the logged
  pivot, not by widening the parquet.
- **One more indirection** (`Evaluator` above `RelevanceProvider`). Justified only because §8.4
  already names three shapes; it would be over-engineering for one.
- **`param_name`/`param_value` is a typed-column compromise** — `int32` covers `k` and `n_clusters`.
  An evaluator needing a float sweep axis adds `param_value_f64`, or the axis goes in
  `evaluation_id`. Worth revisiting if that case arrives; not worth a generic variant column now.
