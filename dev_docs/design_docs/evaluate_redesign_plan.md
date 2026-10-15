# Comparative neighborhood-preservation `evaluate` pipeline — design

> **Status:** proposed design, not yet implemented. Supersedes the Phase 1.8 `evaluate` pipeline
> described in `target_architecture.md` §5.5a. Phase 1.9 (alignment ranking, §5.5) is unaffected and
> slots into the same machinery.

## Context

Phase 1.8 shipped an `evaluate` pipeline that scores **one** learned embedding against **two**
embedding-space references. The paper needs something different: a **comparison table** showing that
the dual-encoder models preserve neighborhoods better than the baselines they are built from.

That means four methods as *rows* — DeepWalk, domain text (SecRoBERTa), dual-encoder GCN, dual-encoder
GAT — scored against two ground-truth *views*: shortest-path structure, and generic BERT text
similarity.

The current pipeline cannot express this:

- `_EvaluateState` (`pipelines/evaluate/pipeline.py`) has exactly three named slots
  (`learned_matrix`, `text_matrix`, `structural_matrix`).
- `NeighborhoodStep` hardcodes `reference_by_view = {"text": ..., "structural": ...}` and rejects
  anything else.
- `LoadStep` resolves exactly one `gcl_train` run and walks its manifest chain to find the stores
  *that run was trained on* — which structurally cannot reach a generic-text store that was never in
  any training lineage.
- `metrics.json` is shaped `view → role → k`, with no method axis.

The outcome we want: one command produces a `summary.parquet` that pivots directly into the paper's
figures, plus generic machinery so a future view is a new file rather than a refactor.

---

## Design decisions

| Decision | Choice |
|---|---|
| Structural relevance | **Hop-radius set**, not top-k. Relevant = all nodes within radius r. Report r=1 and r=2. |
| Textual relevance | Top-k by cosine over a **generic BERT** store (`bert-base-uncased`), fixed `top_k`. |
| Metrics under set relevance | **normalized precision@k** = `\|L_k ∩ R\| / min(k,\|R\|)`; **recall@k** = `\|L_k ∩ R\| / \|R\|`; `Hit@k` = any relevant in top-k; `MRR` = reciprocal rank of *first* relevant. |
| Query universe | **Per-view**: nodes scoreable by *every* method ∩ defined under that view. Isolates drop from structural only. |
| Method selection | **Explicit `methods:` list** in config; partition `models/gcl/` by `model_id` so GCN/GAT get separate `LATEST`. |
| Named evaluations | **`configs/experiments/` overlays** on a thin `configs/pipelines/evaluate.yaml`. No CLI change. |
| Hit@k saturation | Compute all ks; choose non-saturated columns at figure time. |
| Text encoder | `bert-base-uncased` — works with the **existing** mean-pool path, so **no `embed` code change**. |

### Why hop-radius rather than top-k for the structural view

Hop distance is integer-valued and massively tied: a node's "10 nearest by shortest path" is usually
2 nodes at distance 1 plus an arbitrary 8-of-40 slice of the distance-2 ring. Any top-k definition
therefore makes relevance an artifact of the tie-break rule. Defining relevance as *the set of all
nodes within radius r* removes the arbitrariness entirely — no tie-break is needed, because no
ordering is needed.

### Metric naming

The set-based denominator `min(k,|R|)` makes this not plain precision, so name it explicitly in code,
`metrics.json`, and `summary.parquet`:

- **`normalized_precision_at_k`** — `|L_k ∩ R| / min(k, |R|)`. Recall-bounded: when `|R| < k` the
  denominator is `|R|`, so a method that retrieves every relevant item scores 1.0 rather than being
  penalized for `k` exceeding the relevant set.
- **`recall_at_k`** — `|L_k ∩ R| / |R|`. For the textual view at `k > top_k` this is the honest name
  for the quantity; emit it as its own column rather than mislabeling it precision.

Emit **both** for every cell. They coincide when `k ≤ |R|`, and where they diverge the reader can see
exactly why. No footnote needed, because neither column is mislabeled.

### The key abstraction

The scoring loop never learns whether relevance is a set or a ranked list. All metrics are pure set
operations, so a `RelevanceProvider` returns `set[str]` and nothing else. The textual provider
consumes rank *internally* to pick its top-k, then discards it.

Deliberately **no `is_ranked` flag** on the interface — that would force the scoring loop to branch
and reintroduce the two code paths the abstraction exists to remove. (If a graded-relevance metric
like NDCG is ever needed, add a separate optional `graded()` method; do not contaminate the primary
path now.)

---

## Implementation

### Step 1 — `src/cga/pipelines/evaluate/metrics.py` (modify)

Generalize to relevance **sets**:

- `normalized_precision_at_k(learned_ranked, relevant: set[str], k)` → `|L_k ∩ R| / min(k,|R|)`;
  0.0 if `k==0` or `R` empty.
- `recall_at_k(learned_ranked, relevant, k)` → `|L_k ∩ R| / |R|`.
- `hit_at_k(learned_ranked, relevant, k)` → `1.0 if set(L_k) & R else 0.0`.
- `mrr(learned_ranked, relevant)` → reciprocal rank of first `L[i] ∈ R`, else 0.0.

Keep `rank_by_similarity` / `top_k`. Add `top_k_by_similarity` using `np.argpartition` — the hot path.
No back-compat shim; these are internal pure functions with three call sites.

> Note the behavioral change: today's `hit_at_k`/`mrr` use only `reference_top_k_ids[0]` (a single
> target). The set-based versions are strictly more lenient. Do not compare new textual numbers
> against previously computed ones.

### Step 2 — `src/cga/core/registry.py` (modify)

Add `relevance_providers = Registry("relevance provider")` alongside the existing registries. One
line. Views get registry dispatch (free `ConfigError` with known-names on typo; a future view
registers by import). **Methods do not** — two `kind`s both ending in `AlignedMatrix.load` is a
15-line `if/elif`, and a registry there is ceremony.

### Step 3 — `src/cga/pipelines/evaluate/relevance.py` (new)

```python
class RelevanceProvider(Protocol):
    view_id: str
    def prepare(self, *, candidate_ids: list[str]) -> None: ...
    def defined_for(self) -> set[str]: ...          # where this view has non-empty ground truth
    def relevant(self, query_id: str) -> set[str]: ...
```

Contract: `prepare` is called once per (view, role) with the final frozen candidate list; providers
precompute there. `defined_for()` returns the subset with meaningful non-empty ground truth — this is
the hook that expresses "structural excludes isolates" without the pipeline hardcoding that rule.
`relevant()` excludes the query itself and is guaranteed ⊆ `candidate_ids`.

- **`HopRadiusRelevance(graph, radius)`** — `prepare` takes `graph.subgraph(candidate_ids)` (same-role
  containment becomes structural), then `nx.single_source_shortest_path_length(sub, n, cutoff=radius)`
  per node. `defined_for` = nodes with a non-empty set, so the ~55 attack / ~25 cwe **isolates fall
  out with no isolate-specific code anywhere**.
- **`CosineTopKRelevance(matrix, top_k)`** — one normalized `X @ X.T`, diagonal `-inf`, `argpartition`
  then a k-element sort with the `(-sim, node_id)` tiebreak.

Reuse `build_graph` from `cga.pipelines.embed.graph_embedders` (graph_embedders.py:141) — it already
builds an undirected `nx.Graph` including isolates. Do **not** write a second graph builder.
`networkx>=3.3` is already a dependency.

`RunContext`-free so it unit-tests in isolation. Providers built by registered factories taking
`(params, resources)`.

### Step 4 — `src/cga/pipelines/evaluate/methods.py` (new)

`MethodSpec` dataclass (`method_id`, `kind`, `ref`, `embedder_id`/`model_id`, `label`) and one
`load_method(ctx, spec) -> AlignedMatrix`:

- `embedding_store` → `artifacts/embeddings/{dataset}/{embedder_id}/` + `ref`
- `gcl_run` → `models/gcl/{dataset}/{model_id}/` + `ref`, then `/node_embeddings`

Both go through `ctx.resolve(f"method:{id}", ...)` so all four land in `manifest.inputs[]` with
resolved run_ids. **A view's `reference` block reuses `MethodSpec`**, so one loader and one
traceability path serve both methods and references.

`LoadStep` stops walking the `gcl_train` manifest chain: with four methods there is no canonical
chain, and picking one method's chain would be arbitrary. `ingest` (for `nodes.parquet` /
`edges.parquet`) resolves directly from `evaluate.inputs.ingest_run`.

### Step 5 — `src/cga/pipelines/evaluate/scoring.py` (new; replaces `neighborhood.py`)

`score_cell(...)` for one (method, view, role); `score_all(...)` owning the loops, eligibility, and
`query_set_sha256`. Delete `neighborhood.py` — its name encodes the old two-fixed-views assumption.

Eligibility per (view, role), in this exact order:

```
candidates = sorted(role_ids ∩ (∩ over methods of method.node_ids))
provider.prepare(candidate_ids=candidates)
queries    = sorted(candidates ∩ provider.defined_for())
```

Two correctness points that each deserve a dedicated test:

1. `prepare` receives the **method-intersected** candidates, not raw role ids — otherwise hop-radius
   counts as relevant a neighbor that no method can score.
2. The intersection applies to the **candidate pool**, not just the query set. If method B lacks node
   `x`, then `x` must leave *every* method's pool — otherwise A ranks against 600 candidates and B
   against 598, and precision is pool-size sensitive.

### Step 6 — `src/cga/data/schemas.py` (modify)

Add `SUMMARY_SCHEMA` next to the existing schemas.

### Step 7 — `src/cga/pipelines/evaluate/pipeline.py` (rewrite)

New `_EvaluateState`: `methods: dict[str, AlignedMatrix]`, `method_specs`, `view_plans`,
`graph: nx.Graph`, `graph_role_by_node`, `rows`, `view_stats`.

Steps: `load` (validate config eagerly — duplicate ids, unknown `kind`/`provider`, empty `ks` — then
resolve ingest → read nodes/edges → build graph → load methods + view references → construct
providers) → `neighborhood` (`score_all`) → `visualize` (loop, `plots/tsne_{method_id}.png`) →
`validate_and_manifest` (`metrics.json` + `summary.parquet`, record both, log the pivot via
`ctx.logger`).

### Step 8 — `gcl_train` `model_id` partition

`src/cga/pipelines/gcl_train/pipeline.py` + `src/cga/cli.py:_run_root` (lines 62-63), mirroring
`embed`'s `embedder_id`. `build_model_id` returns `model_cfg.get("model_id") or model_cfg["encoder"]`,
so the default is `gcn`/`gat` with no required config change.

Today `models/gcl/{dataset}/LATEST` is a single flat file overwritten by every successful run, so a
GAT run silently clobbers the GCN pointer — which makes a four-method comparison impossible to
express with `ref: latest`.

> **Migration:** existing runs at `models/gcl/{dataset}/{run_id}/` become invisible under the new
> `models/gcl/{dataset}/{model_id}/{run_id}/` layout. Re-run both trainings — a GAT run is needed
> anyway. No compatibility shim for a research repo.

Independent of Steps 1–7; can be done first.

### Not needed: making the `pooling` config key live

An earlier draft used BGE as the generic-text reference, which requires CLS pooling and would have
meant wiring up the (currently dead) `pooling` key in `configs/pipelines/embed.yaml`.

Using `bert-base-uncased` instead removes that entirely: the **existing** `mean_last_hidden` +
`add_special_tokens: false` path is correct for it, and matches how SecRoBERTa is already embedded —
which also makes the domain-vs-generic contrast cleaner, since both sides are pooled identically.

No change to `text_embedders.py`, `embed/pipeline.py`, or `configs/pipelines/embed.yaml`. Producing
the reference is pure CLI:

```bash
cga run embed --dataset attack_cwe \
  --set embed.embedder.model_ref=bert-base-uncased \
  --set embed.embedder.label=bert_base
```

→ `artifacts/embeddings/attack_cwe/hf_mean_pool-bert_base/`.

> Known no-op: `pooling`, `batch_size`, and `chunking` in `configs/pipelines/embed.yaml` are never
> read by any code. Out of scope here; worth a separate cleanup.

### Step 9 — Config: thin default + named evaluation overlays

The CLI resolves `configs/pipelines/{pipeline_name}.yaml` by pipeline name, so a named evaluation
cannot be its own top-level pipeline file. Use the **existing** `--experiment` layering instead —
and note `deep_merge` (`core/config.py:26-38`) **replaces lists wholesale**, so an overlay setting
`evaluate.methods` fully replaces the default list. That is exactly the semantics wanted.

`configs/pipelines/evaluate.yaml` — thin defaults only:

```yaml
evaluate:
  inputs: {ingest_run: latest}
  methods: []                   # named evaluations supply these
  views: []
  neighborhood:
    ks: [1, 5, 10, 20]
    roles: [graph_1, graph_2]
    common_query_set: false     # true intersects queries across views for a strictly-comparable variant
  visualize: {enabled: false, perplexity: 25.0, methods: []}
```

`configs/experiments/neighborhood_compare.yaml` — the paper's table:

```yaml
evaluate:
  methods:
    - {method_id: deepwalk,   label: "DeepWalk (graph)",  kind: embedding_store, embedder_id: deepwalk-deepwalk,          ref: latest}
    - {method_id: secroberta, label: "SecRoBERTa (text)", kind: embedding_store, embedder_id: hf_mean_pool-pt_SecRoBERTa, ref: latest}
    - {method_id: gcn,        label: "Dual-encoder GCN",  kind: gcl_run,         model_id: gcn,                           ref: latest}
    - {method_id: gat,        label: "Dual-encoder GAT",  kind: gcl_run,         model_id: gat,                           ref: latest}
  views:
    - {view_id: structural_r1, provider: hop_radius,   label: "Graph (1-hop)", candidate_scope: same_role, params: {radius: 1}}
    - {view_id: structural_r2, provider: hop_radius,   label: "Graph (2-hop)", candidate_scope: same_role, params: {radius: 2}}
    - {view_id: textual,       provider: cosine_top_k, label: "Text (BERT)",   candidate_scope: same_role,
       params: {top_k: 10, reference: {kind: embedding_store, embedder_id: hf_mean_pool-bert_base, ref: latest}}}
  visualize: {enabled: true, methods: [gcn, gat]}
```

Run as `cga run evaluate --dataset attack_cwe --experiment neighborhood_compare`. Future evaluations
(ablations, alignment ranking) are additional files in the same directory, no code change.

`params` is opaque — handed verbatim to the registered factory, never inspected by the pipeline. That
is what keeps a future view additive. `candidate_scope` exists now (~10 lines) because alignment
relevance is cross-role; adding it later would be a refactor.

### Step 10 — Docs

`target_architecture.md` §5.5a: the method axis, set-based relevance and the two precision namings,
the `model_id` partition, and amend the §5.5 outputs table to mark `rankings.parquet` **not
implemented** (≈290k rows of largely re-derivable data that nothing in the paper consumes — a
documented omission beats an undelivered promise). Then `sprints/sprint_1.md` and an execution log.

---

## Outputs

**`metrics.json`** — `n_queries` hoists from the per-k leaf up to the (view, role) level, because it
is identical across methods and ks by construction; leaving it at the leaf invites the exact
apples-to-oranges bug the design prevents. `query_set_sha256` proves the methods really shared a query
set; `mean_relevant_set_size` makes the r=1 vs r=2 denominator difference legible.

```json
{"schema_version": 2,
 "neighborhood": {"<view_id>": {"<role>": {
    "n_eligible_queries": 611, "n_excluded_undefined": 55,
    "mean_relevant_set_size": 3.2, "query_set_sha256": "…",
    "methods": {"<method_id>": {"<k>": {
        "normalized_precision_at_k": 0.41, "recall_at_k": 0.33,
        "hit_at_k": 0.78, "mrr": 0.55}}}}}}}
```

**`summary.parquet`** — one tidy long-format row per `(method, view, role, k)`; ~96 rows. Columns:

| column | type |
|---|---|
| `dataset_id`, `evaluate_run_id` | string |
| `method_id`, `method_label`, `method_kind`, `method_run_id` | string |
| `view_id`, `view_label`, `graph_role` | string |
| `k`, `n_queries` | int32 |
| `mean_relevant_set_size` | float64 |
| `normalized_precision_at_k`, `recall_at_k`, `hit_at_k`, `mrr` | float64 |

Parquet over CSV: it is what every other artifact here is, `write_parquet` already enforces schemas
(`core/io.py`), and floats round-trip exactly. Answer the "I want to eyeball it" need by logging the
pivot table, not by degrading the artifact.

Paper figure: `pd.read_parquet(...)` → `seaborn.lineplot(x="k", y="normalized_precision_at_k",
hue="method_label")`, no reshaping.

---

## Performance

24 cells (4 methods × 3 views × 2 roles) at n≈600/role. Three changes, ~3–5 lines each, taking the
sweep from ~3s of pure-Python sorting to well under a second — worth doing up front since this gets
re-run many times while tuning the paper:

1. **Cache learned similarity** by `(method_id, role)` — otherwise recomputed once per view, 3×
   redundant.
2. **`np.argpartition`** for learned top-k and inside `CosineTopKRelevance.prepare` — takes 10–20 of
   ~600 elements, O(n + k log k) in C vs O(n log n) in Python.
3. **MRR via `np.lexsort((np.arange(n), -sim_row))`** then `argmax` on the relevance mask — gives the
   exact `(-sim, node_id)` tie order in C with no Python tuple sort. MRR is the one metric that can
   need a deep rank, so it cannot use `argpartition`.

Memory is negligible (~22 MB peak).

---

## Tests (`tests/unit/pipelines/evaluate/`)

- **`test_metrics.py`** (rewrite) — the `min(k,|R|)` denominator (`|R|=2, k=5` → 1.0, not 0.4) is the
  single most important new test; `normalized_precision_at_k` and `recall_at_k` coincide when
  `k ≤ |R|` and diverge exactly when `k > |R|`; `hit_at_k` fires on *any* relevant, not just `R`'s
  first; `mrr` returns first-relevant; empty `R` → zeros; `top_k_by_similarity` agrees with
  `rank_by_similarity(...)[:k]` across ~50 seeds **including exact ties** (proves the argpartition
  optimization is behavior-preserving).
- **`test_relevance.py`** (new) — path graph `a-b-c-d`: r=1 `b→{a,c}`, r=2 `b→{a,c,d}`, r=2 ⊇ r=1;
  isolate ∉ `defined_for()`; a node whose only neighbor is outside `candidate_ids` is treated as an
  isolate (proves same-role containment is structural); cosine ties break by node_id; query never in
  its own relevant set; unknown provider → `ConfigError`.
- **`test_scoring.py`** (new) — identical query set + identical `query_set_sha256` across methods when
  one method is missing nodes; provider candidates are method-intersected; a method that *is* the
  textual reference scores `normalized_precision_at_k`=1.0 at k=top_k; clique role → Hit@k=1; role
  with <2 nodes zeroes rather than crashing; determinism.
- **`test_methods.py`** (new) — both `kind`s resolve to the right layout; both record a `method:<id>`
  manifest input; unknown `kind` → `ConfigError`; missing `LATEST` → `MissingInputError`.
- **`test_pipeline.py`** (rewrite) — extend the fabricated-artifact helpers to write two embedding
  stores, two gcl runs under `gcl/{dataset}/{gcn,gat}/`, a BERT store, and `edges.parquet`. Assert
  `summary.parquet` has exactly `4*3*2*len(ks)` rows matching `SUMMARY_SCHEMA`, that it agrees
  numerically with `metrics.json` on a sampled cell (guards the two serializers drifting), duplicate
  `method_id` → `ConfigError`, an experiment overlay fully replaces the default `methods` list, one
  PNG per visualized method, determinism.
- **`gcl_train/test_pipeline.py`** (modify) — `LATEST` lands under `{model_id}/`; a `gcn` run and a
  `gat` run do not clobber each other. That regression is the entire reason for Step 8.

No `embed` test changes — the pooling step was dropped.

---

## Verification

```bash
uv run pytest tests/unit -q                     # full suite

# Produce the two missing inputs (no code change needed for either)
uv run cga run embed --dataset attack_cwe \
  --set embed.embedder.model_ref=bert-base-uncased \
  --set embed.embedder.label=bert_base
uv run cga run gcl_train --dataset attack_cwe                                  # gcn (re-run post-migration)
uv run cga run gcl_train --dataset attack_cwe --set gcl_train.model.encoder=gat

uv run cga run evaluate --dataset attack_cwe --experiment neighborhood_compare
uv run cga inspect artifacts/evaluations/attack_cwe/<run_id>
```

Confirm: `summary.parquet` has 96 rows covering all 4 methods × 3 views × 2 roles × 4 ks;
`query_set_sha256` identical across the four methods within each (view, role); manifest inputs list
all four methods plus the BERT reference plus ingest; re-running yields a byte-identical
`summary.parquet`; and the dual-encoder rows beat the baseline rows on at least the structural views
(the paper's actual claim).

---

## Known caveats to state in the paper

1. **Hit@k saturates** at k ≥ 5 in the textual view (|R|=10) — expect ~1.0 for all methods, since
   "any of 10 relevant in the top-10" is easy. Use Hit@1 / Hit@5 in the figure; the artifact keeps
   every k so the choice is deferred to figure time.
2. **r=1 and r=2 use effectively different denominators** — r=1 sets (|R|≈2-5) almost always hit the
   `min(k,|R|)` branch while r=2 (|R|≈50-200) reverts to `/k`, so the two columns are not on a common
   scale. `mean_relevant_set_size` is in both artifacts so a reader can interpret this. Likely the
   first reviewer question about the table.
3. **Query populations differ per view** (~611 structural vs ~666 textual in attack) because isolates
   are excluded only where they are undefined. Correct trade-off — maximizing power within each view
   beats a lowest-common-denominator query set — but state it in the caption.
   `common_query_set: true` offers a strictly-comparable variant if a reviewer insists.
4. **Both precision columns are reported.** `normalized_precision_at_k` and `recall_at_k` coincide for
   `k ≤ |R|`. Quote whichever the venue expects; neither is mislabeled, so no footnote is needed.
5. **The structural graph is undirected.** Every existing consumer symmetrizes edges (`build_graph`,
   `gcl_train`'s `edge_index`, and the legacy hop script), and no edge has its reverse present, so
   shortest path is computed on the undirected graph for consistency.

---

## Open questions for refinement

- **Radius choice.** r=1 and r=2 are proposed. Is r=3 worth reporting, or does the r=2 set already
  approach a large fraction of the giant component?
- **Textual `top_k = 10`** is a free parameter. It sets the relevance-set size and therefore where
  `normalized_precision_at_k` and `recall_at_k` diverge. Worth a sensitivity check.
- **Whether to compute distances on the full ingest graph or the universe-restricted graph.** They
  differ; the universe-restricted one is what the model actually saw. Current plan uses the
  method-intersected candidate set, which is the universe-restricted reading.
- **`common_query_set`** defaults to `false`. Confirm that is the variant the paper reports.
