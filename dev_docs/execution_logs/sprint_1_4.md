# Sprint 1.4 Execution Log — `embed` pipeline

## Summary
Built `src/cga/pipelines/embed/`: `text_embedders.py` (`hf_mean_pool`, model-agnostic HF
mean-pooling), `graph_embedders.py` (`deepwalk`/`node2vec`, ported from
`legacy/GCL/get_graph_features/2_node2vec_deepwalk.py`), and `pipeline.py` (`EmbedPipeline`:
`generate` → `materialize` → `validate_and_manifest`, with `--resume`). Both `configs/pipelines/
embed.yaml` (default: `hf_mean_pool` / `jackaduma/SecRoBERTa`) and the structural-embedder variant
`configs/pipelines/embed_deepwalk.yaml` are in place.

## Pre-implementation check: legacy GPT-2 embedding concern
Before starting, the user flagged that vanilla GPT-2 hidden states are known to be highly
anisotropic and a poor fit for semantic similarity/clustering — raw cosine similarity between
unrelated passages tends toward 1.0 for decoder-only, next-token-trained models, unlike encoders
with a supervised/contrastive pretraining objective (BERT-family, sentence-transformers). Checked
this against both the legacy code and the target architecture doc before writing anything:
- **Legacy** (`legacy/LLM_embeddings/2_extract_embeddings.py`): GPT-2/GPT2-XL was one of several
  interchangeable `model_id` options (alongside SecureBERT, SecRoBERTa) through one shared
  mean-pool-last-hidden-layer routine — never the sole or default choice.
- **Target architecture** (`dev_docs/target_architecture.md` §7.5, §7.9): `configs/pipelines/
  embed.yaml`'s default `embedder.model_ref` is already `jackaduma/SecRoBERTa` (an MLM-trained
  RoBERTa-family encoder), and the `attack_cwe_secroberta_gcn` experiment config locks that in.
  GPT-2 is never the default anywhere in the design.

Resolution (confirmed with the user via AskUserQuestion): keep `hf_mean_pool` fully model-agnostic
via `AutoModel`/`AutoTokenizer` + `model_ref` — no GPT-2-specific special-casing, no default change.
This required no design changes; `text_embedders.py` was written generic from the start.

## Implementation

**`pipelines/embed/text_embedders.py`**:
- `chunk_tokens(token_ids, chunk_size)` — splits a token id list into fixed-size chunks; an empty
  input returns `[[]]` (one empty chunk) so the embed loop's "no chunks" case degenerates cleanly.
- `HfMeanPoolEmbedder` (dataclass) — loads `AutoTokenizer`/`AutoModel` once from `model_ref` in
  `__post_init__`, then `.embed(text)` tokenizes, chunks, forward-passes each chunk with
  `output_hidden_states=True`, mean-pools the last hidden layer over the sequence dimension per
  chunk, then means across chunks — matching the legacy `retrieveEmbeddings` algorithm exactly, but
  generalized to any encoder via `AutoModel` (legacy branched on `model_name` string comparisons
  against a hardcoded model list). Empty text short-circuits to a zero vector without a forward pass.

**`pipelines/embed/graph_embedders.py`** (ported from
`legacy/GCL/get_graph_features/2_node2vec_deepwalk.py`):
- `generate_random_walks` / `generate_biased_random_walks` — uniform and 2nd-order biased
  (Grover & Leskovec) walk generation, unchanged from legacy except operating on string `node_id`s
  instead of integer row indices (the legacy script indexed nodes by array position via
  `attack_weak_range.json`; the canonical `node_id` replaces that entirely, per
  target_architecture.md §1.2/§3.3).
- `deepwalk_embed` / `node2vec_embed` — walks → `gensim.Word2Vec(sg=1, seed=1)` → `{node_id: vector}`
  dict. `workers` defaults to `1` (not the legacy `4`): gensim's multi-threaded training is
  non-deterministic even with a fixed seed, and target_architecture.md §7.5 explicitly calls this
  out (`workers: 1  # >1 makes gensim non-deterministic`).
- `build_graph(node_ids, edges)` — builds an undirected `networkx.Graph` with every `node_id`
  explicitly added (not only inferred from edges), so isolated nodes still appear with degree zero
  and the pipeline can record them `status = skipped` per §3.4 rather than silently dropping them.

**`pipelines/embed/pipeline.py`** — `EmbedPipeline` (`@pipelines.register("embed")`), three steps
per §5.2 (the optional `finetune` step is deferred to Phase 1.8 per the sprint plan):
1. `GenerateStep` — resolves the `ingest_run` `ArtifactRef`, reads `nodes.parquet` for the full node
   set, opens an `EmbeddingStore` at `ctx.run_dir` (or a prior run's directory when `embed.resume` is
   set), and computes `pending_nodes()`. Dispatches to `_generate_text` (`hf_mean_pool`) or
   `_generate_graph` (`deepwalk`/`node2vec`) by `embedder.kind`.
   - Text path: loads `texts.parquet` filtered to the configured `text_variant`; a node absent from
     that table gets `missing_text`, an empty-string text gets `empty_text`, a successful embed
     appends a shard and records `ok` with `input_sha256` (hash of the source text, ties the vector
     to exactly what produced it per §3.4) and `n_chunks`; an exception is caught **per node**,
     logged, and recorded `failed` — the loop continues (`on_error: continue`), matching §3.4's "the
     legacy `try/except: break` truncation is gone."
   - Graph path: builds the full graph via `build_graph`, runs the walk+Word2Vec embedder **once**
     for the whole graph (walks aren't per-node work), then per node: `graph.degree(node_id) == 0` →
     `skipped`; otherwise `ok` with `input_sha256` over the sorted edge list (so a changed edge set
     invalidates the store, matching the text path's staleness detection). A failure during walk
     generation itself (not per-node) marks every pending node `failed` and returns, since there's no
     smaller unit of work to isolate.
2. `MaterializeStep` — thin wrapper over `EmbeddingStore.materialize()` (built in Phase 1.2).
3. `ValidateAndManifestStep` — writes `stats.json` (`n_total`/`n_ok`/`n_failed`/`n_missing_text`/
   `n_empty_text`/`n_skipped`), records `vectors`/`index`/`status`/`stats` as manifest outputs (with
   `shape`/`dtype` on `vectors`), sets `manifest.summary`.

`build_embedder_id(embedder_cfg)` — `{kind}-{label}[-ft:{finetune_run_id[:8]}]` per §5.2, with
`label` defaulting to a slugified `model_ref` when not explicitly set (`slugify` replaces `/` and
spaces with `_`, since embedder ids are used as directory names).

`EmbedPipeline.run()` overrides the base class to write `LATEST` into
`artifacts/embeddings/{dataset_id}/{embedder_id}/` after a successful run, mirroring
`IngestPipeline.run()`'s pattern from Phase 1.3.

**`cli.py`** — added `_run_root(pipeline_name, config)`, replacing the inline `run_root` computation
in `cmd_run`. `embed`'s output directory depends on `embedder_id`, which is only known after the
embedder config is resolved (unlike `ingest`, whose output path is a static `paths.processed`
lookup), so `_run_root` special-cases `embed` by calling `build_embedder_id` on the resolved
`embed.embedder` block; every other pipeline keeps the original `_PIPELINE_OUTPUT_ROOT` lookup. This
means `ctx.run_dir` **is** the embedding store's run directory directly (no extra nesting) — the
embed pipeline never constructs its own store path independently of what the CLI already resolved.

**`configs/pipelines/embed.yaml`** — matches target_architecture.md §7.5 exactly: `hf_mean_pool` /
`jackaduma/SecRoBERTa`, `max_length: 512`, `add_special_tokens: false`, `on_error: continue`.

**`configs/pipelines/embed_deepwalk.yaml`** — the structural-embedder variant, per §7.5's "a second
yaml or `--set` overrides" — loaded via `cga run embed --dataset attack_cwe --config
configs/pipelines/embed_deepwalk.yaml` (deep-merges over the base `embed.yaml` since `build_config`
always loads the pipeline default first, then any `--config` extras).

## Tests
`tests/unit/pipelines/embed/` (20 new tests):
- `test_text_embedders.py` (3) — `chunk_tokens` splits a long sequence into fixed-size chunks, a
  short sequence is one chunk, an empty sequence returns one empty chunk.
- `test_graph_embedders.py` (5) — isolated nodes keep degree zero in `build_graph`; `deepwalk_embed`
  and `node2vec_embed` are reproducible under a fixed `np.random.seed`; embedding dimension matches
  the configured `dimensions`.
- `test_pipeline.py` (12) — `build_embedder_id` (default-label-from-model_ref, explicit label,
  finetune-run-id suffix); end-to-end runs against hand-built ingest-shaped fixtures (not the real
  adapters — mirrors `tests/unit/pipelines/ingest/test_pipeline.py`'s approach) with a monkeypatched
  `HfMeanPoolEmbedder` (`_FakeEmbedder`, avoiding a real model download in unit tests): a node with
  text embeds `ok`, an empty-text node gets `empty_text`, a node with no text row gets `missing_text`,
  and `materialize()`'s output excludes both (so `index.parquet` never shifts for a bad node, per the
  checklist's explicit ask); an embedder that raises on one specific text records that node `failed`
  and the run still succeeds with the other nodes `ok`; `LATEST` is written pointing at the completed
  run; a `deepwalk` run over a graph with one isolated node marks that node `skipped` and the other
  two `ok`.

Combined with Phases 1.0–1.3: `uv run pytest tests/unit -q` → **129 passed**.

## End-to-end verification against real data
```
cga run ingest --dataset attack_cwe                                    → 1,217 nodes / 1,922 edges
cga run embed --dataset attack_cwe --config configs/pipelines/embed_deepwalk.yaml
    → 1,137 ok / 80 skipped (isolated nodes) / 1,217 total, vectors.npy shape (1137, 128)
cga run embed --dataset attack_cwe   (default hf_mean_pool / jackaduma/SecRoBERTa)
    → 1,214 ok / 3 empty_text / 0 failed / 1,217 total, vectors.npy shape (1214, 768)
```
The `hf_mean_pool` run loaded `jackaduma/SecRoBERTa` via `AutoModel` — the transformers load report
shows only `lm_head.*` weights as "unexpected" (correctly dropped, since only encoder hidden states
are used, not the MLM head) and the `RobertaModel` backbone loads cleanly. This is the real-data
confirmation of the pre-implementation design check above: the default embedder is the RoBERTa-family
encoder, not GPT-2, and the run produced zero per-node failures. Both runs' `status.parquet` show
every node accounted for (`n_ok + n_skipped == n_total` for deepwalk, `n_ok + n_empty_text ==
n_total` for hf_mean_pool); `cga validate`/`cga inspect` were not re-run against embed output
specifically since those commands' checks (in `cmd_validate`) currently only read the four `ingest`
tables — extending `cga validate` to embedding stores is naturally deferred to whichever later phase
first needs it (not called for by this phase's checklist).

## Files Created
- `src/cga/pipelines/embed/__init__.py`, `text_embedders.py`, `graph_embedders.py`, `pipeline.py`
- `configs/pipelines/embed.yaml`, `configs/pipelines/embed_deepwalk.yaml`
- `tests/unit/pipelines/embed/__init__.py`, `test_text_embedders.py`, `test_graph_embedders.py`,
  `test_pipeline.py`

## Files Modified
- `src/cga/pipelines/__init__.py` (also imports `embed` now, registering `EmbedPipeline`)
- `src/cga/cli.py` (`_run_root` helper factored out of `cmd_run`, special-casing `embed`'s
  `embedder_id`-dependent output path)
- `dev_docs/sprints/sprint_1.md` (Phase 1.4 items marked complete)

## Notes / assumptions
- **Verify line's reference shape `(1136, 768)` vs. actual `(1137, 128)` for deepwalk / pending count
  for hf_mean_pool:** the sprint checklist's Phase 1.4 verify step quotes the *legacy* node count
  (1,136) and a text-embedding hidden size (768). Phase 1.3 already established that the new
  node-set policy keeps all 1,217 parsed nodes (not just edge-reachable/text-bearing ones), so any
  embedder run's `n_total` is 1,217, and the "ok" count depends on the embedder: deepwalk marks the
  80 edge-isolated nodes `skipped` (1,137 ok), while `hf_mean_pool` marks text-less nodes
  `missing_text`/`empty_text` instead of skipping. This mirrors the Phase 1.3 log's resolution of the
  same discrepancy for `ingest`'s node/edge counts — not a bug, the architecturally-intended
  behavior per §5.1/§3.4.
- `EmbedPipeline`'s `--resume` support re-opens a prior run's store directory in place
  (`ctx.run_dir.parent / resume_from`) rather than creating a new run directory that copies forward
  old shards; this matches §3.4's "`cga run embed --resume <run_id>` re-processes only non-`ok` rows
  ... re-materializes" — resuming is explicitly an in-place continuation of the *same* run directory,
  not a new sibling run. `LATEST` still only gets updated on success, so a resumed-and-still-failing
  run doesn't get promoted.
- `on_error: abort` is implemented (re-raises out of the per-node/per-walk try/except) but not
  covered by a dedicated test — the checklist's test bullet asks for "a raising embedder records
  `failed` and continues," which is the `on_error: continue` (default) path; `abort` was implemented
  for config completeness matching `configs/pipelines/embed.yaml`'s documented values.
- Graph-embedder `input_sha256` is computed over `repr(sorted(edges))`, not a file hash — there is no
  single "the edges file" the way text embedding has one text string per node; hashing the resolved
  edge list is the closest equivalent for staleness detection (a changed edge set for the same node
  set produces a different hash).
- `cli.py`'s `_run_root` imports `build_embedder_id` from `cga.pipelines.embed.pipeline` lazily
  (inside the function, not at module top) to avoid a two-way import between `cli.py` and the
  pipelines package growing awkward as more pipelines gain path-dependent-on-config output dirs;
  revisit if a third pipeline needs the same treatment.

## CLI to verify
```
uv run pytest tests/unit -q
uv run cga run ingest --dataset attack_cwe
uv run cga run embed --dataset attack_cwe --config configs/pipelines/embed_deepwalk.yaml
uv run cga run embed --dataset attack_cwe
```

## Next Phase
Phase 1.5: `training_data` pipeline — `pipelines/training_data/similarity.py` (cosine matrices,
cross-graph text + within-graph structural), `samplers.py` (`rank_topk`/`percentile`/`supervised`),
`pipeline.py` (`build_universe` → `similarity` → `sample` → `split` → `validate_and_manifest`),
`configs/pipelines/training_data.yaml`.
