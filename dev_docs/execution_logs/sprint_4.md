# Sprint 4 — Supervised `ncit_doid` run on the gold alignments (execution log)

Plan: [sprint_4.md](../sprints/sprint_4.md).

## Phase 4.0 — Supervised dataset config and ingest

### Task 1 — `ncit_doid_supervised.yaml`

Created [configs/datasets/ncit_doid_supervised.yaml](../../configs/datasets/ncit_doid_supervised.yaml) as a copy of
`ncit_doid.yaml`. It differs in exactly three places (checked with `diff --strip-trailing-cr`):
- a header comment pointing to Sprint 4 that records the required `--set` for the node split;
- `dataset.id: ncit_doid_supervised`;
- the `alignments.equivalence` block (`train`, `test`, `candidates`) is enabled in place of Sprint 3's
  commented-out block.

`adapter`, `root` (`data/raw/ncit_doid`), `adapter_config` (including NCIT `edge_sources`) and `text_variants` are
unchanged. The file uses CRLF line endings, like its sibling.

`ncit_doid.yaml`, the `ncit_doid_*` sweep, experiment and plot configs, and the Sprint 3 logs were **not** edited.

### Task 2 — Dry-run check

`cga run ingest --dataset ncit_doid_supervised --dry-run` resolves (it prints only the config keys). Paths were
checked with `cga config show ingest --dataset ncit_doid_supervised --set "ingest.split.fractions=[1.0,0.0,0.0]"`:
- `dataset.id = ncit_doid_supervised`, so the run goes to `data/processed/ncit_doid_supervised/`;
- `ingest.split.fractions = [1.0, 0.0, 0.0]`;
- all three `refs_equiv` files (`train.tsv`, `test.tsv`, `test.cands.tsv`) exist under the resolved root.

### Task 3 — Ingest (all-train node split)

```
uv run cga run ingest --dataset ncit_doid_supervised --set "ingest.split.fractions=[1.0,0.0,0.0]"
```
- Run `20261014T185110Z-c44babf2`, about 10 s. `data/processed/ncit_doid_supervised/LATEST` points to it.
- `data/processed/ncit_doid/LATEST` is still `20261011T134334Z-6d45b8af` (untouched).

### Task 4 — `stats.json` check

| Check | Expected | Got |
|---|---|---|
| `n_alignments` | 4686 | 4686 |
| `n_alignments_by_split` | train 1406, test 3280 | train 1406, test 3280 |
| node split | all `train` | NCIT 15,762 train, DOID 8,465 train |
| `n_nodes` / `n_edges` | 24,227 / 32,874 | 24,227 / 32,874 (NCIT 23,486, DOID 9,388 edges; 0 dangling) |
| `n_texts` | — | 72,681 (every node has text) |

Compared with Sprint 3 ingest `20261011T134334Z-6d45b8af`, the new ingest is identical on:
- the `node_id` order;
- the `(node_id, text_variant, text_sha256)` rows;
- the sorted edge list.

### Task 5 — Reuse the embedding stores

Copied each run directory, plus its `LATEST` file, from `artifacts/embeddings/ncit_doid/<embedder_id>/` to
`artifacts/embeddings/ncit_doid_supervised/<embedder_id>/`:

| Store | Source run (copied, same run id) | Rows | Validity check against the new ingest |
|---|---|---|---|
| `hf_mean_pool-sapbert` | `20261009T182919Z-95011340` | 24,227 | `input_sha256` matches `label_synonyms_definition` `text_sha256` for 24,227/24,227 nodes; index order = ingest order |
| `hf_mean_pool-bert_base` | `20261009T191700Z-1278c23a` | 24,227 | same, 24,227/24,227 |
| `deepwalk-deepwalk` | `20261011T134446Z-d5aa89ed` | 24,221 | `input_sha256` matches the hash of the new sorted edge list; 6 nodes `skipped` (isolated) |

- For all three stores, the copied `vectors.npy`, `index.parquet` and `status.parquet` are byte-identical to their
  sources.
- **Lineage caveat.** The copied manifests still describe the `ncit_doid` runs:
  - bert_base's manifest names ingest `20261009T164439Z-2d34c6da`; deepwalk's names `20261011T134334Z-6d45b8af`.
  - sapbert's `manifest.json` holds only the vector-hash block (`sha256_vectors`/`sha256_index`/shape), as in the
    source.
  - Nothing downstream reads `dataset_id` or `inputs` from a store manifest. `data/aligned.py` checks only
    `sha256_vectors`/`sha256_index` when they are present, and those still match. The node/text/edge identity
    checks above are what make the copies valid for the new ingest.

### Task 6 — Gold pairs vs. the `intersection` universe

- **0 gold pairs touch a node outside the universe.** None of the 6 isolated NCIT nodes without a DeepWalk vector
  (`C12218`, `C20181`, `C22187`, `C22188`, `C43431`, `C97325`) appears in any gold pair, so every pair can be
  trained on or evaluated.
- Held-out nodes from test endpoints: 6,515 (NCIT 3,245, DOID 3,270).
- 23 train pairs touch a test endpoint, which leaves **1,383 usable train pairs** for Phase 4.1.
- Test candidates:
  - every test row has 101 candidates, and the true target is always among them;
  - every candidate id exists in the ingest.

### Files

- Created: `configs/datasets/ncit_doid_supervised.yaml`, `dev_docs/execution_logs/sprint_4.md`.
- New artifacts (not tracked by git):
  - `data/processed/ncit_doid_supervised/20261014T185110Z-c44babf2/`;
  - `artifacts/embeddings/ncit_doid_supervised/{hf_mean_pool-sapbert,hf_mean_pool-bert_base,deepwalk-deepwalk}/`
    (about 650 MB copied).
- No code changes.

### Notes

- The plan's convention line named the log `sprint4.md`. This log uses `sprint_4.md` as requested, and the plan was
  updated to match.
- `cga run ... --dry-run` does not print output paths, so `cga config show` was used to check the resolved paths.

## Phase 4.1 — Fix the `supervised` sampler

### What changed

[samplers.py](../../src/cga/pipelines/training_data/samplers.py):
- `_topk_bottomk_indices(row, k, *, exclude_index=None, exclude_indices=None)`: both kinds of exclusion are merged
  into one mask and dropped before the stable argsort. With no exclusions the code path is the old one, so the
  `rank_topk`/`percentile` tests pass unchanged.
- New `supervised_positives(alignments, text_g1_g2, graph_g1) -> (pairs, n_skipped)`:
  - A positive is a `train` row whose anchor is a row of both `text_g1_g2` and `graph_g1` and whose target is a
    column of `text_g1_g2`. Rows failing that (endpoint held out or outside the universe) are counted as skipped.
  - Duplicate pairs are kept once, so no `example_id` is duplicated.
  - The sampler and `stats.json` both use this helper, so their counts always agree.
- `supervised_sampler` was rewritten (same signature, same `ExampleTuple`). For each positive `(a, p)`, with
  `n = negatives_per_positive` (default 5):
  - **Text negatives:** the bottom-n of `a`'s `text_g1_g2` row from `_topk_bottomk_indices`, with
    `exclude_indices` = the columns of all of `a`'s gold targets (any split). Held-out targets are already absent
    from the matrix.
  - **Graph pos/neg:** the top-n/bottom-n of `a`'s `graph_g1` row with self excluded (the `rank_topk` rule).
    Negative j is zipped with structural rank j.
  - Rows per positive = `min(n, eligible text negatives, other graph_1 nodes)`.
  - Sampling is deterministic. The seed only drives the final `random.Random(seed).shuffle`.
  - Index lookups and sets are built once, outside the loop.
  - Anchors are NCIT only, because the gold files are NCIT→DOID.

[pipeline.py](../../src/cga/pipelines/training_data/pipeline.py):
- `SampleStep` passes **all** alignment rows (with `split`) to the sampler instead of pre-filtering to trainable
  pairs. Isolation still holds structurally: held-out nodes are absent from the matrices, and the
  `ValidateAndManifestStep` leakage assertion is unchanged.
- `_SAMPLER_ID_PARAMS["supervised"] = ("negatives_per_positive",)`, so the id is
  `supervised-negatives_per_positive5`.
- `stats.json` gains `n_gold_train_pairs`, `n_gold_train_pairs_used` and `n_gold_train_pairs_skipped` for
  supervised runs only, so other samplers' stats are unchanged. These satisfy `used + skipped + duplicates = train
  rows`; the gold set has no duplicates.
- The module docstring and the `SampleStep` comment were updated.

[configs/pipelines/training_data.yaml](../../configs/pipelines/training_data.yaml): added
`sampler.negatives_per_positive: 5`.
- `build_sampler_id` only appends params present in the config, so without this key the id would be the bare
  `supervised`.
- Other samplers ignore the key, and their sampler ids are unchanged. It does change the resolved-config hash of
  new runs (not existing paths).

### Tests

`uv run pytest tests/unit -q` → **389 passed**.

New and updated tests in `tests/unit/pipelines/training_data/`:
- [test_samplers.py](../../tests/unit/pipelines/training_data/test_samplers.py):
  - `_gold_views` fixture: anchor `a:1` has two gold targets (train `b:1`, test `b:5`), and `b:5` is its *least*
    text-similar column. `a:2` has two train targets. The train pair `(a:9, b:3)` shares the held-out endpoint
    `a:9` with a test pair.
  - `_topk_bottomk_indices`: `exclude_indices` (alone and with `exclude_index`) never returns an excluded column,
    and the no-exclusion behavior is unchanged.
  - Split isolation: no tuple contains `a:9`/`b:9`, and `(a:9, b:3)` is skipped.
  - Negative validity: every `text_neg_id` is cross-graph, not held out, and not gold in either split. `b:5` is
    skipped in favor of the next column, and the negatives come out in rank order.
  - Exactly n rows per positive, capped at the number of eligible negatives. Output is identical for the same
    seed, and the same multiset for different seeds.
  - Graph pos/neg are `graph_1` nodes, never the anchor, and match the structural ranking.
- [test_pipeline.py](../../tests/unit/pipelines/training_data/test_pipeline.py):
  - `test_supervised_split_keeps_test_alignment_out_of_train_examples` was updated to expect 2 rows (one per
    eligible negative).
  - New end-to-end test: no example column contains a test endpoint; negatives are non-gold `doid:` nodes; the
    `stats.json` counts are 3 train / 2 used / 1 skipped.
  - New `build_sampler_id` test for `supervised-negatives_per_positive5`.
  - The empty-alignments no-op and the val-split exclusion tests were kept unchanged and still pass.

### Smoke check on real data (in memory, no artifacts written)

The sampler was run directly on the Phase 4.0 ingest and stores, with SapBERT text, DeepWalk graph and the same
held-out rule:
- Matrices: `text_g1_g2` 12,511 × 5,195 and `graph_g1` 12,511 × 12,511.
- Gold train pairs: 1,406, of which 1,383 were used and 23 skipped (matches Phase 4.0). Result: **6,915 examples**
  (5 per positive) in 1.3 s.
- Asserted:
  - no held-out node in any column;
  - every `text_neg_id` is a DOID node and not a gold pair with its anchor;
  - graph pos/neg are NCIT nodes other than the anchor;
  - no duplicate `(anchor, pos, neg)`;
  - identical output for the same seed.
- **Observation:** only 520 distinct DOID nodes appear as text negatives. The bottom-5 concentrates on a small set
  of "far" DOID nodes shared by many anchors. This is the expected easy-negative effect already listed in the
  plan's caveats. Keep it in mind when reading the Phase 4.4 results.

### Files

Modified:
- `src/cga/pipelines/training_data/samplers.py`
- `src/cga/pipelines/training_data/pipeline.py`
- `configs/pipelines/training_data.yaml`
- `tests/unit/pipelines/training_data/test_samplers.py`
- `tests/unit/pipelines/training_data/test_pipeline.py`
- `dev_docs/sprints/sprint_4.md` (Phase 4.1 boxes ticked)

### Notes

- `ruff` is not installed in the venv, so lint was not run.
- The plan scoped changes to `samplers.py`/`pipeline.py`. The one-line YAML default was added so the sampler id
  is `supervised-negatives_per_positive5` even when `--set training_data.sampler.negatives_per_positive=5` is left
  out. The Phase 4.2 command passes it explicitly anyway, so the YAML line can be reverted if you prefer the
  pipeline config untouched.

### Commands

```
uv run pytest tests/unit/pipelines/training_data -q
# Phase 4.2 (next):
uv run cga run training_data --dataset ncit_doid_supervised --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5
```

## Phase 4.2 — Supervised training run

All runs use ingest `20261014T185110Z-c44babf2` and the copied stores (SapBERT text, DeepWalk graph). The `gcl_train`
settings are the Sprint 3 defaults: GCN, 128/128, `alpha=0.5`, triplet loss (margin 1.0), 100 epochs.

### Task 1 — Supervised `training_data`

```
uv run cga run training_data --dataset ncit_doid_supervised --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5
```
- Run `20261014T195623Z-d97cb67f` under `artifacts/training_data/ncit_doid_supervised/supervised-negatives_per_positive5/`
  (about 7 s).
- The manifest inputs are the `ncit_doid_supervised` ingest and the copied sapbert/deepwalk stores.

### Task 2 — Record stats

| Stat | Value |
|---|---|
| `n_universe_nodes` | 24,221 (24,227 minus the 6 isolated NCIT nodes) |
| `n_held_out_nodes` | 6,515 (= all gold test endpoints) |
| `n_gold_train_pairs` / `_used` / `_skipped` | 1,406 / 1,383 / 23 |
| `n_examples` | 6,915 (= 1,383 × 5), all `train` |

The leakage assertion in `validate_and_manifest` passed. The numbers match the Phase 4.1 in-memory smoke run.

### Task 3 — Supervised `gcl_train`

```
uv run cga run gcl_train --dataset ncit_doid_supervised --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.model_id=gcn-supervised --set gcl_train.train.epochs=100
```
- Run `20261014T195655Z-fcc54097` under `models/gcl/ncit_doid_supervised/gcn-supervised/` (about 12 s).
- Exported 24,221 × 128 node embeddings.

### Task 4 — Loss sanity check

| Model | `loss` epoch 1 → 100 | `loss_text` | `loss_graph` | NaN | epochs where loss rose |
|---|---|---|---|---|---|
| `gcn-supervised` | 0.6712 → 0.0037 | 0.7175 → 0.0038 | 0.6249 → 0.0035 | none | 0 (all three curves monotone) |
| `gcn-rank_topk-k5` (reference) | 0.6924 → 0.0653 | 0.7884 → 0.0724 | 0.5964 → 0.0581 | none | `loss`/`loss_text` 0; `loss_graph` 2 small upticks |

The supervised loss is near zero by about epoch 50 (0.013). The triplet margin is met for almost every example,
which is consistent with the bottom-5 negatives being easy ones (Phase 4.1 observation: only 520 distinct DOID
negatives). The training signal therefore fades early. Phase 4.4 will show whether that still moves held-out
retrieval.

### Task 5 — Unsupervised reference on the same split

```
uv run cga run training_data --dataset ncit_doid_supervised --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=rank_topk --set training_data.sampler.k=5
uv run cga run gcl_train --dataset ncit_doid_supervised --set gcl_train.inputs.training_data_sampler_id=rank_topk-k5 --set gcl_train.model.model_id=gcn-rank_topk-k5 --set gcl_train.train.epochs=100
```
- `training_data` run `20261014T195748Z-ff89cf3d` (`rank_topk-k5`): 88,530 examples, 6,515 held out. It has the same
  universe and held-out set as the supervised run, so the two models are compared on the same split.
- `gcl_train` run `20261014T195834Z-2a804d8e` under `models/gcl/ncit_doid_supervised/gcn-rank_topk-k5/` (about 20 s).
- These are the same settings as Sprint 3's `rank_topk-k5` point (SapBERT, k=5, GCN, `alpha=0.5`, 100 epochs).
  The difference is that the gold test endpoints are now held out, so this model never trained on them.

### State after Phase 4.2

| Lineage | `LATEST` |
|---|---|
| `artifacts/training_data/ncit_doid_supervised/supervised-negatives_per_positive5` | `20261014T195623Z-d97cb67f` |
| `artifacts/training_data/ncit_doid_supervised/rank_topk-k5` | `20261014T195748Z-ff89cf3d` |
| `models/gcl/ncit_doid_supervised/gcn-supervised` | `20261014T195655Z-fcc54097` |
| `models/gcl/ncit_doid_supervised/gcn-rank_topk-k5` | `20261014T195834Z-2a804d8e` |

`models/gcl/ncit_doid/` is untouched (for example, `gcn-rank_topk-k5/LATEST` is still `20261012T010104Z-c369f70c`).

No code or config changes in this phase; only artifacts. The sprint plan's Phase 4.2 boxes are ticked.
