# Sprint 12 — Execution log

## Phase 12.1 — Verify the reused artifacts

### What was done
Read-only verification of every pinned artifact in [sprint_12.md](../sprints/sprint_12.md). **All checks passed; nothing
was regenerated, trained or modified.** No repo files changed other than this log.

### How
One throwaway script (manifests, parquet indexes, `.npy` vectors, configs, `metrics.jsonl`), kept outside the repo in the
session scratchpad: `C:\Users\rowna\AppData\Local\Temp\claude\d--utep-post-doc-coding-AttWeakBridge-GCL\b6d3b225-d906-461e-b967-92290f6f6604\scratchpad\verify.py`.
Re-run from the repo root: `uv run python <that path>`.

### Results

**1. Run directories and manifests** — all 8 pinned runs exist; `manifest.json` has `status: success`, `seed: 42`, and
every file listed under `outputs` exists on disk.

| Artifact | Run id | Status |
|---|---|---|
| Ingest (11.10) | `20261027T194643Z-5d3f8baf` | success |
| mpnet-chunk store | `20261027T194928Z-8cf38e55` | success |
| DeepWalk store | `20261027T135210Z-3b18ba0c` | success |
| SecRoBERTa_desc store | `20261027T200841Z-09088f9f` | success |
| Pseudo-pairs | `20261027T205523Z-ac890ca3` | success |
| A0 | `20261027T205648Z-2aeb3259` | success |
| A1 | `20261027T232407Z-adc619c8` | success |
| A2 | `20261027T232541Z-d91ebd8e` | success |

**2. Stores** — universe = 2,927 nodes (1,983 `graph_1` ATT&CK + 944 `graph_2` CWE). Each store's index covers the full
universe; all vectors finite, no all-zero rows.

| Store | Vectors | Dim |
|---|---|---|
| mpnet-chunk | 5,131 | 768 |
| DeepWalk | 3,632 | 128 |
| SecRoBERTa_desc | 5,131 | 768 |

(mpnet / SecRoBERTa embed every text node; DeepWalk embeds graph nodes; the universe is a subset of each.)

**3. Pseudo-pairs** — manifest inputs: ingest `…194643Z-5d3f8baf`, text `…194928Z-8cf38e55`, graph
`…135210Z-3b18ba0c` (exactly the pinned runs). Summary: sampler `rank_topk`, 2,927 universe nodes, 14,635 examples
(= 2,927 × 5, all `train`), 0 held out (`held_out_policy: none`), seed 42, protocol `unsupervised`. The sampler `k = 5`
is read from the models' `config.resolved.yaml` (`training_data.sampler: {name: rank_topk, k: 5}`, no `negatives` key,
so the default `bottomk`); the pseudo-pair run itself stores no `config.resolved.yaml`, so k / negatives are inferred
from the model configs and the example count (5 per node).

**4. Models** — all `encoder: gcn`, `loss: infonce`, temperature 0.1, `alpha 0.5`, hidden / out 128, 100 epochs, seed 42,
`training_data_run: 20261027T205523Z-ac890ca3`; `node_embeddings` 2,927 × 128, finite, index ids equal the universe.

| Model | `branch` | Trainable params | Loss ep 1 → ep 100 | `metrics.jsonl` rows | NaN |
|---|---|---|---|---|---|
| A0 | none (= `both`) | 328,576 | 8.855 → 5.799 | 100 | no |
| A1 | `text` | 246,272 | 8.849 → 6.108 | 100 | no |
| A2 | `graph` | 82,432 | 8.773 → 6.595 | 100 | no |

These also cover the 12.2 loss / parameter checks; 12.2 only needs to record them in the run table.

**5. DeepWalk vs 11.10 graph** — the DeepWalk store was built from ingest `20261027T134320Z-32a16662` (11.6). Compared to
the 11.10 ingest `…194643Z-5d3f8baf`: node ids identical (5,840), edge ids identical (22,979), and the
`(src, dst, relation, graph_role)` tuples identical. All DeepWalk index ids are present in the 11.10 nodes, so
`hop_radius` reading the 11.10 graph is consistent with the DeepWalk store.

### Notes / limitations
- A0's `config.resolved.yaml` still carries the default `training_data.inputs` block (`hf_mean_pool-bert_base`,
  `run: latest`). It is stale and unused: the run consumed the pinned `training_data_run`, and the manifest inputs show the
  correct mpnet-chunk and DeepWalk runs.
- Manifest `git_dirty: true` for all Sprint 11 runs (as recorded in earlier logs).
- A1 / A2 `branch` lives at `gcl_train.model.branch`; A0 has no such key (default `both`).

### Commands
```
cd D:\utep\post_doc\coding\AttWeakBridge-GCL
uv run python C:\Users\rowna\AppData\Local\Temp\claude\d--utep-post-doc-coding-AttWeakBridge-GCL\b6d3b225-d906-461e-b967-92290f6f6604\scratchpad\verify.py
```

### Status
Phase 12.1 acceptance met: every check passes with no regeneration. Ready for Phase 12.2. The checkboxes in
`sprint_12.md` are now ticked (done at the start of the 12.2 pass).

---

## Phase 12.2 — A0 / A1 / A2 runs (reused, no training)

### What was done
Recorded the sanity checks and the exact configuration of the three reused models and the reproducibility evidence.
**Nothing was trained.** Read-only; the only file changes are the checkboxes in `sprint_12.md` (Phases 12.1 and 12.2
ticked) and this log. Source data: each run's `manifest.json`, `config.resolved.yaml` and `metrics.jsonl`.

### Training sanity (from `metrics.jsonl`)
Each file has 100 rows, epochs 1..100 consecutive, fields `epoch, loss, loss_graph, loss_text, lr, elapsed_s`, no NaN.
`loss = alpha * loss_graph + (1 - alpha) * loss_text` with `alpha 0.5` (checked on A0 epoch 1: 0.5 × 7.025 + 0.5 × 10.685 = 8.855).

| Model | Rows | Loss ep 1 → ep 100 | `loss_graph` ep 1 → 100 | `loss_text` ep 1 → 100 | Lower at 100 | Trainable params (manifest) | Duration |
|---|---|---|---|---|---|---|---|
| A0 | 100 | 8.855 → 5.799 | 7.025 → 6.408 | 10.685 → 5.191 | yes | 328,576 | 87.0 s |
| A1 | 100 | 8.849 → 6.108 | 8.766 → 7.165 | 8.932 → 5.051 | yes | 246,272 | 86.1 s |
| A2 | 100 | 8.773 → 6.595 | 6.457 → 5.817 | 11.090 → 7.373 | yes | 82,432 | 87.4 s |

Parameter counts equal the expected 328,576 / 246,272 / 82,432. For A1 and A2 the unused branch's loss term is still
logged (it is computed on the output embeddings of the same encoder), so the totals are not comparable across rows as
quality measures; they are only checked for sanity here.

### Run table (the paper's run specification)

| Field | A0 (full dual GCN) | A1 (text-only GCN) | A2 (graph-only GCN) |
|---|---|---|---|
| Model id | `ent-dual-gcn-infonce-mpnetc-s42` | `ent-abl-text-gcn-nce-mpnetc-s42` | `ent-abl-graph-gcn-nce-mpnetc-s42` |
| Run id | `20261027T205648Z-2aeb3259` | `20261027T232407Z-adc619c8` | `20261027T232541Z-d91ebd8e` |
| `branch` | none (= `both`) | `text` | `graph` |
| Encoder | `gcn` (message passing on) | `gcn` (`message_passing: true`) | `gcn` (`message_passing: true`) |
| Hidden / out dim, dropout | 128 / 128, 0.0 | 128 / 128, 0.0 | 128 / 128, 0.0 |
| Text input | mpnet-chunk `20261027T194928Z-8cf38e55` | same | same (not used by the graph branch) |
| Graph input | DeepWalk `20261027T135210Z-3b18ba0c` | same (not used by the text branch) | same |
| Ingest | `20261027T194643Z-5d3f8baf` | same | same |
| Sampler id / training data run | `s11c-rank_topk-k5` / `20261027T205523Z-ac890ca3` | same | same |
| Sampler | `rank_topk`, `k = 5`, `negatives` default (`bottomk`), 14,635 examples, 0 held out | same | same |
| Loss | `infonce`, temperature 0.1, `alpha 0.5` | same | same |
| Optimiser | Adam, lr 0.001, weight decay 0 | same | same |
| Epochs, seed, device | 100, 42, cpu | same | same |
| Checkpoints | every 50 epochs | same | same |
| Output | `node_embeddings` 2,927 × 128 | same | same |

All values come from the manifests and `config.resolved.yaml` files. The only config differences between the rows are
`model_id` and `branch` (plus the `message_passing: true` key, which is the default and is simply written out in the
Sprint 11.12 configs). `config_hash` values: `2aeb3259…`, `adc619c8…`, `d91ebd8e…` (the hash suffix of each run id).
The `training_data.inputs` block (`bert_base`, `latest`) in the configs is unused (see Phase 12.1 notes).

### Reproducibility statement
Sprint 11.12 retrained the pinned configurations and found bit-identical embeddings; this sprint cites that evidence
instead of retraining:

- **A0:** the 11.12 gate retrain `ent-abl-full-gcn-nce-mpnetc-s42` / `20261027T232219Z-0bded1cb` reproduces the 11.11 A0 model.
- **A1:** the rerun `20261027T233417Z-adc619c8` reproduces `20261027T232407Z-adc619c8`.
- **Independent check made now** (SHA-256 of `node_embeddings/vectors.npy`, first 16 hex chars):

| Run | SHA-256 prefix |
|---|---|
| A0 `…205648Z-2aeb3259` | `6cdc5558f803b5cb` |
| A0 gate retrain `…232219Z-0bded1cb` | `6cdc5558f803b5cb` |
| A1 `…232407Z-adc619c8` | `21a9a168f4caaea8` |
| A1 rerun `…233417Z-adc619c8` | `21a9a168f4caaea8` |

The A0 hash also matches the `node_embeddings_vectors` sha256 in the A0 manifest (`6cdc5558…`). A2 was trained once
(no repeat run), so no bit-identity claim is made for it; CPU, seed 42, same code path as A1.

### Notes / limitations
- Single seed (42) and a single pseudo-pair draw, as decided for the sprint.
- Training losses are not comparable across rows as a quality measure (different branches); quality is judged in Phase 12.3.

### Commands
```
# Re-derive the tables above (read-only), from the repo root
uv run python C:\Users\rowna\AppData\Local\Temp\claude\d--utep-post-doc-coding-AttWeakBridge-GCL\b6d3b225-d906-461e-b967-92290f6f6604\scratchpad\runs.py
sha256sum models/gcl/attack_cwe_ent/ent-abl-text-gcn-nce-mpnetc-s42/*/node_embeddings/vectors.npy
```

### Status
Phase 12.2 acceptance met: all three runs are sane and fully specified; nothing was trained. Next: Phase 12.3
(experiment and plot configs, evaluate, plot).

---

## Phase 12.3 — Evaluate and plot (Sprint 3 protocol)

### What was done
Created the experiment and plot configs, ran evaluate twice (identical), and plotted. No training, no evaluator change.

**Files created**
- `configs/experiments/attack_cwe_ent_paper_compare.yaml` — 5 methods pinned by run id (`mpnet-chunk`, `deepwalk` as `embedding_store`; `a0`, `a1`, `a2` as `gcl_run`); evaluation `neigh` (`neighborhood`, ks 1/5/10/20, roles `graph_1`/`graph_2`, views `structural_r1`, `structural_r2` via `hop_radius`, `textual` via `cosine_top_k` top_k 10 with the pinned SecRoBERTa_desc store `20261027T200841Z-09088f9f`); evaluation `clusters` (`cluster_quality`, n_clusters 2/4/8/16/32); `visualize` for a0/a1/a2.
- `configs/plots/attack_cwe_ent_paper_compare.yaml` — `precision_at_k` (neigh, `normalized_precision_at_k`, x `param_value`, hue `method_label`, facet `[view_id, graph_role]`), `silhouette` (clusters, facet `[graph_role]`), `loss_curves` enabled, `plot.inputs.evaluate_run: 20261028T044615Z-b20dd0c5`.
- Modified: `dev_docs/sprints/sprint_12.md` (Phase 12.3 boxes ticked).

### Runs

| Step | Run id | Output |
|---|---|---|
| Evaluate (run 1, pinned for the plot) | `20261028T044615Z-b20dd0c5` | `artifacts/evaluations/attack_cwe_ent/20261028T044615Z-b20dd0c5/` (`metrics.parquet`, `metrics.json`, `plots/tsne_a0.png`, `tsne_a1.png`, `tsne_a2.png`) |
| Evaluate (repeat) | `20261028T044920Z-b20dd0c5` | same layout |
| Plot | `20261028T045216Z-320e3234` | `artifacts/plots/attack_cwe_ent/20261028T045216Z-320e3234/figures/` (`precision_at_k.png`, `silhouette.png`, `loss_curve_<model_id>.png`) |

The suffix `b20dd0c5` is the config hash, identical in both evaluate runs. Evaluate produced 480 `neigh` rows and 50 `clusters` rows
(530 total; 106 per method); ingest input `20261027T194643Z-5d3f8baf`.

### Checks
- **Repeat identical:** `metrics.parquet` of the two runs (530 × 15) differs only in the `run_id` column; max |Δ metric_value| = 0.0.
- **Query sets (`neighborhood`):** `n_queries` is the same for all 5 methods in every (view, role):

| View | `graph_1` (ATT&CK) | `graph_2` (CWE) |
|---|---|---|
| `structural_r1` | 1,981 | 944 |
| `structural_r2` | 1,981 | 944 |
| `textual` | 1,983 | 944 |

  (`structural_*` drop 2 ATT&CK nodes with no neighbours within the radius.)
- **Query sets (`cluster_quality`) — NOT equal across methods.** The cluster evaluator scores every node a store contains and does not intersect:

| Method | `graph_1` nodes | `graph_2` nodes |
|---|---|---|
| A0 / A1 / A2 | 1,983 | 944 |
| DeepWalk (frozen) | 2,688 | 944 |
| mpnet-chunk (frozen) | 4,162 | 969 |

  The silhouette of the frozen rows is therefore computed on a larger node set (all ATT&CK techniques / sub-techniques or
  CWE entries that have vectors) than the trained rows (the 2,927-node pseudo-pair universe). Evaluator changes are out of
  scope, so this is **not fixed**; it is a caveat for Phase 12.4 (silhouette column compares different node sets).
- **Figures / projections exist:** `precision_at_k.png` (6 panels: 3 views × 2 graphs), `silhouette.png` (2 panels), and t-SNE projections for A0, A1, A2. Both figures were opened and render with 5 labelled lines per panel.
- `loss_curves` wrote one PNG per `model_id` under `attack_cwe_ent` (all Sprint 11 models, as the spec predicted). The three for the paper package:
  `loss_curve_ent-dual-gcn-infonce-mpnetc-s42.png` (A0), `loss_curve_ent-abl-text-gcn-nce-mpnetc-s42.png` (A1), `loss_curve_ent-abl-graph-gcn-nce-mpnetc-s42.png` (A2).

### Early observations (for the 12.4 reading; no tuning, nothing concluded yet)
- Mean silhouette over n_clusters 2–32 (only partly comparable, see above) — `graph_1`: A2 0.068, A1 0.064, mpnet-chunk 0.054, A0 0.043, DeepWalk 0.042; `graph_2`: DeepWalk 0.147, A2 0.114, A1 0.099, A0 0.095, mpnet-chunk 0.058.
- From `precision_at_k.png`: structural views are led by DeepWalk, then A2 / A0 (`graph_1`); on the `textual` view mpnet-chunk leads, then A1 and A0, with DeepWalk and A2 lowest. The full table and reading come in Phase 12.4.

### Commands
```
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_paper_compare --set evaluate.inputs.ingest_run=20261027T194643Z-5d3f8baf   # run twice
uv run cga run plot --dataset attack_cwe_ent --config configs/plots/attack_cwe_ent_paper_compare.yaml
```

### Status
Phase 12.3 acceptance met: repeat evaluate identical, neighborhood query sets equal across methods (counts recorded), both
figures and the A0 / A1 / A2 projections exist. One caveat found: cluster_quality node sets differ for the frozen rows.
Next: Phase 12.4 (table script, package, reading, caveats).

---

## Phase 12.4 — Paper table and package

### What was done
Added the table script and its unit test, generated the table from the pinned evaluate run, assembled the paper package, and wrote the reading and caveats. No evaluator change, nothing retrained.

**Files created / modified**
- `scripts/paper_proxy_table.py` — reads `metrics.parquet` of an evaluate run, writes `paper_table.md` and `paper_table.csv` next to it. Rows = methods in config order; per view (`structural_r1`, `structural_r2`, `textual`) and graph (ATT&CK = `graph_1`, CWE = `graph_2`): MRR and normalized precision@10; plus the mean silhouette over `n_clusters` 2–32 and both graphs (10 values per method). Best per column in bold (Markdown only; ties all bold; values compared after rounding to 4 digits). MRR does not depend on k, so the per-k copies are averaged into one value.
- `tests/unit/test_paper_proxy_table.py` — 4 tests on a hand-built `metrics.parquet`-shaped fixture: values (including that P@10 uses k = 10, not k = 1), bolding, error on a missing cell, byte-identical rewrite. `uv run pytest tests/unit -q`: 618 passed.
- `artifacts/paper/attack_cwe_ent/sprint_12/` (package, see below).
- `dev_docs/sprints/sprint_12.md`: Phase 12.4 boxes ticked.

### Table (evaluate run `20261028T044615Z-b20dd0c5`; best per column in bold)

| Method | r1 ATT&CK MRR | r1 ATT&CK P@10 | r1 CWE MRR | r1 CWE P@10 | r2 ATT&CK MRR | r2 ATT&CK P@10 | r2 CWE MRR | r2 CWE P@10 | text ATT&CK MRR | text ATT&CK P@10 | text CWE MRR | text CWE P@10 | Silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 full dual GCN | 0.6758 | 0.4255 | 0.5868 | 0.8075 | 0.9977 | 0.9020 | 0.9760 | 0.7974 | 0.4100 | 0.1205 | 0.4531 | 0.1724 | 0.0691 |
| A1 text-only GCN | 0.4597 | 0.2587 | **0.6397** | 0.7841 | 0.9531 | 0.7747 | 0.9575 | 0.7194 | 0.4789 | 0.1484 | 0.4840 | 0.1875 | 0.0814 |
| A2 graph-only GCN | 0.7798 | 0.5485 | 0.5507 | 0.8250 | **1.0000** | 0.9410 | 0.9966 | 0.8848 | 0.3523 | 0.0930 | 0.3816 | 0.1500 | 0.0911 |
| DeepWalk (frozen) | **0.8655** | **0.6887** | 0.5605 | **0.8443** | **1.0000** | **0.9801** | **1.0000** | **0.9466** | 0.3460 | 0.0855 | 0.3818 | 0.1500 | **0.0945** |
| mpnet-chunk (frozen) | 0.3439 | 0.1589 | 0.2760 | 0.3398 | 0.9250 | 0.7273 | 0.6108 | 0.3244 | **0.5649** | **0.1908** | **0.6103** | **0.2422** | 0.0558 |

`r1` / `r2` = `structural_r1` / `structural_r2`; `text` = `textual`. Query counts per cell are in Phase 12.3.

### Package: `artifacts/paper/attack_cwe_ent/sprint_12/`
- `paper_table.md`, `paper_table.csv` (copies of the files in the evaluate run directory)
- `figures/precision_at_k.png`, `figures/silhouette.png`
- `loss_curves/loss_curve_a0.png`, `loss_curve_a1.png`, `loss_curve_a2.png` (renamed from `loss_curve_<model_id>.png`)
- `projections/tsne_a0.png`, `tsne_a1.png`, `tsne_a2.png`
- `runs.yaml` — every pinned run id (ingest, three stores, pseudo-pairs, A0 / A1 / A2, the 11.12 reproducibility runs), the evaluate run, its repeat and the plot run, and the three commands.

### Reading (no tuning; single seed)
- **Structural views.** Frozen DeepWalk leads almost every column on `structural_r1` and `structural_r2` (ATT&CK `structural_r1` MRR 0.866, P@10 0.689; `structural_r2` MRR 1.000, P@10 0.980 / 0.947). Exceptions: on CWE `structural_r1` the MRR is highest for A1 (0.640; A0 0.587, DeepWalk 0.561). A2 is the closest trained row to DeepWalk (`structural_r2` P@10 0.941 / 0.885), then A0, then A1. Frozen mpnet-chunk is lowest in every structural column.
- **Textual view.** Frozen mpnet-chunk leads all four columns (MRR 0.565 / 0.610, P@10 0.191 / 0.242). A1 is second, then A0; A2 and frozen DeepWalk are lowest and nearly tied (CWE P@10 0.150 for both, MRR 0.382 for both).
- **Silhouette.** DeepWalk 0.0945 > A2 0.0911 > A1 0.0814 > A0 0.0691 > mpnet-chunk 0.0558. The differences are small and the node sets differ (caveat 4).
- **Where A0 / A1 / A2 sit.** Every trained row lies between the two frozen baselines in most columns: the structural-leaning rows (A2, then A0) sit near DeepWalk on structural views and near it (low) on the textual view; A1 sits toward mpnet-chunk (second on the textual view, weaker on structural). A0, with both branches, is intermediate on both kinds of view and leads neither. On CWE `structural_r1` MRR the trained rows A0 and A1 exceed both frozen baselines.

### Caveats (carry into the paper text)
1. **Proxy relevance sets, not gold labels.** Relevant items are graph neighbours (radius 1 / 2) or the top 10 by cosine in a reference text space.
2. **Not an alignment measure.** Neighborhoods are scored within each graph, so these metrics do not measure ATT&CK → CWE alignment (see the gold evaluations of Sprint 11).
3. **Textual-view bias.** The reference store (SecRoBERTa, `description_only`) is itself a text encoder, so text-based rows (mpnet-chunk, A0, A1) have an advantage on `textual`; likewise DeepWalk and A2 are close to the structural relevance sets by construction.
4. **Silhouette node sets differ.** `cluster_quality` scores all nodes in each store: 1,983 / 944 (A0, A1, A2), 2,688 / 944 (DeepWalk), 4,162 / 969 (mpnet-chunk) for ATT&CK / CWE. The silhouette column is therefore not on one common node set; treat it as indicative only (the neighborhood columns are on identical query sets).
5. **Single seed (42)** and one pseudo-pair draw; differences of a few hundredths are not tested against seed variation.
6. Trained rows were optimised for cross-graph pseudo-pairs, not for these within-graph proxies.

### Commands
```
uv run python scripts/paper_proxy_table.py --dataset attack_cwe_ent --run 20261028T044615Z-b20dd0c5   # regenerates paper_table.md / .csv
uv run pytest tests/unit/test_paper_proxy_table.py -q
uv run pytest tests/unit -q
```
The table was generated twice; SHA-256 of both files is identical between the two runs
(`paper_table.md` `b2b9e7ec…`, `paper_table.csv` `9570abba…`).

### Status
Phase 12.4 acceptance met: the table regenerates byte-identically with one command, and the package folder holds every
figure and run id referenced above. Sprint 12 phases 12.1–12.4 are complete.
