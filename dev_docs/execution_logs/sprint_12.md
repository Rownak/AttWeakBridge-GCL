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
