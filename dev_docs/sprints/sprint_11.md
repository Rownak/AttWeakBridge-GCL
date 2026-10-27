# Sprint 11: Dual-encoder GCN on Enterprise ATT&CK × CWE (training, then gold evaluation)

## Objective

Run the Sprint 1 `training_data` (Phase 1.5) and `gcl_train` (Phase 1.7) pipelines as they are on `data/raw/enterprise_cwe/`, with the fewest possible variations: the `DualEncoder` with GCN towers, trained once with `triplet` and once with `infonce`, on top-k pseudo-pairs from SecRoBERTa text embeddings. Phases 11.1–11.3 produce two trained models and their node embeddings. Phase 11.4 scores them against the four ATT&CK→CWE gold sets with every Sprint 10 metric, next to the Sprint 10 reference rows.

This is the method of the paper (text embedding plus DeepWalk embedding, each passed through a GCN tower, concatenated), not the Sprint 9/10 `residual_adapter`. Nothing new is implemented: both pipelines, the `gcn` encoder, both losses and the Sprint 10 ingest/embeddings already exist.

## Fixed settings

| Setting | Value | Source |
|---|---|---|
| Dataset | `attack_cwe_ent` (ATT&CK Enterprise 20.0 × CWE v4.20) | `configs/datasets/attack_cwe_ent.yaml`, unchanged |
| Text embedding | `hf_mean_pool-pt_SecRoBERTa` (the only text encoder) | Sprint 10.5 store |
| Graph embedding | `deepwalk-deepwalk` (graph tower input, as in the paper) | Sprint 10.2 store |
| Pseudo-labels | `rank_topk`, `k = 5`, no other sampler | `unsup_u2` base |
| Training protocol | `protocol: unsupervised`, `held_out_policy: none`; every node is trained on; gold CSVs are not read | dataset YAML, `unsup_u2` |
| Model | `DualEncoder`, `encoder: gcn`, hidden 128, out 128, dropout 0 | `unsup_u2` |
| Loss | `triplet` (margin 1.0) or `infonce` (temperature 0.1); `alpha = 0.5` | `gcl_train.yaml` defaults |
| Optimiser | Adam, lr 1e-3, weight decay 0, 100 epochs | `unsup_u2` |
| Seed | 42 only | minimum variation |

Only two things differ between the two runs: `gcl_train.loss.name` and the model id. No tuning.

## Runs

| Run | Loss | Model id | Sampler id |
|---|---|---|---|
| `ent-dual-gcn-triplet-s42` | `triplet` | `ent-dual-gcn-triplet-s42` | `s11-rank_topk-k5` |
| `ent-dual-gcn-infonce-s42` | `infonce` | `ent-dual-gcn-infonce-s42` | `s11-rank_topk-k5` |

Both runs read the same `training_data` run, so the pseudo-pairs are identical and the loss is the only difference.

## Decision gate

| After | Go if | Else |
|---|---|---|
| 11.1 | `examples.parquet` is non-empty, every example id resolves in `universe.parquet`, the universe holds all nodes present in both stores (2,927 per Sprint 10.2), `held_out` is false for all of them | Fix before training |
| 11.2 | Both runs finish; `metrics.jsonl` has 100 rows each; loss at epoch 100 is below epoch 1; `node_embeddings` has one row per universe node | Fix before closing |
| 11.4 | The reference rows (frozen SecRoBERTa, GCN(SecRoBERTa), random) reproduce the Sprint 10.5 / 10.6 numbers exactly in every evaluate run; `n_queries` per source equals the 10.6 value (e.g. 284 for stix, 625 for union); a repeat run is identical | Fix before reading the table |

---

## Phase 11.1: Pseudo-pairs (`training_data`)

- [x] Confirm the Sprint 10 inputs are present and unchanged: ingest `LATEST` for `attack_cwe_ent`, SecRoBERTa store, DeepWalk store. No re-ingest, no re-embed. (The gold CSV choice in the dataset YAML does not affect training.)
- [x] Build the pseudo-pairs once:
  ```
  uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
    --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-pt_SecRoBERTa \
    --set training_data.sampler.sampler_id=s11-rank_topk-k5
  ```
  `unsup_u2` supplies `rank_topk`, `k = 5`, `held_out_policy: none`; the `--set` lines swap the text encoder and give the lineage its own sampler id.
- [x] Check the manifest: universe size, example count (cross-graph text positives and negatives plus within-graph structural ones), seed 42, text embedder id.
- [x] Unit tests unchanged and green: `uv run pytest tests/unit -q`.

## Phase 11.2: Train the dual-encoder GCN (`gcl_train`)

- [x] Pin the 11.1 run so both trainings read it even if `training_data` is rerun:
  ```
  RUN=$(ls -d artifacts/training_data/attack_cwe_ent/s11-rank_topk-k5/2* | tail -1 | xargs basename)
  ```
- [x] Triplet:
  ```
  uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
    --set gcl_train.model.model_id=ent-dual-gcn-triplet-s42 \
    --set gcl_train.inputs.training_data_sampler_id=s11-rank_topk-k5 \
    --set gcl_train.inputs.training_data_run=$RUN \
    --set gcl_train.loss.name=triplet
  ```
- [x] InfoNCE: the same command with `ent-dual-gcn-infonce-s42` and `--set gcl_train.loss.name=infonce`.
- [x] Check each `models/gcl/attack_cwe_ent/<model id>/`: manifest (encoder `gcn`, loss, seed, `training_data_run`, `universe_sha256`), `metrics.jsonl` (100 rows, loss decreasing), `node_embeddings/` (one row per universe node).
- [ ] Rerun one of the two with the same seed and confirm the node embeddings are identical (determinism).

## Phase 11.3: Execution log

- [x] `dev_docs/execution_logs/sprint_11.md`: what ran, run ids, example counts, loss at epochs 1 and 100 per loss, output paths, commands.
- [x] Mark the 11.1 and 11.2 checkboxes.

## Phase 11.4: Evaluate on the gold sets (`evaluate`)

Uses only existing evaluators, ingests, the 151-CWE pool file and random stores from Sprint 10.6. Gold is read here and nowhere else.

**Methods** (one config, `configs/experiments/attack_cwe_ent_dual_test.yaml`):

| method_id | Row | Source |
|---|---|---|
| `ent-d-trip` | Dual GCN, triplet | `gcl_run`, `ent-dual-gcn-triplet-s42` |
| `ent-d-nce` | Dual GCN, InfoNCE | `gcl_run`, `ent-dual-gcn-infonce-s42` |
| `ent-s-r1` | SecRoBERTa (frozen), reference | `embedding_store`, `hf_mean_pool-pt_SecRoBERTa` |
| `ent-s-r2-s42` | GCN(SecRoBERTa) residual adapter, reference | `gcl_run`, `ent-g-srb-s42` (Sprint 10.5) |
| `ent-rand-s42/43/44` | Random ranking, reference | `embedding_store`, `random-rand42/43/44` |

No fusion row and no new seeds: the sprint keeps the minimum variation, so the dual-encoder rows are single-seed and are read as a single draw, not a mean.

**Evaluations** (the Sprint 10.6 set, unchanged parameters):

| evaluation_id | kind | Metrics | Candidates |
|---|---|---|---|
| `gold_full` | `gold_alignment` | Hits@1, Hits@5, Hits@10, MRR (global mode, `sibling_filter_splits: [test]`) | all CWEs in the common set (944) |
| `prf_full` | `gold_alignment_prf` | precision, recall, F1, `n_predicted`, `n_correct`, `n_gold` (top-3, greedy one-to-one, threshold 0.0, `biogitom_filters: false`) | all CWEs |
| `gold_pool` | `gold_alignment` | same as `gold_full` | the 151-CWE shared pool |
| `prf_pool` | `gold_alignment_prf` | same as `prf_full` | the 151-CWE shared pool |

- [x] Write `configs/experiments/attack_cwe_ent_dual_test.yaml` with the methods and the four evaluations above (copy the evaluation block of `attack_cwe_ent_pool_test.yaml`; same pool path `data/processed/attack_cwe_ent/target_pool_union.json`).
- [x] Run one `evaluate` per gold source, one entire set at a time, using the ingest run of that source (stix `20261026T224906Z-a0725a37`, capec `20261026T224923Z-1b4b5b3d`, intersection `20261026T224935Z-dee50c06`, union `20261026T224948Z-0f7e98dc`):
  ```
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_dual_test \
    --set evaluate.inputs.ingest_run=<ingest run of that source>
  ```
- [x] Reproduction check: for the reference rows, compare against the Sprint 10.6 runs (stix `20261027T021941Z-01c5d7f6`, capec `…021953Z-fddc75ee`, intersection `…022008Z-ec24ba1c`, union `…022021Z-8c74b3b3`) on `(method_id, evaluation_id, view_id, metric_name, param_value)`; max absolute difference must be 0.0 and `n_queries` equal.
- [x] Repeat the union run and confirm all metric rows are identical.
- [x] Tables per source and view (full / pool), rows = the methods above, columns = Hits@1/5/10, MRR, P, R, F1, `n_correct`/`n_gold`. Add the lift over random (method metric ÷ mean of the three random rows) with `scripts/unsup_summary.py --lift`, adding a row set for this config if needed.
- [x] Report, without tuning anything: whether either dual-encoder row beats frozen SecRoBERTa on MRR and F1 in either view, and which loss is higher. Disclose the known limits from 10.6: 3 gold techniques fall outside the training universe for stix and union; F1 rests on few correct pairs; the pool is built from gold and is optimistic; single seed.
- [x] Unit tests still green: `uv run pytest tests/unit -q`.

## Phase 11.5: Evaluation log

- [x] Append the 11.4 results to `dev_docs/execution_logs/sprint_11.md`: evaluate run ids, tables, reproduction and repeat checks, commands, limits.
- [x] Mark the 11.4 checkboxes.

## Out of scope

- Plots, other metrics or evaluators, fusion rows, and anything that changes how a method is scored.
- Other text encoders, other `k`, other samplers, GAT, other seeds, hyperparameter changes.
- Code changes. If a pipeline fails on this dataset, stop and report it rather than patching it in this sprint.
