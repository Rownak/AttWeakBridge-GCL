# Sprint 11: Dual-encoder GCN on Enterprise ATT&CK × CWE (training only)

## Objective

Run the Sprint 1 `training_data` (Phase 1.5) and `gcl_train` (Phase 1.7) pipelines as they are on `data/raw/enterprise_cwe/`, with the fewest possible variations: the `DualEncoder` with GCN towers, trained once with `triplet` and once with `infonce`, on top-k pseudo-pairs from SecRoBERTa text embeddings. Evaluation is out of scope; the outputs are two trained models and their node embeddings.

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

---

## Phase 11.1: Pseudo-pairs (`training_data`)

- [ ] Confirm the Sprint 10 inputs are present and unchanged: ingest `LATEST` for `attack_cwe_ent`, SecRoBERTa store, DeepWalk store. No re-ingest, no re-embed. (The gold CSV choice in the dataset YAML does not affect training.)
- [ ] Build the pseudo-pairs once:
  ```
  uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
    --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-pt_SecRoBERTa \
    --set training_data.sampler.sampler_id=s11-rank_topk-k5
  ```
  `unsup_u2` supplies `rank_topk`, `k = 5`, `held_out_policy: none`; the `--set` lines swap the text encoder and give the lineage its own sampler id.
- [ ] Check the manifest: universe size, example count (cross-graph text positives and negatives plus within-graph structural ones), seed 42, text embedder id.
- [ ] Unit tests unchanged and green: `uv run pytest tests/unit -q`.

## Phase 11.2: Train the dual-encoder GCN (`gcl_train`)

- [ ] Pin the 11.1 run so both trainings read it even if `training_data` is rerun:
  ```
  RUN=$(ls -d artifacts/training_data/attack_cwe_ent/s11-rank_topk-k5/2* | tail -1 | xargs basename)
  ```
- [ ] Triplet:
  ```
  uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
    --set gcl_train.model.model_id=ent-dual-gcn-triplet-s42 \
    --set gcl_train.inputs.training_data_sampler_id=s11-rank_topk-k5 \
    --set gcl_train.inputs.training_data_run=$RUN \
    --set gcl_train.loss.name=triplet
  ```
- [ ] InfoNCE: the same command with `ent-dual-gcn-infonce-s42` and `--set gcl_train.loss.name=infonce`.
- [ ] Check each `models/gcl/attack_cwe_ent/<model id>/`: manifest (encoder `gcn`, loss, seed, `training_data_run`, `universe_sha256`), `metrics.jsonl` (100 rows, loss decreasing), `node_embeddings/` (one row per universe node).
- [ ] Rerun one of the two with the same seed and confirm the node embeddings are identical (determinism).

## Phase 11.3: Execution log

- [ ] `dev_docs/execution_logs/sprint_11.md`: what ran, run ids, example counts, loss at epochs 1 and 100 per loss, output paths, commands.
- [ ] Mark the 11.1 and 11.2 checkboxes.

## Out of scope

- Evaluation of any kind (gold alignment, ranking, P/R/F1, plots). `evaluate` is not run in this sprint.
- Other text encoders, other `k`, other samplers, GAT, other seeds, hyperparameter changes.
- Code changes. If a pipeline fails on this dataset, stop and report it rather than patching it in this sprint.
