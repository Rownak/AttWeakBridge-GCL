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

## Phase 11.6: Fix the CWE description bug, re-ingest, re-embed

Bug (see `dev_docs/data_analysis/attack_capec_cwe_gold_alignment_analysis.md` §6): `cwe_parser._flatten_block(weakness, "Description")` searches `.//cwe:Description/xhtml:p`, so nested `Detection_Method` / `Mitigation` descriptions replace the weakness's own plain-text `<Description>`. 241 of 969 CWEs are affected, including 53 of the 151 gold CWEs (43–46% of gold pairs). Every Sprint 10/11 store and model on `attack_cwe_ent` is built on these texts.

- [x] **Fix** `src/cga/adapters/attack_cwe/cwe_parser.py`: read only the direct child (`weakness.find("cwe:Description", _NS)`, flattened with `itertext()` and whitespace-normalised). Do the same for `Extended_Description`. Leave the nested-field parsers (detection, mitigations, examples) unchanged.
- [x] **Unit tests** in `tests/unit/adapters/test_cwe_parser.py`: (a) a weakness whose `<Description>` is plain text and whose `Mitigation/Description` has `xhtml:p` children keeps its own description; (b) an `xhtml:p` description is still flattened; (c) a weakness with no `<Description>` gives `N/A`. `uv run pytest tests/unit -q` green.
- [x] **Audit** on `cwec_v4.20.xml`: 0 of 969 CWEs whose `description` differs from the direct-child `<Description>` (it was 241). Spot-check CWE-284, CWE-120, CWE-732 and CWE-200.
- [x] **Re-ingest** all four gold sources (same command as Sprint 10.1; `union` last so it is `LATEST`):
  ```
  for k in stix capec intersection union; do
    uv run cga run ingest --dataset attack_cwe_ent --set ingest.split.fractions=[1.0,0.0,0.0] \
      --set dataset.alignments.equivalence.source=$k
  done
  ```
  Check: node, edge and gold-pair counts are equal to the Sprint 10.1 ingests (287 / 566 / 221 / 628 pairs). Only `texts.parquet` should change, for the affected CWEs only.
- [x] **Re-embed** from the new union ingest:
  ```
  uv run cga run embed --dataset attack_cwe_ent --set embed.embedder.kind=hf_mean_pool \
    --set embed.embedder.model_ref=jackaduma/SecRoBERTa --set embed.embedder.label=pt_SecRoBERTa \
    --set embed.inputs.text_variant=description
  uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_deepwalk.yaml
  ```
  DeepWalk does not read text; rerun it only so every store points at the same ingest run. Check: 5,131 SecRoBERTa vectors, and the ATT&CK vectors are identical to the old store (their texts did not change).
- [x] **Random stores and pool file:** keep `random-rand42/43/44` and `target_pool_union.json` unless the node set or the union gold changed. Re-check the pool with `scripts/make_target_pool.py --check`.
- [x] **Seeding** (Sprint 11.2 open item): call `cga.core.seeds.seed_everything(seed)` at the start of `gcl_train`. Then a rerun with the same seed must give identical node embeddings.
- [x] **Retrain** on the new stores, as in 11.1 / 11.2: `training_data` with sampler id `s11b-rank_topk-k5`, then `gcl_train` with model ids `ent-dual-gcn-triplet-s42b` and `ent-dual-gcn-infonce-s42b`. Rerun one model to confirm the embeddings are identical.
- [x] **Hubness check** (frozen SecRoBERTa, union, full view): how many times CWE-120 and CWE-732 appear in the techniques' top-3 lists, before (77 and 34) and after the fix.

| Gate | Go if | Else |
|---|---|---|
| 11.6 | Audit is 0; ingest counts unchanged; tests green; retrained models are deterministic | Fix before 11.7 |

## Phase 11.7: Set-based gold evaluation (Hits@k, Recall@k, Precision@k)

The gold gives each technique an **unordered set** of correct CWEs (1–23, median 1–2). So the query unit is the **technique**, not the gold pair, and only order-free top-k metrics are used.

For technique `t` with gold set `R_t`, and `L_t@k` = the method's top-k CWEs (cosine, ties broken by node id):

| Metric | Per technique | Meaning |
|---|---|---|
| Hits@k | 1 if `L_t@k ∩ R_t ≠ ∅` | at least one correct CWE in the top k |
| Recall@k | `|L_t@k ∩ R_t| / |R_t|` | share of the technique's CWEs found |
| Precision@k | `|L_t@k ∩ R_t| / k` | share of the top k that is correct |
| P@k ceiling | `min(|R_t|, k) / k` | best possible Precision@k for that technique |

All four are macro-averaged over techniques. Use `k ∈ {1, 5, 10, 15, 20}`.

Precision@k is capped by `|R_t|`: a technique with 1 gold CWE can reach at most 0.067 at k = 15. That is why the ceiling is reported next to it, and P@k is read as a share of the ceiling. Recall@k carries the main signal; Hits@k is the lenient view.

- [x] **Evaluator** `gold_set_retrieval` (new file `src/cga/pipelines/evaluate/evaluators/gold_set_retrieval.py`; the existing `gold_alignment` and `gold_alignment_prf` stay unchanged so older runs remain comparable):
  - **Queries:** one per technique that has a test gold pair and appears in every method. Its relevant set is all of that technique's gold CWEs that are in the candidate pool.
  - **Candidates:** every shared CWE, or the `target_pool` CWEs. There is no sibling filter, because all gold CWEs are relevant together.
  - **Output:** rows `hits_at_k`, `recall_at_k`, `precision_at_k` and `precision_ceiling_at_k` (`param_name = k`). `n_queries` is the number of techniques.
  - **Provenance:** query-set sha256, `n_dropped_missing_in_a_method`, and the `|R_t|` distribution (mean, median, max).
  - Reuse `metrics.top_k_by_similarity`, `hit_at_k` and `recall_at_k`, and add `precision_at_k` to `metrics.py`.
- [x] **Unit tests:**
  - A hand-built 3-technique case covering `|R| = 1`, `|R| > k`, and a relevant CWE outside the pool.
  - Tie-break by id.
  - Ceiling values.
  - A random-vector sanity check: mean Recall@k ≈ k / |pool| within tolerance.
- [x] **Config** `configs/experiments/attack_cwe_ent_set_eval.yaml`:
  - **Methods:** frozen SecRoBERTa (new store), dual triplet / InfoNCE (`…-s42b`), and `random-rand42/43/44`.
  - **Evaluations:** `set_full` and `set_pool` (`gold_set_retrieval`, `ks: [1, 5, 10, 15, 20, 50]`).
  - Keep `gold_full` / `prf_full` from 11.4 as a bridge to earlier numbers.
- [x] **Run** one `evaluate` per gold source, using the 11.6 ingest runs. Repeat the union run and confirm the metric rows are identical.
- [x] **Old vs new texts** (same evaluator, frozen SecRoBERTa): score the old store `20261027T010723Z-8bc08adc` against the old ingests and the new store against the new ingests. This isolates the effect of the bug fix.
- [x] **Tables** per source and view: rows = methods; columns = Hits@k, Recall@k and Precision@k (with ceiling) for k = 5, 10, 15, 20; plus the lift of Recall@10 over random. Extend `scripts/unsup_summary.py` (new row set and column set).
- [x] **Report, without tuning:**
  - whether the fix changes frozen SecRoBERTa's Recall@k and the hubness;
  - whether either dual row beats frozen on Recall@10 / Hits@10 in either view;
  - which loss is higher.
  - Limits: single seed; the pool is built from gold; gold is transitive through CAPEC ("exploits", not "equivalent").

## Phase 11.8: Execution log

- [ ] Append 11.6–11.7 to `dev_docs/execution_logs/sprint_11.md`: the fix and its tests, audit numbers, new ingest / embed / training / evaluate run ids, determinism check, old vs new texts, set-based tables, commands, limits.
- [ ] Mark the 11.6–11.7 checkboxes.

## Out of scope

- Plots, fusion rows, and any change to the existing `gold_alignment` / `gold_alignment_prf` scoring.
- Other text encoders, other samplers, GAT, other seeds, hyperparameter changes.
- Phases 11.1–11.5: no code changes. Phase 11.6 changes only the CWE description parser and `gcl_train` seeding; Phase 11.7 adds only the new evaluator, its metric helper and the summary-script tables.
