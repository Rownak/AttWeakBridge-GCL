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

## Phase 11.9: Controlled text-embedding baseline (`sentence-transformers/multi-qa-mpnet-base-dot-v1`)

Question: does a sentence-level encoder retrieve gold CWEs better than frozen, token-mean-pooled SecRoBERTa? Only the text encoder changes. The ingest, the texts, the candidate sets, the evaluators and the baselines stay the same, so any difference comes from the encoder. Nothing is trained and nothing is tuned.

**Fixed for the comparison**

| Setting | Value |
|---|---|
| Ingest | the 11.6 runs (union `20261027T134320Z-32a16662`, stix `…134239Z-8101c66c`, capec `…134252Z-cbfb9a99`, intersection `…134304Z-d983eac7`) |
| Text | `text_variant: description`, same text per node as the SecRoBERTa store (same `texts.parquet`) |
| Nodes | every node of the ingest, as in `hf_mean_pool-pt_SecRoBERTa/20261027T134403Z-8bc08adc` |
| Encoder | `sentence-transformers/multi-qa-mpnet-base-dot-v1`, one `encode` call per text, no token-level mean pooling and no chunking |
| Reference rows | frozen SecRoBERTa (`20261027T134403Z-8bc08adc`) and `random-rand42/43/44`, both existing stores |
| Scoring | cosine, as in every earlier evaluation. The evaluators L2-normalise the vectors; the stored vectors are left as the model outputs them |

**Known differences, to be reported (not removed)**

- The model is trained for dot-product scoring, so cosine is not its native metric. Cosine is kept so the protocol matches the other rows.
- It was trained for question → passage retrieval. Both sides are encoded the same way, with no prefix (the model defines none).
- Texts longer than the model's `max_seq_length` are truncated. SecRoBERTa instead chunks and averages. Record `max_seq_length` and the number of truncated nodes, split by graph, in the log.

- [x] **Dependency:** add `sentence-transformers` to `pyproject.toml` and `uv.lock` (`uv add sentence-transformers`). Check that the existing `torch` / `transformers` pins still resolve and that `uv run pytest tests/unit -q` stays green.
- [x] **Embedder** (smallest change): add `SentenceTransformerEmbedder` to `src/cga/pipelines/embed/text_embedders.py` with the same `embed(text) -> (vector, n_chunks)` interface as `HfMeanPoolEmbedder`, so the store writer stays unchanged:
  - Load `SentenceTransformer(model_ref, device=...)` once; empty text gives a zero vector, as in the existing embedder.
  - Return `(vector, 1)`; do not normalise.
  - Register the new kind `sentence_transformer` in `_TEXT_KINDS`, in the `embed` dispatch and in the `Unknown embedder.kind` message of `pipeline.py`.
  - Google-style docstrings; note why truncation replaces chunking.
- [x] **Unit test** `tests/unit/pipelines/embed/`: a stub `SentenceTransformer` (no download) checks the output length, the zero vector for empty text, that the vector is not normalised, and that the new kind is accepted. `uv run pytest tests/unit -q` green.
- [x] **Config** `configs/pipelines/embed_mpnet.yaml` (copy of `embed.yaml` with `kind: sentence_transformer`, `model_ref: sentence-transformers/multi-qa-mpnet-base-dot-v1`, `label: multi_qa_mpnet_dot`, `text_variant: description`, `batch_size: 32`; drop the HF-only keys). Embedder id: `sentence_transformer-multi_qa_mpnet_dot`.
- [x] **GPU check first:** the machine has an NVIDIA GPU with 16 GB and CUDA. Confirm PyTorch sees it before embedding:
  ```
  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  nvidia-smi
  ```
  `torch.cuda.is_available()` must print `True`. If it is `False`, stop and fix the CUDA build of `torch` before continuing; do not fall back to CPU silently.
- [x] **Embed** from the union ingest on the GPU. The `embed` pipeline passes the run's `device` to the embedder, so the override is enough:
  ```
  uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_mpnet.yaml --set device=cuda
  ```
  - **Embedder:** load the model on `device`; encode in batches (`embed_mpnet.yaml` keeps `batch_size: 32`). The 768-dimensional base model needs well under 16 GB at that batch size. If it runs out of memory, halve the batch size and note it in the log. Batch size changes the vectors only at the floating-point level, so it is fixed in the config and logged.
  - **Check:** `index.parquet` has the same node ids and order as the SecRoBERTa store; dimension 768; no failed rows in `status.parquet`; no all-zero vector other than for nodes without text. Record the device, `torch` / `sentence-transformers` versions and the wall time.
  - **Repeat:** run the embed a second time on the GPU. GPU kernels can reorder floating-point sums, so do not require bit-identical vectors: require max absolute difference below 1e-4 and, for each evaluation, identical metric rows (or differences below 1e-9) when both stores are scored. If a metric row differs, report it with the store ids rather than choosing one run.
  - **Pin:** all later configs reference the first run id (`ref: <run id>`), not `latest`, so a later embed run cannot change the scored store.
- [x] **Config** `configs/experiments/attack_cwe_ent_mpnet_eval.yaml`:
  - **Methods:** `ent-mpnet` (new store, pinned to the embed run id), `ent-s-r1` (frozen SecRoBERTa, pinned to `20261027T134403Z-8bc08adc`), `ent-rand-s42/43/44`.
  - **Evaluations**, all existing kinds with the parameters used in 11.4 and 11.7: `gold_full`, `prf_full`, `gold_pool`, `prf_pool` (`attack_cwe_ent_dual_test.yaml`) and `set_full`, `set_pool` (`attack_cwe_ent_set_eval.yaml`).
- [x] **Run** one `evaluate` per gold source with its 11.6 ingest run, one entire set at a time (see 11.4 for the command form).
- [x] **Checks:**
  - **Reproduction:** for `ent-s-r1` and the random rows, compare with the 11.4 / 11.7 runs on `(method_id, evaluation_id, view_id, metric_name, param_value)`; max absolute difference 0.0 and equal `n_queries`.
  - **Repeat:** rerun the union evaluation on the same pinned store; all rows identical (the evaluation reads stored vectors, so it does not depend on the GPU).
  - **Query set:** `n_queries` is the same for `ent-mpnet` and `ent-s-r1` in every evaluation (same techniques, same candidates).
- [x] **Metrics reported per source and view (full / pool)**, with the existing names:
  - **Ranking** (`gold_alignment`): Hits@1, Hits@10 (Hits@5 too, as in 11.4), MRR.
  - **Assignment** (`gold_alignment_prf`, top-3, greedy one-to-one): Precision, Recall, F1, and the counts `n_predicted`, `n_correct`, `n_gold`.
  - **Set-based** (`gold_set_retrieval`): Hits@k, Recall@k, Precision@k with the ceiling, k = 5, 10, 15, 20, plus the lift of Recall@10 over random.
  - **Cosine and CSLS:** the evaluators do cosine only; `grep -ri csls src scripts` finds nothing. CSLS is therefore not applicable here and is not added in this phase (it would need a new scoring option). The log states this. The CSLS and mean-centring probe from `dev_docs/data_analysis/attack_cwe_low_set_scores_case_study.md` stays a separate, optional follow-up.
- [x] **Tables:** `scripts/unsup_summary.py` with a new row set `attack_ent_mpnet` (mpnet, frozen SecRoBERTa, random) for the 11.4 and 11.7 table formats, including the lift over random.
- [x] **Report, without tuning:**
  - whether mpnet beats frozen SecRoBERTa on MRR and F1 (ranking, assignment) and on Recall@10 / Hits@10 (set-based), in either view, in each source;
  - whether its top-10 lists show the same hardware / SoC hubs (count of the five most frequent CWEs, as in the case-study report);
  - the truncation counts and what they could mean for the ATT&CK texts;
  - limits: one encoder run; cosine instead of the model's native dot product; gold is transitive through CAPEC; the pool is built from gold and is optimistic; mpnet and SecRoBERTa differ in pretraining, pooling and truncation, so the comparison measures "encoder as a whole", not one factor.

- [x] **Execution log** (last task of this phase): write `dev_docs/execution_logs/sprint_11_9.md` with the dependency change, embedder and test, embed run id and checks (device, versions, wall time, truncation counts), evaluate run ids per source, reproduction and repeat checks, tables, report, commands and limits. Then mark the 11.9 checkboxes.

## Phase 11.10: Description-only texts and chunked sentence embeddings

Question: how much of the 11.9 result depends on (a) truncation at 512 tokens and (b) the extra text in the `description` variant (CWE name and extended description)? Two changes, tested separately: the text becomes **only the standard `description` field** of each ATT&CK and CWE node (no CWE name, no extended description, nothing else), and long texts are **chunked and mean-pooled** instead of truncated, so the whole description is embedded. It repeats the 11.9 tasks on the new texts.

**Decisions to confirm** (taken here, stated in the log): (1) the CWE `name` is excluded as well, since the request says "only the standard description field"; (2) the chunk vectors are averaged **weighted by token count**, so a text that fits one chunk gives exactly the 11.9 vector and a long text gets the mean over all its tokens (SecRoBERTa's `hf_mean_pool` averages chunks without weights); (3) the ATT&CK text is not cleaned (`(Citation: …)` markers stay).

- [x] **Text variant:** add `description_only` to `configs/datasets/attack_cwe_ent.yaml` (`attack: [description]`, `cwe: [description]`); `description` and `full_metadata` stay unchanged so older runs reproduce. Re-ingest the four gold sources (same command as 11.6, `union` last). Check against the 11.6 ingests: identical node ids, edges and gold pairs; the `description` texts are unchanged; for `description_only` report the text-length change per graph and the nodes whose description is empty or `N/A`.
- [x] **Chunked embedder:** add `chunking: true|false` to `SentenceTransformerEmbedder` (default `false` = 11.9 behaviour). With `true`: tokenize with the model's tokenizer, split into chunks of at most `max_seq_length − 2` tokens, add the special tokens, encode each chunk, return the token-weighted mean and the number of chunks as `n_chunks`. Unit tests (stub model): a one-chunk text equals the unchunked vector; a long text gives `n_chunks > 1` and the weighted mean of its chunk vectors; empty text stays a zero vector.
- [x] **Configs:** `configs/pipelines/embed_mpnet_chunk.yaml` and `embed_bge_chunk.yaml` (`text_variant: description_only`, `chunking: true`, `batch_size: 32`).
- [x] **Embed on the GPU** from the new union ingest (check `torch.cuda.is_available()` first, as in 11.9):
  - `mpnet-chunk` and `bge-chunk` (chunked, `description_only`);
  - `mpnet-trunc` (11.9 config with `text_variant=description_only`; isolates chunking);
  - `SecRoBERTa` on `description_only` with `embed.yaml` (isolates the text change; same encoder as 11.9).
  - Checks per store: same node index as the new ingest, no NaN, no zero rows other than empty texts, `n_chunks` distribution; for the chunked stores, vectors of nodes with one chunk equal the `mpnet-trunc` / 11.9 behaviour (max difference); repeat one store and compare (max difference).
- [x] **Config** `configs/experiments/attack_cwe_ent_desc_eval.yaml`, with the six evaluations of 11.9. Methods (all pinned by run id): the four new stores; the 11.9 stores on the `description` text as references (`mpnet`, `bge`, SecRoBERTa); `random-rand42/43/44` (same node set).
- [x] **Run** one `evaluate` per gold source with the new ingest runs.
- [x] **Checks:** the 11.9 reference rows, scored on the new ingests, reproduce the 11.9 numbers exactly (max difference 0.0, equal `n_queries`); a repeat of the union run is identical; `n_queries` is equal across methods.
- [x] **Metrics and tables:** the same as 11.9 (Hits@1, Hits@10, MRR; P, R, F1 with predicted / correct / gold counts; set-based Hits / Recall / Precision at k with the ceiling; lift over random; cosine only, CSLS not applicable). New row set in `scripts/unsup_summary.py`.
- [x] **Report, without tuning:**
  - effect of chunking: `mpnet-chunk` against `mpnet-trunc` (same text), and the number of chunked nodes;
  - effect of the text: `mpnet-trunc` and SecRoBERTa on `description_only` against their 11.9 rows on `description`;
  - the order of mpnet, bge and SecRoBERTa on the new text, and the hub check;
  - limits: one run per encoder; `description_only` removes the CWE name, which may carry signal; the weighting choice; gold is transitive through CAPEC; the pool is optimistic.
- [x] **Execution log** (last task): `dev_docs/execution_logs/sprint_11_10.md` with the ingest and embed run ids, checks, evaluate run ids, tables, report, commands and limits. Then mark the 11.10 checkboxes.

## Phase 11.11: Dual-encoder GCN on chunked mpnet text vectors

Question: with the same training recipe, does a dual-encoder GCN built on the chunked mpnet text vectors of 11.10 do better than the one built on SecRoBERTa, and does either beat its own frozen text vectors? This repeats Phases 11.1, 11.2 and 11.4 with a different text store: `sentence_transformer-multi_qa_mpnet_dot_chunk_desc` (run `20261027T194928Z-8cf38e55`; `description_only` text, whole description embedded as token-weighted chunk vectors). It starts after 11.10.

**Fixed for the comparison:** everything in the "Fixed settings" table (DeepWalk graph store, `rank_topk` with k = 5, `unsup_u2`, `DualEncoder` with GCN towers, hidden / out 128, dropout 0, Adam lr 1e-3, 100 epochs, seed 42, `triplet` margin 1.0 / `infonce` temperature 0.1, `alpha = 0.5`). The pseudo-pairs differ with the text store, because `rank_topk` takes its top-5 from the text vectors; that is part of "using a different text encoder", not a second change.

**Same-text control:** the 11.6 SecRoBERTa duals were trained on the older `description` text (CWE name and extended description included), so a comparison of those with the mpnet duals mixes encoder and text. The control trains the same two SecRoBERTa duals on the 11.10 SecRoBERTa store on `description_only` (`hf_mean_pool-pt_SecRoBERTa_desc`, run `20261027T200841Z-09088f9f`). Only the encoder then differs between the mpnet and control duals.

| Run | Text store | Loss | Model id | Sampler id |
|---|---|---|---|---|
| mpnet-chunk triplet | `…multi_qa_mpnet_dot_chunk_desc` | `triplet` | `ent-dual-gcn-triplet-mpnetc-s42` | `s11c-rank_topk-k5` |
| mpnet-chunk InfoNCE | same | `infonce` | `ent-dual-gcn-infonce-mpnetc-s42` | `s11c-rank_topk-k5` |
| control triplet | `hf_mean_pool-pt_SecRoBERTa_desc` | `triplet` | `ent-dual-gcn-triplet-srbd-s42` | `s11d-rank_topk-k5` |
| control InfoNCE | same | `infonce` | `ent-dual-gcn-infonce-srbd-s42` | `s11d-rank_topk-k5` |

- [x] **Inputs:** confirm the two text stores above (pinned run ids), the DeepWalk store `20261027T135210Z-3b18ba0c` and the 11.10 union ingest `20261027T194643Z-5d3f8baf`. The DeepWalk store was built from the 11.6 union ingest; the 11.10 ingests have the same node and edge ids (checked in 11.10) and DeepWalk does not read text, so it stays valid. Nothing is re-ingested or re-embedded here.
- [x] **Pseudo-pairs** (`training_data`), once per text store, as in 11.1. Pin both stores by run id so a later embed cannot change the input:
  ```
  uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
    --set training_data.inputs.text_embeddings.embedder_id=<text embedder id> \
    --set training_data.inputs.text_embeddings.run=<text embed run id> \
    --set training_data.inputs.graph_embeddings.run=20261027T135210Z-3b18ba0c \
    --set training_data.sampler.sampler_id=<s11c|s11d>-rank_topk-k5
  ```
  Check each manifest: universe size (expect the same 2,927 nodes as in 11.1; if it differs, find out why before training), example count, seed 42, text embedder id and run, `held_out` false for all.
- [x] **Pseudo-pair diagnostics** (read-only; the gold is used for reporting, not for training): overlap of the cross-graph pairs with the union gold pairs, the share of technique-side positives taken by the five most frequent CWEs, and the share that are CWE-12xx/13xx, for both text stores, next to the 11.6 SecRoBERTa values (22 of 14,084 pairs, 31%, 53%).
- [x] **Train** (`gcl_train`), four runs. It takes the text and graph embeddings from the pinned `training_data` run, so no embedder is set here. Use the CPU with one thread, as in 11.6, because that was the only setting where the same seed gave identical embeddings; the towers are small, so the GPU is not needed:
  ```
  RUN=<training_data run id of that text store>
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
    --set seed=42 --set device=cpu --set gcl_train.model.model_id=ent-dual-gcn-<loss>-<mpnetc|srbd>-s42 \
    --set gcl_train.inputs.training_data_sampler_id=<s11c|s11d>-rank_topk-k5 \
    --set gcl_train.inputs.training_data_run=$RUN --set gcl_train.loss.name=<triplet|infonce>
  ```
  Check each model directory as in 11.2: manifest (encoder `gcn`, loss, seed, `training_data_run`, `universe_sha256`), `metrics.jsonl` (100 rows, loss lower at epoch 100 than at epoch 1), `node_embeddings/` (one row per universe node), and that the input dimension is the text dimension (768) plus the DeepWalk dimension. Rerun one model with the same seed and confirm identical node embeddings.
- [x] **Config** `configs/experiments/attack_cwe_ent_mpnet_dual_eval.yaml`, with the six evaluations of 11.9 / 11.10 (pinned run ids):
  - **mpnet-chunk:** `ent-mpnet-chunk` (frozen), `ent-d-mpnetc-trip`, `ent-d-mpnetc-nce` (the new duals).
  - **Same-text control:** `ent-srb-new` (frozen SecRoBERTa, `description_only`), `ent-d-srbd-trip`, `ent-d-srbd-nce`.
  - **Old-text references:** `ent-s-r1` (frozen SecRoBERTa), `ent-d-trip` and `ent-d-nce` (SecRoBERTa duals `…-s42b`, pinned to `20261027T141524Z-5ac0fba7` and `20261027T141732Z-3f70b1a2`).
  - `ent-rand-s42/43/44`.
  - Main table: 2 encoders × {frozen, triplet, InfoNCE} on the same `description_only` text (mpnet-chunk against control); the old-text rows are references.
- [x] **Run** one `evaluate` per gold source with its 11.10 ingest run (stix `20261027T194605Z-44eeb7c4`, capec `…194618Z-cba9bfdb`, intersection `…194631Z-950ba779`, union `…194643Z-5d3f8baf`).
- [x] **Checks:** the old-text references (SecRoBERTa frozen, both duals, random) reproduce the 11.7 numbers exactly (max difference 0.0, equal `n_queries`); this holds when all duals share the 2,927-node universe, since the candidate set depends on it, otherwise compare on the 11.7 query set and say so. A repeat of the union run is identical; `n_queries` is equal across all methods.
- [x] **Metrics and tables:** the same as 11.9 (ranking, assignment with counts, set-based, lift over random), plus a "gain from training" column: dual minus frozen, within each encoder and text. Add the row set `attack_ent_mpnet_dual` to `scripts/unsup_summary.py`. Cosine only; CSLS is not applicable, as in 11.9.
- [x] **Report, without tuning:**
  - whether each mpnet-chunk dual beats frozen mpnet-chunk, in either view and each source;
  - whether the mpnet-chunk duals beat the same-text control duals, and whether the encoder gap seen in 11.10 gets smaller or larger after training;
  - the effect of the text on the SecRoBERTa duals (control against the 11.6 duals);
  - which loss is higher;
  - the hub check: the five most frequent CWEs in the top-10 lists of each dual;
  - limits: single seed; the pseudo-pairs differ between encoders; the gold pairs play no part in training; gold is transitive through CAPEC; the pool is built from gold and is optimistic.
- [x] **Execution log** (last task of this phase): write `dev_docs/execution_logs/sprint_11_11.md` with the run ids, pseudo-pair diagnostics, training losses at epochs 1 and 100, determinism check, evaluate run ids per source, checks, tables, report, commands and limits. Then mark the 11.11 checkboxes.

## Phase 11.12: Ablation of the mpnet-chunk InfoNCE dual (single-branch encoders)

Question: where does the gain of `ent-dual-gcn-infonce-mpnetc-s42` (11.11) come from: the text branch, the DeepWalk branch, the graph convolution, or their combination? Everything is held fixed except the encoder: the same pseudo-pairs (`s11c-rank_topk-k5`, run `training_data` of 11.11), `infonce` (temperature 0.1), `alpha = 0.5`, hidden / out 128, Adam lr 1e-3, 100 epochs, seed 42, CPU with one thread. The loss still uses the text and the graph positives / negatives of the pseudo-pairs; only what the encoder is allowed to see changes. No other loss, no triplet, no new seeds.

`DualEncoder` has no single-branch mode, so this phase needs one small code change (task 1). It must leave every existing run unchanged.

| Run | What the encoder sees | Structure | Model id |
|---|---|---|---|
| A0 full dual (reference, from 11.11) | mpnet-chunk + DeepWalk | two GCN towers + `fc1`/`fc2` skips + `fc3` | `ent-dual-gcn-infonce-mpnetc-s42` |
| A1 text-only GCN | mpnet-chunk | text GCN tower (768→128→128) + `fc1` skip → `fc3` | `ent-abl-text-gcn-nce-mpnetc-s42` |
| A2 graph-only GCN | DeepWalk | graph GCN tower (128→128→128) + `fc2` skip → `fc3` | `ent-abl-graph-gcn-nce-mpnetc-s42` |
| A3 text-only, no GCN | mpnet-chunk | `fc1` → `fc3` (no message passing) | `ent-abl-text-lin-nce-mpnetc-s42` |
| A4 graph-only, no GCN | DeepWalk | `fc2` → `fc3` (no message passing) | `ent-abl-graph-lin-nce-mpnetc-s42` |

A1 and A2 are the requested ablations. A3 and A4 are controls that separate the learned projection from the graph convolution (A1 − A3 and A2 − A4 are the effect of message passing; A0 − A1 and A0 − A2 are the effect of adding the other branch). Because two graphs are disjoint components and DeepWalk vectors of the two graphs live in unrelated spaces, A2 and A4 can align the graphs only through what InfoNCE learns from the mpnet pseudo-pairs; they never read text at the input. Expect them to be weak. That is the point of the control, not a failure.

- [x] **Code (minimal):** add `model.branch: both | text | graph` (default `both`) and `model.message_passing: true | false` (default `true`) to the `gcn` `DualEncoder` in `src/cga/pipelines/gcl_train/models.py` and `build_dual_encoder`, read in `pipeline.py` with `model_cfg.get(...)`. A missing key must build the same model as today (same parameter shapes and creation order, so seed 42 still gives the same weights). The unused tower and skip are not created. Update the docstrings, and ask before changing the manifest schema: record `branch` and `message_passing` only if the manifest already stores the model config; otherwise keep them in `config.snapshot` and the model id.
- [x] **Tests:** unit tests for the four new shapes and parameter counts, that an unchanged config still gives 328,576 parameters, that `branch: text` output does not depend on `struct_emb` (change the DeepWalk input, the output is identical) and `branch: graph` does not depend on `text_emb`, and that `message_passing: false` does not depend on `edge_index`. Run `uv run pytest tests/unit -q`.
- [x] **Inputs:** reuse the 11.11 `training_data` run `s11c-rank_topk-k5/20261027T205523Z-ac890ca3` as it is (no new pseudo-pairs). Confirm that its manifest still lists the mpnet-chunk store `…20261027T194928Z-8cf38e55` and DeepWalk `…20261027T135210Z-3b18ba0c`.
- [x] **Reproduction gate (before any new run):** retrain A0 with the new code and the unchanged config; node embeddings must equal those of the 11.11 model `ent-dual-gcn-infonce-mpnetc-s42` (`20261027T205648Z-2aeb3259`) exactly. If they differ, fix the code before training anything else.
- [x] **Train** (`gcl_train`), four runs A1–A4:
  ```
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
    --set seed=42 --set device=cpu --set gcl_train.loss.name=infonce \
    --set gcl_train.model.model_id=<model id> \
    --set gcl_train.model.branch=<text|graph> --set gcl_train.model.message_passing=<true|false> \
    --set gcl_train.inputs.training_data_sampler_id=s11c-rank_topk-k5 \
    --set gcl_train.inputs.training_data_run=20261027T205523Z-ac890ca3
  ```
  Check each model directory as in 11.2 (100 metric rows, loss lower at epoch 100 than epoch 1, one embedding row per universe node of 2,927, parameter count matches the table in the tests). Record loss at epochs 1 and 100 for A0–A4 side by side. Rerun A1 with the same seed and confirm identical node embeddings.
- [x] **Config** `configs/experiments/attack_cwe_ent_mpnet_ablation_eval.yaml` with the six evaluations of 11.11 (pinned run ids, same 11.10 ingests):
  - **Ablation rows:** `ent-a0-full` (A0), `ent-a1-text`, `ent-a2-graph`, `ent-a3-text-lin`, `ent-a4-graph-lin`.
  - **Frozen references:** `ent-mpnet-chunk` (frozen text, no training), the DeepWalk store alone (`deepwalk-deepwalk`, the Sprint 9.4 reference, frozen), `ent-rand-s42/43/44`.
  - All trained rows share the 2,927-node universe, so the candidate set and `n_queries` match 11.11.
- [x] **Run** one `evaluate` per gold source (stix, capec, intersection, union) with the 11.10 ingests listed in 11.11. Add the row set `attack_ent_mpnet_ablation` to `scripts/unsup_summary.py`, with groups: frozen, A1/A2 (GCN), A3/A4 (no GCN), A0.
- [x] **Checks:** A0 and `ent-mpnet-chunk` rows equal the 11.11 numbers exactly (max difference 0.0, equal `n_queries`); a repeat of the union run is identical; `n_queries` equal across all methods. Frozen DeepWalk may need a separate evaluate run if it does not cover the 2,927-node universe; if so, say that it is on a different candidate set.
- [x] **Metrics and tables:** the same as 11.11 (ranking, assignment with predicted / correct / gold counts, set-based, lift over random; cosine only, CSLS not applicable), full and pool views. Add two difference columns: "branch effect" (A0 minus A1, A0 minus A2) and "GCN effect" (A1 minus A3, A2 minus A4).
- [x] **Report, without tuning:**
  - whether A1 (text only) matches A0, which tells whether DeepWalk adds anything to the 11.11 gain;
  - whether A2 (graph only) is above random and above frozen DeepWalk, which tells whether the mpnet pseudo-pairs can carry alignment into the structural space;
  - whether the graph convolution helps for each branch (A1 against A3, A2 against A4);
  - whether A1 and A3 beat frozen `ent-mpnet-chunk`, i.e. whether the 11.11 gain needs the graph at all;
  - hub check: the five most frequent CWEs in the top-10 lists for A1–A4;
  - limits: single seed (differences of about 0.01 are not tested against seed noise); one pseudo-pair draw; the loss still contains the graph-side terms for the single-branch runs; pool is built from gold and is optimistic; gold is transitive through CAPEC.
- [x] **Execution log** (last task of this phase): write `dev_docs/execution_logs/sprint_11_12.md` with the code change, run ids, reproduction gate result, training losses at epochs 1 and 100, determinism check, evaluate run ids per source, checks, tables, report, commands and limits. Then mark the 11.12 checkboxes.

## Phase 11.13: Pseudo-pair `top_k` and loss ablation (pool view, set-based, k = 10)

Question: for the mpnet-chunk dual (A0), text-only GCN (A1) and graph-only GCN (A2) of 11.12, do a larger pseudo-pair `top_k` (5 against 10) and the triplet loss (against InfoNCE) change the per-technique result? Compact: one table, no per-source or full-view tables.

**Fixed protocol:** candidate view `pool` only (`target_pool_union.json`); gold source `union` only (11.10 ingest `20261027T194643Z-5d3f8baf`); `gold_set_retrieval`, split `test`, one query per technique, cutoff k = 10; report only Hits@1, Hits@5, Hits@10, Recall@10, Precision@10. Cosine only.

**Fixed for all trained variants:** mpnet-chunk text store `20261027T194928Z-8cf38e55`, DeepWalk store `20261027T135210Z-3b18ba0c`, `unsup_u2` (hidden / out 128, dropout 0, Adam lr 1e-3, 100 epochs), `alpha 0.5`, seed 42, CPU with one thread, `triplet` margin 1.0, `infonce` temperature 0.1. `rank_topk` takes `k` for the positives and the (bottom-`k`) negatives together, as in 11.1; text and graph sides use the same `k`.

**Varied:** `k` in {5, 10} (new sampler id `s11c-rank_topk-k10`; k = 5 reuses `s11c-rank_topk-k5`, run `20261027T205523Z-ac890ca3`); loss in {InfoNCE, triplet}; model in {A0, A1, A2}. Rows that already exist with the identical configuration are reused (their retrains are bit-identical, 11.12): k = 5 with InfoNCE for A0, A1, A2 and k = 5 with triplet for A0. Eight runs are new.

| Row | Model | `k` | Loss | Model id | Source |
|---|---|---|---|---|---|
| mpnet-chunk | frozen | – | – | – | store `20261027T194928Z-8cf38e55` |
| DeepWalk | frozen | – | – | – | store `20261027T135210Z-3b18ba0c` |
| A0 / A1 / A2 | `branch` both / text / graph | 5 | InfoNCE | 11.11 / 11.12 models | reuse |
| A0 | both | 5 | triplet | `ent-dual-gcn-triplet-mpnetc-s42` | reuse (11.11) |
| A1, A2 | text, graph | 5 | triplet | `ent-p13-{a1,a2}-trip-k5-mpnetc-s42` | new |
| A0 / A1 / A2 | both / text / graph | 10 | InfoNCE | `ent-p13-{a0,a1,a2}-nce-k10-mpnetc-s42` | new |
| A0 / A1 / A2 | both / text / graph | 10 | triplet | `ent-p13-{a0,a1,a2}-trip-k10-mpnetc-s42` | new |

- [x] **Pseudo-pairs, k = 10:** `training_data` with `--set training_data.sampler.k=10 --set training_data.sampler.sampler_id=s11c-rank_topk-k10`, the same pinned text, graph and ingest runs as `s11c-rank_topk-k5`, seed 42. Check universe 2,927, examples 29,270 (10 per anchor), `held_out` false, and the share of gold pairs among the distinct cross-graph pairs next to k = 5 (62 of 13,828).
- [x] **Train** the eight new runs with the 11.12 command (`gcl_train.model.branch`, `message_passing=true`, `gcl_train.loss.name`, `gcl_train.inputs.training_data_sampler_id`, `…training_data_run`, `OMP_NUM_THREADS=1`). Check each: 100 metric rows, loss lower at epoch 100, 2,927 embedding rows, the resolved config shows the intended `k`, loss, branch and training data run. Record the exact configuration of every run in the log (table of model id, run id, sampler id, training data run, loss and its parameters, branch, parameters).
- [x] **Config** `configs/experiments/attack_cwe_ent_p13_eval.yaml`: the 14 rows above (the 6 reused models pinned by run id), one evaluation `set_pool` with `ks: [1, 5, 10]`, `split: test`, `target_pool` as in 11.11. Run `evaluate` once on the union ingest.
- [x] **Checks:** reused and frozen rows reproduce the 11.12 values (Hits@10, Recall@10, Precision@10 of `set_pool`) exactly; `n_queries` is 183 for all rows.
- [x] **One table** (rows as above; columns Hits@1, Hits@5, Hits@10, Recall@10, Precision@10), and a short report: effect of `k` (5 against 10) and of the loss for each of A0, A1, A2, and whether the 11.12 ordering (A1 at least A0 above A2) holds. State the limits: single seed, one gold source and view, pool optimistic, differences of a few hundredths not tested against seed noise.
- [x] **Execution log** (last task): `dev_docs/execution_logs/sprint_11_13.md` with the run configuration table, run ids, checks, the table, the report, commands and limits. Then mark the 11.13 checkboxes.

## Phase 11.14: Negative-sampling ablation (positive-only and cross-modal hard negatives)

Question: for A0 (full dual), A1 (text-only GCN) and A2 (graph-only GCN) with `top_k = 5` and InfoNCE, how much of the 11.13 result depends on the negatives? The positives are fixed (the 11.11 pseudo-pairs, `s11c-rank_topk-k5`); only the negative side changes. Narrow: no new architecture, no sweep, one table.

**Reference (reused, no training):** the 11.13 rows `a0-nce-k5`, `a1-nce-k5`, `a2-nce-k5` (explicit bottom-5 negatives, the least similar nodes, i.e. easy negatives), plus frozen mpnet-chunk and frozen DeepWalk.

**Variants (each for A0, A1, A2; 6 new trainings):**

1. **Positive-only InfoNCE** (`gcl_train.loss.name=infonce_inbatch`, training data run unchanged, `20261027T205523Z-ac890ca3`). The text term uses the cross-graph text positive of each row, the graph term the same-graph DeepWalk positive. No explicit negative is used (the negative columns of the examples are not read). The only negatives are in-batch: for a row, the other rows' positives of the same side and the same graph role as its own positive (full batch, all 14,635 rows). Masked out of the denominator, because they are not negatives: the node equal to the row's own positive (duplicates counted once) and every other positive of the same anchor.
2. **Positive + cross-modal hard negative InfoNCE** (`infonce`, the existing loss, with a new training data run). For every positive the row carries one explicitly sampled hard negative (5 per anchor and side, as before):
   - *Text side:* for anchor `a` and text positive `p` (top-5 by mpnet cosine among the other graph's nodes), the candidate pool is the nodes ranked 6 to `5 + M` by text similarity to `a` (plausible by text, not positives). The negative is the pool node with the **lowest DeepWalk cosine to `p`** (both are in the same graph, so the DeepWalk similarity is defined). Positives of one anchor get distinct negatives (greedy in positive rank order, ties by column index).
   - *Graph side:* for anchor `a` and DeepWalk positive `q` (top-5 same-graph neighbours), the pool is the same-graph nodes ranked 6 to `5 + M` by DeepWalk similarity to `a`; the negative is the pool node with the **lowest text cosine to `q`**.
   - `M = 50` (fixed, not tuned). The anchor itself and any positive of the anchor are excluded. No gold pair is read.

Everything else as in 11.13: mpnet-chunk text store `20261027T194928Z-8cf38e55`, DeepWalk `20261027T135210Z-3b18ba0c`, `unsup_u2`, InfoNCE temperature 0.1, `alpha 0.5`, hidden / out 128, Adam 1e-3, 100 epochs, seed 42, CPU with one thread.

**Open interpretation (please confirm before running):** "hard negatives from nodes with low structural/text similarity" is read above as *near the anchor in its own modality but far from the positive in the other modality*. The simpler reading (negatives = the lowest other-modality similarity to the anchor, as in the existing bottom-k) would give easy negatives, not hard ones. If the intended meaning differs, change the pool definition only; nothing else in the phase depends on it.

| Row | Model | Negatives | Loss | Source |
|---|---|---|---|---|
| mpnet-chunk, DeepWalk | frozen | – | – | stores, as in 11.13 |
| A0 / A1 / A2 | both / text / graph | bottom-5 explicit (reference) | `infonce` | 11.13 `a*-nce-k5`, reused |
| A0 / A1 / A2 | both / text / graph | none explicit (in-batch only) | `infonce_inbatch` | new: `ent-p14-{a0,a1,a2}-pos-k5-mpnetc-s42` |
| A0 / A1 / A2 | both / text / graph | cross-modal hard, `M = 50` | `infonce` | new: `ent-p14-{a0,a1,a2}-hard-k5-mpnetc-s42`, sampler `s11c-rank_topk-k5-xhard` |

- [x] **Sampling (configurable):** add to the `rank_topk` sampler `negatives: cross_modal_hard` with `hard_pool` (default 50), `k`-aware (`hard_pool >= 1`, pool size capped by the node count). It needs same-graph text similarity (graph_1 x graph_1, graph_2 x graph_2) from the same text store: add it next to `compute_text_similarity` in `similarity.py` and pass it to the sampler. Defaults leave every existing run unchanged (the existing test suite and the 11.13 k = 5 `examples.parquet` must reproduce exactly). The example schema is unchanged (one negative per positive).
- [x] **Loss (configurable):** add `infonce_inbatch` to `gcl_train` (`losses.py`, a branch in `trainer.py`/`pipeline.py` that passes node indices for the masks, as `infonce_full` does). It must not read `text_neg` / `graph_neg`, must use the same temperature, chunking and normalisation as `infonce`, and write the same `metrics.jsonl` fields.
- [x] **Tests:** *hard negatives:* a negative is never the anchor or one of its positives, lies in the intended graph, is within ranks 6 to `5 + M` of the anchor's own-modality ranking, and is the brute-force argmin of the other-modality similarity to its positive on a small fixture; no held-out node is drawn; same seed gives identical output; the positive columns equal the `bottomk` run's (as a set per anchor); the old options are unchanged. *Positive-only:* the loss is unchanged when the negative columns are changed; on a hand-computed fixture it equals the plain cross-entropy over the masked candidates; duplicates of the row's own positive and the anchor's other positives are masked; the value is finite and reproducible. Run `uv run pytest tests/unit -q`.
- [x] **Training data, hard negatives:** one `training_data` run with `--set training_data.sampler.negatives=cross_modal_hard --set training_data.sampler.hard_pool=50 --set training_data.sampler.sampler_id=s11c-rank_topk-k5-xhard`, same pinned inputs as `s11c-rank_topk-k5`. Check universe 2,927, examples 14,635, `held_out` false. Report (read-only; the gold is not used in training): the share of sampled negatives that are gold partners of their anchor (false negatives) for the hard variant against the bottom-5 reference; the mean DeepWalk cosine (text side) and mean text cosine (graph side) between negative and positive for both; the number of distinct negative nodes.
- [x] **Train** the six new models (command as 11.13 with `gcl_train.loss.name=infonce_inbatch` and the existing sampler run for the positive-only variant, `infonce` and the new sampler run for the hard variant). Check each: 100 metric rows, loss lower at epoch 100, 2,927 embedding rows, finite, resolved config shows loss, branch and training data run. Rerun one model per variant (A1) and confirm identical node embeddings.
- [x] **Evaluate:** config `configs/experiments/attack_cwe_ent_p14_eval.yaml` with the 11 rows (frozen x2, reference x3 pinned to the 11.13 runs, new x6), one evaluation `set_pool` (`ks: [1, 5, 10]`, `split: test`, union pool). One `evaluate` run on the union ingest `20261027T194643Z-5d3f8baf`, plus a repeat.
- [x] **Checks:** frozen and reference rows reproduce the 11.13 values exactly; the repeat run is identical; `n_queries` is 183 for all rows.
- [x] **One table:** the 11 rows, columns Hits@1, Hits@5, Hits@10, Recall@10, Precision@10. Short report: effect of the negatives for each of A0, A1, A2 (hard against reference, positive-only against reference, hard against positive-only), the false-negative rate of the hard variant, and whether the 11.13 branch order (A1 above A0 above A2) holds. Limits: single seed; union gold and pool view only; one `M`; the pool view is optimistic.
- [x] **Execution log** (last task): `dev_docs/execution_logs/sprint_11_14.md` with the sampler and loss definitions as implemented, tests, run ids and configuration of every run, diagnostics, checks, the table, report, commands and limits. Then mark the 11.14 checkboxes.

## Out of scope

- Plots, fusion rows, and any change to the existing `gold_alignment` / `gold_alignment_prf` scoring.
- Other text encoders (except `multi-qa-mpnet-base-dot-v1` and `BAAI/bge-base-en-v1.5` in Phases 11.9–11.11, and SecRoBERTa as the reference), other samplers, GAT, other seeds, hyperparameter changes.
- Phases 11.9–11.11: no fine-tuning of the sentence encoders, no other seeds or loss settings, no query/passage prefixes, no CSLS or centring, no change to any evaluator.
- Phase 11.13: only `k` in {5, 10}, InfoNCE and triplet, seed 42, union gold and the pool view; no other sampler, text store, seed or hyperparameter, no code change. (11.12 itself: only InfoNCE, seed 42 and the mpnet-chunk pseudo-pairs; no GAT, no new text store, no new sampler. The only code change is the `branch` / `message_passing` options of the `gcn` `DualEncoder`; no evaluator changes.)
- Phase 11.14: only A0, A1, A2 with InfoNCE and `top_k = 5` (seed 42, union gold, pool view); the two negative variants above, `M = 50`; no other architecture, loss, sampler option, `top_k`, text store or sweep. Code changes are limited to the `cross_modal_hard` negatives option (with same-graph text similarity) and the `infonce_inbatch` loss; no evaluator changes.
- Phase 11.10 changes only the `description_only` text variant, the `chunking` option of the sentence embedder and the summary-script row set; no evaluator changes.
- Phases 11.1–11.5: no code changes. Phase 11.6 changes only the CWE description parser and `gcl_train` seeding; Phase 11.7 adds only the new evaluator, its metric helper and the summary-script tables.
