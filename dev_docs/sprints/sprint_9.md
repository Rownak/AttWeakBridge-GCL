# Sprint 9: Unsupervised, dataset-agnostic ablations (encoder × pseudo-pairs × loss)

## Objective

Measure how the encoder, the pseudo-pair strategy and the loss affect cross-graph alignment **without any gold alignment in training**.
Every component already exists except the leakage protocol and two sampler options; this sprint adds those,
then runs a small ablation matrix. Gold-scored results are produced for NCIT–DOID only (3,280 held-out test pairs). ATT&CK–CWE runs the
same pipeline end-to-end with gold-free outputs, so its gold set can be added later as a config change.

Branch: `v1.3_ncit_doid_biogitom_bench` (already contains the Sprint 8 `gold_alignment_prf` evaluator, so nothing is cherry-picked).

## Conventions for every phase

- Google-style docstrings, tests in `tests/unit/`, `uv run pytest tests/unit -q` green.
- Each phase ends with `dev_docs/execution_logs/sprint_9_N.md`: method, tables with run ids, decision, caveats.
- Commit prefix `[Sprint-9.N]`. Every new option is default-off, so earlier runs and configs are unchanged.
- No dataset name appears in `src/`; datasets differ only in `configs/datasets/*.yaml`.

## Protocol (no supervision, no leakage)

| Data | Used for | Never used for |
|---|---|---|
| All nodes of both graphs: text, intra-graph edges, DeepWalk vectors (test endpoints included) | features, message passing, pseudo-pairs | — |
| Gold `test` pairs (NCIT–DOID) | final evaluation, read **once** per configuration | pseudo-pairs, training, hyperparameters, thresholds |
| Gold `train` pairs (NCIT–DOID) | nothing | everything, including the evaluator's sibling filter |

- **No split.** Ingest runs with `ingest.split.fractions = [1.0, 0.0, 0.0]`: every node of both graphs is an anchor and a candidate, and no node is held out because it is a gold endpoint.
- **No selection.** With no gold and no split there is nothing to select on, so hyperparameters, epochs and the threshold are **fixed before any run** (shared settings in Phase 9.2) and never tuned. Every matrix row is reported.
- **Guard.** `protocol: unsupervised` (top-level config key) makes `training_data` and `gcl_train` refuse: the `supervised` sampler, `infonce_full` (its softmax mask reads gold pairs), `monitor.enabled`, and any read of `alignments.parquet`.

## Prediction policy (how alignments are produced)

1. Cosine of every gold-test source against every `graph_2` node (global; candidate pools are not used).
2. **Ranking metrics:** `gold_alignment`, `global` mode, Hits@1, Hits@10, MRR; the "other gold targets" sibling filter is restricted to the test split.
3. **Pair set:** `gold_alignment_prf`, top-3 per source, greedy one-to-one, **threshold 0.0**, `biogitom_filters: false` (that filter reads gold train labels).
   Reports P, R, F1, `n_predicted`, `n_correct`, `n_gold`. The threshold is fixed at 0.0 and never tuned.

## Decision gate

| After | Go if | Else |
|---|---|---|
| 9.1 | Tests green; pseudo-pairs are byte-identical with and without `alignments.parquet`; U0 reproduces the frozen-BERT test numbers of a plain `embedding_store` evaluation; ATT&CK–CWE runs end-to-end | Fix before 9.2 |
| 9.3 | U0-SapBERT reproduces the frozen-SapBERT `embedding_store` evaluation and the Sprint 5/6.3 A0 test numbers; the SapBERT store passes the 9.1 copy checks | Fix before reading the matrix |

---

## Phase 9.1: Protocol, datasets and sampler options

**Capability:** any dataset with two graphs and node text can train under the unsupervised protocol and be evaluated on gold test pairs when they exist.

- [x] **Datasets.** `configs/datasets/ncit_doid_unsup.yaml` (alignments `train` + `test`, no `val_split`) and `configs/datasets/attack_cwe_unsup.yaml` (`alignments: null`).
      Ingest each with `ingest.split.fractions=[1.0,0.0,0.0]`; copy the `hf_mean_pool-bert_base` and `deepwalk-deepwalk` stores without `shards/` and verify node order (as in Sprint 6.3).
- [x] **`training_data.held_out_policy`** (`legacy` default | `none`): `none` holds no node out and never reads alignments.
- [x] **`protocol: unsupervised` guard** in `training_data` and `gcl_train` (list above). The manifest summary records `protocol`, `held_out_policy` and the number of pseudo-pairs.
- [x] **`rank_topk` options** (defaults = today's behaviour): `negatives: bottomk | random | hard` (`hard` = text ranks `hard_offset+1 … hard_offset+k`, default offset 10, so the top pseudo-positives are not reused as negatives); `positives: topk | mutual` (keep a top-k positive only if the anchor is also in the positive's top-k).
- [x] **`gold_alignment.sibling_filter_splits`** (default: all splits, as today); the unsupervised configs set `[test]`.
- [x] **Check that `residual_adapter` works with the row losses** (`triplet`, `infonce`); Sprint 5 trained it with `infonce_full` only.
- [x] **`scripts/unsup_summary.py`**: reads one evaluate run, groups seeds, writes `summary.md` and `summary.csv` into that run's directory (mean ± sample std; missing gold columns become `—`).
- [x] **Tests:**
  - with `held_out_policy: none`, pseudo-pairs are identical with and without alignments, and every node is eligible;
  - the guard raises in each refused case;
  - the new sampler options on a hand-built similarity matrix;
  - the sibling filter honours `sibling_filter_splits`.
- [x] **Smoke:** U0 and U2 (1 seed, 20 epochs) on both datasets; NCIT–DOID U0 test numbers equal an unrestricted `embedding_store` evaluation of `bert_base`.

**Acceptance:** gate 9.1. **Training:** smoke only.

---

## Phase 9.2: Ablation matrix and paper table

**Capability:** run the prioritised matrix, read test once, and produce the paper table.

Shared settings: text = frozen `bert_base`; structure = DeepWalk; sampler `rank_topk` k = 5 (bottom-k negatives, top-k positives) unless the row says otherwise; Adam lr 1e-3; 100 epochs (the Sprint 2 sweep setting), last epoch exported; triplet margin 1.0 and InfoNCE temperature 0.1 (pipeline defaults); seeds 42 / 43 / 44. All fixed before any run.

| ID | Prio | Encoder (existing component) | Pseudo-pairs | Loss | Question |
|---|---|---|---|---|---|
| U0 | 1 | frozen BERT (`embedding_store`) | — | — | text-only floor |
| U1 | 1 | BERT + trained projection (`residual_adapter`, `adapter: mlp`, λ 0.25, `alpha: 1.0`) | top-k | triplet | does training on pseudo-pairs improve over frozen BERT? |
| U2 | 1 | BERT + GCN, text and DeepWalk towers (`encoder: gcn`, `alpha: 0.5`; the current method) | top-k | triplet | does graph structure add to U1? |
| U1-I | 2 | as U1 | top-k | InfoNCE (`infonce`) | triplet vs InfoNCE |
| U2-I | 2 | as U2 | top-k | InfoNCE | triplet vs InfoNCE with graph |
| U2-rand | 3 | as U2 | random negatives | triplet | easy vs random negatives |
| U2-hard | 3 | as U2 | hard negatives | triplet | hard negatives |
| U2-mut | 3 | as U2 | mutual top-k positives | triplet | cleaner positives |

- [x] **Train** priority 1, then 2, then 3 (21 seeded runs, no tuning runs); stop at a priority level if time runs out and say so.
- [x] **Evaluate once** on NCIT–DOID test with `configs/experiments/ncit_doid_unsup_test.yaml` (`gold_alignment` + `gold_alignment_prf`); ATT&CK–CWE U0 / U1 / U2 with `attack_cwe_unsup_eval.yaml` (gold-free `neighborhood` only).
- [x] **`summary.md`** from `unsup_summary.py`, plus the paper table below, in `sprint_9_2.md`.

**Acceptance:** every priority-1 and priority-2 row on NCIT–DOID with 3 seeds; the paper table filled; test-read disclosure.
**Decision:** report only.

---

## Phase 9.3: Paper table with SapBERT

**Capability:** the Phase 9.2 table (all eight rows) with the frozen text encoder replaced by SapBERT, so the paper can state whether the 9.2 finding (no unsupervised variant beats the frozen text encoder) depends on the encoder. Same protocol, same fixed settings, same test pairs; only the text embeddings change.

**Why SapBERT.** It is a biomedical entity encoder; the Sprint 5–8 results use it as A0 (frozen) and as the residual base. Its frozen score on NCIT–DOID is far above `bert_base`, so the pseudo-pairs are of a different quality, which is the point of the comparison.

- [ ] **Store.** `artifacts/embeddings/ncit_doid/hf_mean_pool-sapbert/20261009T182919Z-95011340` (the run `LATEST` points to; the sibling run `20261009T185331Z-390d8bc3` has no store files and is not used). Copy it to `artifacts/embeddings/ncit_doid_unsup/hf_mean_pool-sapbert/` without `shards/` with `scripts/copy_embedding_store.py --text-variant label_synonyms_definition` against the `ncit_doid_unsup` ingest (the 9.1 checks: byte equality, manifest hashes, node order, `input_sha256` of 24,227 / 24,227 nodes). The DeepWalk store is the one 9.1 copied.
- [ ] **Keep the BERT lineage untouched.** The `rank_topk-k5*` `training_data` lineages and the `u*-s<seed>` model ids belong to 9.2, and `LATEST` is shared. Give every SapBERT artifact its own id:
  - `training_data.sampler.sampler_id` = `sap-rank_topk-k5[-negrandom|-neghard-off10|-posmutual]`;
  - model ids `sap-u1-s<seed>`, `sap-u1-i-s<seed>`, `sap-u2-s<seed>`, `sap-u2-i-s<seed>`, `sap-u2-rand-s<seed>`, `sap-u2-hard-s<seed>`, `sap-u2-mut-s<seed>`.
  `scripts/run_unsup_matrix.sh` takes the text embedder and a prefix (default: today's BERT behaviour, so 9.2 commands are unchanged) and sets `training_data.inputs.text_embeddings.embedder_id`, `sampler_id` and `gcl_train.inputs.training_data_sampler_id`, `gcl_train.model.model_id` accordingly.
- [ ] **Settings are not re-tuned.** Identical to 9.2: `rank_topk` k 5, Adam 1e-3, 100 epochs, last epoch, margin 1.0, temperature 0.1, seeds 42 / 43 / 44, U1 `residual_adapter` / `mlp` λ 0.25 `alpha 1.0`, U2 `gcn` `alpha 0.5`. SapBERT has the same 768-d output, so no dimension changes. Hyperparameters are not adjusted for SapBERT; every row is reported.
- [ ] **Train** priority 1, 2, 3 (21 NCIT–DOID runs; about 2 h 20 min). ATT&CK–CWE is out of scope: there is no SapBERT store for it and the model is biomedical.
- [ ] **`unsup_summary.py`**: a `--prefix sap-` (and row labels `SapBERT (frozen)`, `SapBERT + projection`, …) so the paper table renders from a run that holds the `sap-*` methods, and a `--compare <bert_run>` option that writes the 9.2 BERT and 9.3 SapBERT tables side by side (per row: MRR and F1, difference of means). Unit test on a fixture.
- [ ] **Evaluate once** with `configs/experiments/ncit_doid_unsup_sapbert_test.yaml`: the same two evaluations as `ncit_doid_unsup_test.yaml` (`gold_alignment` global with `sibling_filter_splits: [test]`, and `gold_alignment_prf` top-3 greedy one-to-one threshold 0, no BioGITOM filters) for `sapbert` (U0) and the 21 `sap-*` models.
- [ ] **Check U0.** U0-SapBERT must equal the frozen-SapBERT row evaluated alone as an `embedding_store` on the same 3,280 pairs (as in 9.1), and, with the default all-splits sibling filter, the A0 numbers of Sprint 5/6.3 on the same test queries (`ncit_doid_srga_full`; A0 test is bit-identical across those lineages).
- [ ] **Execution log** `dev_docs/execution_logs/sprint_9_3.md`: the SapBERT paper table, the BERT-versus-SapBERT table, run ids, test-read disclosure, and the caveat below.

**Caveat to state with the table.** SapBERT was pretrained on UMLS synonym pairs. NCIT and DOID concepts that share a UMLS concept (CUI) were pulled together during that pretraining, so frozen SapBERT carries cross-vocabulary alignment knowledge that `bert_base` does not. It is not gold supervision from this dataset and no gold pair is used by our training, but the table must not claim that the SapBERT rows use no external alignment information.

**Acceptance:** gate 9.3; all eight rows with 3 seeds; the table and the comparison filled; test-read disclosure. **Decision:** report only.

---

## Artifacts, metadata and reproducibility

Paths reuse the current layout, so nothing new is introduced:

```
data/processed/{dataset}/{ingest_run}/                     nodes, edges, splits (all train), alignments
artifacts/embeddings/{dataset}/{embedder_id}/{run}/        frozen BERT / SapBERT, DeepWalk
artifacts/training_data/{dataset}/{sampler_id}/{run}/      examples.parquet (every node eligible), universe, stats
models/gcl/{dataset}/u{ID}-s{seed}/{run}/                  node_embeddings/, checkpoints/, metrics.jsonl, manifest.json
artifacts/evaluations/{dataset}/{run}/                     metrics.parquet, metrics.json, summary.md, summary.csv
```

- Each run's `manifest.json` already holds the resolved config, input run ids and hashes, and the seed; Phase 9.1 adds `protocol`, `held_out_policy` and the pseudo-pair counts.
- Model id = matrix ID + seed (`u2-hard-s43`), so `LATEST` pointers never collide; Phase 9.3 prefixes `sap-` (`sap-u2-hard-s43`) and uses `sap-rank_topk-k5*` sampler ids.
- Reproducibility: seeded samplers and training. GCN kernels are not bit-reproducible on GPU, so trained rows report mean ± sample std over 3 seeds; U0 must reproduce exactly.

## Paper table template

```
Table X. Unsupervised ablation on NCIT–DOID (3,280 held-out gold test pairs; no gold pair used in training; all hyperparameters fixed in advance).
Mean ± std over 3 seeds. P/R/F1: greedy one-to-one over top-3 cosine candidates, threshold 0.

| Variant                          | Pseudo-pairs   | Loss    | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct |
|----------------------------------|----------------|---------|--------|---------|-----|---|---|----|-------|----------|
| BERT (frozen)                    | –              | –       |        |         |     |   |   |    |       |          |
| BERT + projection                | top-k          | Triplet |        |         |     |   |   |    |       |          |
| BERT + projection                | top-k          | InfoNCE |        |         |     |   |   |    |       |          |
| BERT + GCN (text + structure)    | top-k          | Triplet |        |         |     |   |   |    |       |          |
| BERT + GCN (text + structure)    | top-k          | InfoNCE |        |         |     |   |   |    |       |          |
| BERT + GCN                       | random neg.    | Triplet |        |         |     |   |   |    |       |          |
| BERT + GCN                       | hard neg.      | Triplet |        |         |     |   |   |    |       |          |
| BERT + GCN                       | mutual top-k   | Triplet |        |         |     |   |   |    |       |          |
#Gold = 3,280 for every row. ATT&CK–CWE columns are added once its gold set exists.
```

## Out of scope

End-to-end BERT fine-tuning; branch merges; gold-supervised training; the BioGITOM train-label filter; ATT&CK–CWE gold metrics.
