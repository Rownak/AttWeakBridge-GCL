# Sprint 10: ATT&CK–CWE gold-alignment ablation (Enterprise ATT&CK × CWE)

## Objective

Make ATT&CK–CWE gold-scored, then rerun the SecureBERT2 rows of the Sprint 9 ablation on it. DeepWalk rows are left out: the gold is cross-graph, and DeepWalk spaces of two graphs with no shared edges are not comparable. First, a configurable gold-alignment step: the dataset YAML picks one of four ATT&CK→CWE gold CSVs, keeps the provenance of every pair, and assigns the pairs to train/val/test by YAML-set fractions (default 0/0/1: the whole gold set is `test`). Then three rows (frozen SecureBERT2, its GCN, and their score fusion) are scored on the `test` pairs with the NCIT–DOID metrics and prediction protocol.

The four CSVs are four separate gold sets, not parts of one set. Each evaluate run scores exactly one of them, in full; the sources are never mixed or compared as splits of each other.

Nothing here is a new method; every row repeats a Sprint 9.2–9.5 configuration. Only the dataset changes: `data/raw/enterprise_cwe/` (`enterprise-attack-20.0.json`, `cwec_v4.20.xml`).

## Conventions

- Google-style docstrings; `uv run pytest tests/unit -q` green; one log per phase, `dev_docs/execution_logs/sprint_10_N.md`; commit prefix `[Sprint-10.N]`.
- New options are default-off. No dataset name in `src/`. `alignments.parquet` keeps its schema.
- Protocol as in Sprint 9: `protocol: unsupervised`, `held_out_policy: none`, node split `[1.0, 0.0, 0.0]`, fixed settings, no tuning. Gold is read only by `evaluate`.

## Gold sources

The CSVs are in `data/raw/capec/`. Each row is one ATT&CK → CAPEC → CWE path. Gold pairs are the distinct `(attack_id_current, cwe_id)`.

| Key | File | Rows | Distinct pairs | Techniques | CWEs |
|---|---|---|---|---|---|
| `stix` | `1_attack_capec_cwe__from_stix_repo.csv` | 376 | 300 | 107 | 77 |
| `capec` | `2_attack_capec_cwe__from_capec_xml.csv` | 657 | 589 | 187 | 149 |
| `union` | `3_attack_capec_cwe__union.csv` | 800 | 656 | 204 | 151 |
| `intersection` | `4_attack_capec_cwe__intersection.csv` | 233 | 229 | 84 | 68 |

Counts are before resolving ids against the Enterprise bundle and CWE v4.20. The gold is 1:n (about 3 CWEs per technique).

## Decision gate

| After | Go if | Else |
|---|---|---|
| 10.1 | Tests green; with the default fractions `[0, 0, 1]` every kept pair is `test`; with non-default fractions the same seed gives the same split and no technique family is in two splits; every kept pair has a provenance row | Fix before 10.2 |
| 10.2 | The four ingests have identical `nodes`/`edges`/`texts` hashes and differ only in `alignments`; resolved and dropped counts are logged per source | Fix before training |
| 10.4 | Frozen SecureBERT2 gives the same numbers in every evaluate run that includes it; fusion rows reproduce their member cosines (fusion unit test) | Fix before reading the table |
| 10.6 | The pool has 151 CWEs and contains every gold target of all four sources; with no pool the new evaluate runs reproduce the 10.4 and 10.5 numbers exactly; the random baseline is stable across seeds | Fix before reading the pool tables |

---

## Phase 10.1: Gold-source selection, provenance and split

**Capability:** `ingest` reads one gold CSV chosen in the dataset YAML. It writes `alignments.parquet` (unchanged schema, with `split` filled from the YAML fractions) and a sidecar `alignment_provenance.parquet`.

```yaml
# configs/datasets/attack_cwe_ent.yaml (excerpt)
alignments:
  equivalence:
    source: union                     # stix | capec | union | intersection
    sources:
      stix:         capec/1_attack_capec_cwe__from_stix_repo.csv
      capec:        capec/2_attack_capec_cwe__from_capec_xml.csv
      union:        capec/3_attack_capec_cwe__union.csv
      intersection: capec/4_attack_capec_cwe__intersection.csv
    columns: {source: attack_id_current, target: cwe_id}
    split: {fractions: [0.0, 0.0, 1.0], seed: 42, group_regex: '^(T\d+)'}   # [train, val, test]; default = all test
```

- [x] **Generic gold reader** (`src/cga/data/gold_csv.py`): read the CSV, map the two columns to node ids by `external_id` (ATT&CK `Txxxx[.yyy]`; CWE `CWE-n`), and dedupe to distinct pairs.
  - **Dropped rows:** counted by reason (`source_not_in_graph`, `target_not_in_graph`, `duplicate`) in the ingest manifest summary, together with `gold_source` and the file's sha256. Expected drops: one mobile technique, some deprecated ids, and CWEs missing from v4.20.
  - **Wiring:** the `attack_cwe` adapter's `iter_alignments` calls the reader when `alignments` is set. With `alignments: null`, ingest behaves as today.
- [x] **Provenance sidecar** `alignment_provenance.parquet`, one row per kept pair:
  - **Columns:** `alignment_id`, `gold_source`, `source_file`, `source_sha256`, `edge_source` (CSV value, e.g. `capec-3.9+v12.1`), `capec_ids` (list), `attack_id_cited`, `n_csv_rows`.
  - **Why a sidecar:** the `alignments.parquet` schema is shared by every dataset and stays unchanged.
- [x] **Grouped, seeded split** (`assign_grouped_split`), driven by `split.fractions` in the YAML:
  - **Default `[0.0, 0.0, 1.0]`:** every pair of the selected source is `test`. This is what Sprint 10 uses: training is unsupervised and never reads gold, so there is nothing to hold out, and the whole gold set is scored.
  - **Non-default fractions** are supported for later supervised work. Pairs are grouped by technique family (`group_regex` keeps the root id, so `T1003` and `T1003.001` share a group) and each group is assigned by a seeded hash of its group id, not by row order. Same seed gives the same assignment.
  - **Sources are separate gold sets:** the split is computed inside the selected source only. Nothing is shared or compared between sources, even though the hash makes a technique's split independent of the source.
- [x] **Focused tests** (`tests/unit/data/test_gold_csv.py`, `tests/unit/pipelines/ingest/`):
  - **(a) source selection:** each `source` key reads its own file, and an unknown key raises `ConfigError`;
  - **(b) split:** with default fractions every pair is `test`; with non-default fractions no group appears in two splits, the same seed gives the same split and a different seed gives a different one;
  - **(c) provenance:** one provenance row per alignment, ids match, and the sha256 equals the file's;
  - **(d) drops:** unresolved ids are dropped and counted, never guessed.

**Acceptance:** gate 10.1.

---

## Phase 10.2: Ingest and embeddings

- [x] **Dataset config** `configs/datasets/attack_cwe_ent.yaml`:
  - **Sources:** `adapter: attack_cwe`, `root: ${paths.raw}`, ATT&CK `enterprise_cwe/enterprise-attack-20.0.json`, CWE `enterprise_cwe/cwec_v4.20.xml` (`version: cwec_v4.20`).
  - **Shared settings:** text variants as in `attack_cwe.yaml`, `protocol: unsupervised`, and the `alignments` block above.
- [x] **Ingest four times into one lineage**, one run per gold source, with `--set ingest.split.fractions=[1.0,0.0,0.0]` and `--set dataset.alignments.equivalence.source=<key>`. Run `union` last so that `LATEST` (used by training) is `union`. Record the four run ids.
- [x] **Check gate 10.2.** `nodes`, `edges` and `texts` hashes are equal across the four runs, so embeddings and models are trained once and scored against every source.
- [x] **Embed:**
  - **SecureBERT2:** `hf_mean_pool`, `model_ref: cisco-ai/SecureBERT2.0-base`, embedder id `hf_mean_pool-SecureBERT2`, text variant `description`, the same settings as the Sprint 3 ICS store.
  - **DeepWalk:** `deepwalk-deepwalk`, the Sprint 9 settings.
- [x] **Coverage table** in the log: per source, the pairs kept and dropped (by reason), and the number of gold pairs and distinct techniques (all `test` under the default split).

**Acceptance:** gate 10.2; four ingests; two embedding stores.

---

## Phase 10.3: Train the GCN row

Pseudo-pairs come from SecureBERT2 cosine top-k, a cross-graph signal (k 5, sampler `rank_topk`, `held_out_policy: none`), one `training_data` run per seed. All other settings are as in 9.2–9.5: InfoNCE, temperature 0.1, Adam 1e-3, 100 epochs, last epoch, seeds 42/43/44.

| Row | Model | Config (existing options only) |
|---|---|---|
| 2 | GCN(SecureBERT2) | `residual_adapter`, `adapter: gcn`, `residual_floor: false` (the Sprint 5 A12 module) |

- [x] **Train** 1 row × 3 seeds = 3 runs, plus 3 `training_data` runs.
  - **Ids:** model ids `ent-g-sb2-s<seed>`; sampler id `sb2-rank_topk-k5`.
  - **Script:** `scripts/run_unsup_matrix.sh` gains one priority, `7`, for this row.

**Acceptance:** three runs succeed; manifests show `protocol: unsupervised`, `held_out_policy: none`.

---

## Phase 10.4: Evaluate on held-out gold and report

| # | Row | Method |
|---|---|---|
| 1 | Frozen SecureBERT2 | `embedding_store` |
| 2 | GCN(SecureBERT2) | `gcl_run` (10.3) |
| 3 | SecureBERT2 + GCN(SecureBERT2) | `fusion`: SecureBERT2 w 1, row 2 w 0.25 |

- **Fusion weight:** w = 0.25 is fixed in advance from Sprint 9.5 (NCIT–DOID) and is not tuned on ATT&CK–CWE gold. Fusion is score-level and untrained (the `fusion` method kind), the variant that helped in 9.5. The trained residuals of 9.5 (R1–R3) did not beat the frozen encoder and are not repeated.
- **Prediction protocol** (identical to NCIT–DOID; with the default split `test` is the whole selected gold set): queries are the `test` techniques against every CWE node. `gold_alignment` (global, `sibling_filter_splits: [test]`) reports Hits@1, Hits@10 and MRR. `gold_alignment_prf` (top-3, greedy one-to-one, threshold 0.0, `biogitom_filters: false`) reports P, R, F1, `n_predicted`, `n_correct` and `n_gold`.

- [x] **Config** `configs/experiments/attack_cwe_ent_test.yaml`:
  - **Methods:** rows 1–3, with rows 2 and 3 per seed.
  - **Running it:** run `evaluate` once per gold source with `--set evaluate.inputs.ingest_run=<that source's ingest run>`. That makes four evaluate runs, each reading `test` once.
- [x] **Metric-correctness test:** a hand-computed fixture with a 1:n source (two gold CWEs, one ranked above the other) checks Hits@K, MRR, the test-only sibling filter, and P/R/F1/`n_gold` under one-to-one assignment.
- [x] **Tables** via `scripts/unsup_summary.py` with a new `--rows attack_ent` row set (3 rows, mean ± std over seeds):
  - **Columns:** Hits@1, Hits@10, MRR, P, R, F1, #Pred, #Correct, #Gold.
  - **Layout:** one table per gold source, plus one compact table of MRR / F1 per row × source.
- [x] **Log** `dev_docs/execution_logs/sprint_10_4.md`, with the run ids and these disclosures:
  - **Gold reads:** four reads of `test`, one per source, each a full gold set.
  - **Fixed fusion weight:** w was fixed from NCIT–DOID, not tuned here.
  - **Recall ceiling:** one-to-one assignment gives each source at most one prediction. With 1:n gold, recall is capped at `#test techniques / #test pairs`; report that ceiling next to R.
  - **Gold-set sizes:** the full sets after id resolution are about 221 (`intersection`), 287 (`stix`), 566 (`capec`) and 628 (`union`) pairs. Report the sizes; sources are separate gold sets, so their scores are not pooled.
  - **Pseudo-pairs:** they come from SecureBERT2.

**Acceptance:** gate 10.4; all three rows × four gold sources filled. **Decision:** report only.

---

## Phase 10.5: SecRoBERTa as the text encoder

**Capability:** repeat 10.2 (embedding), 10.3 (training) and 10.4 (gold evaluation) with SecRoBERTa (`jackaduma/SecRoBERTa`) as the only text embedding, to see whether the 10.4 result depends on the encoder. Nothing else changes: same four ingests (`attack_cwe_ent`, runs of 10.2), same gold sources, same protocol (`protocol: unsupervised`, `held_out_policy: none`), same fixed settings, no tuning. SecureBERT2 results stay as reported in 10.4; they are not recomputed.

| Row | Model | Method |
|---|---|---|
| 1 | Frozen SecRoBERTa | `embedding_store` |
| 2 | GCN(SecRoBERTa) | `gcl_run` (this phase), same module as 10.3 |
| 3 | SecRoBERTa + GCN(SecRoBERTa) | `fusion`: SecRoBERTa w 1, row 2 w 0.25 |

- [x] **Embed** with the `ingest` lineage `attack_cwe_ent` (`LATEST` = union; node tables are identical across sources):
  - **Config:** `hf_mean_pool`, `model_ref: jackaduma/SecRoBERTa`, label `pt_SecRoBERTa` (embedder id `hf_mean_pool-pt_SecRoBERTa`, the id of the earlier ICS store), text variant `description`, the `embed.yaml` defaults (`max_length 512`, mean pooling of the last hidden state, `batch_size 8`), as for SecureBERT2 in 10.2.
  - **Check:** log rows embedded, rows with empty text, and the gold endpoints missing from the store (per source), as in the 10.2 coverage table.
- [x] **Train** 1 row × 3 seeds = 3 runs, plus 3 `training_data` runs, exactly as 10.3 with the text store swapped:
  - **Pseudo-pairs:** SecRoBERTa text-cosine top-k, cross-graph (k 5, `rank_topk`, `held_out_policy: none`), sampler id `srb-rank_topk-k5`; the DeepWalk store is still required by `training_data` for its graph columns and is not an input feature.
  - **Model:** `residual_adapter`, `adapter: gcn`, `residual_floor: false`, InfoNCE, temperature 0.1, Adam 1e-3, 100 epochs, last epoch, seeds 42/43/44; ids `ent-g-srb-s<seed>`.
  - **Script:** `scripts/run_unsup_matrix.sh` gains priority `8` (the 10.3 commands with the sampler row id `g-srb`), run with `TEXT_EMBEDDER=hf_mean_pool-pt_SecRoBERTa PREFIX=ent- SAMPLER_PREFIX=srb-`.
  - **Universe:** `intersection` of the SecRoBERTa and DeepWalk stores; log its size and the gold pairs it loses per source (as in 10.3).
- [x] **Evaluate** with `configs/experiments/attack_cwe_ent_srb_test.yaml` (rows 1–3, rows 2 and 3 per seed; method ids `ent-s-r1`, `ent-s-r2-s<seed>`, `ent-s-r3-s<seed>`), once per gold source with `--set evaluate.inputs.ingest_run=<that source's ingest run>`: four runs, same prediction protocol as 10.4.
- [x] **Tables** via `scripts/unsup_summary.py` with a new `--rows attack_ent_srb` row set, plus a comparison table of MRR and F1 per row and source, SecRoBERTa against SecureBERT2 (the 10.4 runs), with the difference of the means.
- [x] **Log** `dev_docs/execution_logs/sprint_10_5.md`, with the run ids and these disclosures:
  - **Gold reads:** four more reads of `test`, one per source (eight in total with 10.4). The two encoders are reported side by side; neither is selected using gold.
  - **Fixed settings:** fusion weight 0.25 and all training settings come from earlier sprints and are not tuned for SecRoBERTa.
  - **Recall ceiling, gold-set sizes, common query set, pseudo-pair provenance:** as in 10.4, with pseudo-pairs now from SecRoBERTa.
  - **Encoder differences:** SecRoBERTa and SecureBERT2 differ in pre-training data and in how many nodes have text vectors; report the store sizes.

**Acceptance:** gate 10.4 holds for the SecRoBERTa runs (frozen row identical across a repeated run, fusion reproduces its member cosines); all three rows × four gold sources filled. **Decision:** report only.

---

## Phase 10.6: Shared candidate pool (union gold CWEs) and random baseline

**Why:** in 10.4 and 10.5 every technique is ranked against all 969 CWEs, but each gold set links only 68–151 of them, so a random ranking already scores MRR ≈ 0.007 and the table gives no sense of scale. This phase adds a second, clearly secondary view that ranks against a smaller pool, plus a random-ranking baseline. The main tables (all 969 CWEs, no gold used to choose candidates) stay as reported.

**The pool:** the 151 distinct CWE nodes that appear in the `union` gold (the superset of the four sources). One pool is shared by all four sources, so their numbers stay on the same footing. It is built from gold, so the pool view is **optimistic by construction**; it is read next to the baseline run on the same pool, never as an absolute score. Gold is still never read by training or by the main tables; the pool is read only by `evaluate`.

- [x] **Pool file** `scripts/make_target_pool.py`: reads `alignments.parquet` of the `union` ingest and writes the sorted CWE node ids (JSON, with the source run id and sha256 of the list) to `data/processed/attack_cwe_ent/target_pool_union.json`. The script checks, for the other three ingests, that every gold target is inside the pool and fails otherwise.
- [x] **Evaluator option `target_pool`** (default off) in `gold_alignment` and `gold_alignment_prf`: a path to the pool file; the target-side ranking and top-3 candidates use only pool nodes. Everything else is unchanged: queries are the `test` techniques, `sibling_filter_splits: [test]`, greedy one-to-one, threshold 0.0, `biogitom_filters: false`. The option and the pool file hash go into the evaluation provenance. A gold target outside the pool raises `ConfigError`. Unit tests: with the option off, results equal the current ones; with it on, a hand-computed 1:n fixture gives the expected ranks and P/R/F1, and a target outside the pool raises.
- [x] **Random baseline** `scripts/make_random_store.py`: writes a seeded Gaussian-vector store (768 d, seeds 42/43/44) for the nodes of the SecureBERT2 store, in the usual embedding-store format with embedder ids `random-rand42/43/44`, so it runs through the existing `embedding_store` method with no new method kind. Same nodes in every store, so every row has the same ranking queries.
- [x] **Config** `configs/experiments/attack_cwe_ent_pool_test.yaml`:
  - **Methods:** the 10.4 rows (frozen SecureBERT2, GCN(SecureBERT2) per seed, fusion w 0.25 per seed), the 10.5 rows (the same with SecRoBERTa), and the random baseline per seed (seven row groups). No retraining; models are those of 10.3 and 10.5.
  - **Evaluations:** four. `gold_full` and `prf_full` are the 10.4 evaluations (all CWEs); `gold_pool` and `prf_pool` are the same two with `target_pool` set. Both views are computed in the same evaluate run.
  - **Running it:** `evaluate` once per gold source with `--set evaluate.inputs.ingest_run=<that source's ingest run>`: four runs.
- [x] **Reproduction check:** the `*_full` numbers for the SecureBERT2 and SecRoBERTa rows equal the 10.4 and 10.5 tables exactly (max absolute difference 0.0), and the random baseline is identical on a repeated run.
- [x] **Tables** via `scripts/unsup_summary.py` (`--rows attack_ent_pool`, evaluation selection with the existing `--ranking-eval` / `--prf-eval`), mean ± std over seeds:
  - **Columns:** Hits@1, Hits@10, MRR, P, R, F1, #Pred, #Correct, #Gold, as in 10.4.
  - **Layout:** one pool-view table per gold source with the random row; one compact MRR / F1 table (row × source) for the pool view; and per row and source the pool-view MRR divided by the random-baseline MRR on the same pool.
- [x] **Log** `dev_docs/execution_logs/sprint_10_6.md`, with the run ids and these disclosures:
  - **Optimistic pool:** the pool is derived from `union` gold, so pool-view scores are an upper-bound style view; only the lift over the random baseline on the same pool is interpreted.
  - **Gold reads:** four more reads of `test`, one per source (twelve in total with 10.4 and 10.5); both views are read in each run.
  - **Training untouched:** no node or edge was filtered; models are the ones from 10.3 and 10.5.
  - **Pool size and coverage:** 151 CWEs; the number of pool CWEs outside each source's gold (so it is clear how much a smaller source is helped), and the pool CWEs missing from each store.
  - **Recall ceiling, gold-set sizes, common query set:** unchanged from 10.4.
  - **Fixed settings:** fusion weight 0.25 is not re-tuned for the pool.

**Acceptance:** gate 10.6; both encoders and the random baseline × four gold sources × both views filled. **Decision:** report only.

---

## Out of scope

Hyperparameter sweeps, trained residual variants (R1–R3), encoders other than SecureBERT2 (10.2–10.4) and SecRoBERTa (10.5), ICS ATT&CK, CAPEC as a third graph, and supervised training on the `train` split.
