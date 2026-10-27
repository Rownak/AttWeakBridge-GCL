# Sprint 10.6: Shared candidate pool (union gold CWEs) and random baseline

Plan: [sprint_10.md](../sprints/sprint_10.md). Previous: [10.5](sprint_10_5.md). Decision: report only.

## 0. Outcome in brief

- **The 10.4 and 10.5 numbers reproduce exactly** (168 metric rows per source, max absolute difference 0.0), so the new view is added on top of the same evaluation.
- **Random ranking:** MRR 0.0065–0.0070 over all candidate CWEs and 0.0373–0.0390 over the 151-CWE pool. Its F1 is 0.000 on all CWEs and 0.005–0.008 on the pool.
- **Lift over random is large on all CWEs and small on the pool:**

  | Frozen encoder | MRR over random, all CWEs | MRR over random, pool |
  |---|---|---|
  | SecureBERT2 | 2.4–2.8x | 1.4–1.6x |
  | SecRoBERTa | 3.3–4.4x | 1.8–2.1x |

  Much of the full-pool signal is "gold-linked CWEs look more like attack text than the unrelated CWEs"; inside the pool, where every candidate is CAPEC-linked to some technique, the encoders separate the right CWE only weakly.
- **GCN(either encoder) is at or near random on the pool** (0.9–1.3x). Fusion tracks frozen (1.5–1.9x).
- **SecRoBERTa frozen has the one clearly non-random F1 on the pool:** F1 0.046–0.061 against random 0.005–0.008, with 8–16 correct pairs (P 0.17–0.25). SecureBERT2 frozen (F1 0.006–0.010, 1–3 correct) is not separable from random on F1.

## 1. What was done

- **Pool** (`scripts/make_target_pool.py`): the 151 distinct CWE nodes of the `union` gold, written to `data/processed/attack_cwe_ent/target_pool_union.json` (sha256 `d66518c5ff42…`, source ingest `20261026T224948Z-0f7e98dc`). The script also checked the other three ingests: every gold target of `stix`, `capec` and `intersection` is inside the pool.
- **Evaluator option `target_pool`** (default off; `src/cga/pipelines/evaluate/evaluators/target_pool.py`, used by `gold_alignment` and `gold_alignment_prf`): ranking and top-3 candidates use pool nodes only. It checks the file hash, raises `ConfigError` on a missing or tampered file or a gold target outside the pool, and records `target_pool_sha256` and `n_target_pool` in the evaluation provenance. With the option off, results are unchanged.
- **Random baseline** (`scripts/make_random_store.py`): standard-normal vectors, 768 d, for the 5,131 nodes of the SecureBERT2 store, seeds 42/43/44 (`random-rand42/43/44`, run ids `rand-s42-ddcde579`, `rand-s43-94f8763a`, `rand-s44-0c4c3b21`). They run through `embedding_store`; no new method kind.
- **Config** `configs/experiments/attack_cwe_ent_pool_test.yaml`: 17 methods (SecureBERT2 and SecRoBERTa frozen, GCN and fusion per seed, random per seed) and four evaluations: `gold_full` / `prf_full` (all CWEs, as in 10.4 and 10.5) and `gold_pool` / `prf_pool`. No retraining: models are `ent-g-sb2-s*` (10.3) and `ent-g-srb-s*` (10.5).
- **Tables** (`scripts/unsup_summary.py`): `--rows attack_ent_pool`, `--ranking-eval` / `--prf-eval` selecting the view, `--lift` (MRR over the random baseline of the same view), `--out-name`.

## 2. Runs

| Gold source | Ingest run | Evaluate run |
|---|---|---|
| stix | `20261026T224906Z-a0725a37` | `20261027T021941Z-01c5d7f6` |
| capec | `20261026T224923Z-1b4b5b3d` | `20261027T021953Z-fddc75ee` |
| intersection | `20261026T224935Z-dee50c06` | `20261027T022008Z-ec24ba1c` |
| union | `20261026T224948Z-0f7e98dc` | `20261027T022021Z-8c74b3b3` |

Repeat of `union` for determinism: `20261027T022149Z-8c74b3b3`. Reports: `artifacts/evaluations/attack_cwe_ent/20261027T022021Z-8c74b3b3/sources_summary_pool.md` and `sources_summary_full.md`.

## 3. Pool and candidate sets

| | CWE candidates |
|---|---|
| CWE nodes in the graph | 969 |
| Ranked in the full view (`gold_full`, CWEs in the 2,927-node common set) | **944** |
| Shared pool (`gold_pool`), all in the common set | **151** |

The full-view candidate set is 944 CWEs, not 969: ranking uses the nodes every method has, and the GCN universe is the 2,927-node intersection of 10.3 (the 10.4 and 10.5 logs say "969"; the measured random MRR 0.0065–0.0070 is what the metric shows, so no result changes). Every pool CWE has a vector in the SecureBERT2, SecRoBERTa and DeepWalk stores.

Pool CWEs outside each source's own gold (how much extra ground a smaller source has to rank over): `stix` 74, `capec` 2, `intersection` 83, `union` 0. So for `stix` and `intersection` the pool is much larger than their own targets (77 and 68 CWEs).

Ranking queries are the common set of 10.4: 284 (`stix`), 566 (`capec`), 221 (`intersection`), 625 (`union`), identical in both views. `#Gold` is 287 / 566 / 221 / 628.

## 4. Results

Mean ± sample std over 3 seeds (frozen rows have 1 seed). Full per-source tables with Hits@1/10, P, R, #Pred, #Correct, `R ceiling` are in the two report files; below is the MRR / F1 view.

### Pool view (151 CWEs): MRR / F1

| Row | stix | capec | intersection | union |
|---|---|---|---|---|
| SecureBERT2 (frozen) | 0.0533 / 0.0061 | 0.0613 / 0.0097 | 0.0622 / 0.0078 | 0.0575 / 0.0088 |
| GCN(SecureBERT2) | 0.0372 / 0.0112 | 0.0367 / 0.0110 | 0.0355 / 0.0071 | 0.0376 / 0.0109 |
| SecureBERT2 + 0.25 × GCN | 0.0574 / 0.0102 | 0.0621 / 0.0086 | 0.0675 / 0.0128 | 0.0583 / 0.0078 |
| SecRoBERTa (frozen) | 0.0705 / 0.0531 | 0.0790 / 0.0509 | 0.0805 / 0.0606 | 0.0734 / 0.0462 |
| GCN(SecRoBERTa) | 0.0359 / 0.0092 | 0.0479 / 0.0131 | 0.0398 / 0.0116 | 0.0453 / 0.0129 |
| SecRoBERTa + 0.25 × GCN | 0.0584 / 0.0377 | 0.0724 / 0.0443 | 0.0647 / 0.0402 | 0.0676 / 0.0404 |
| Random ranking | 0.0389 / 0.0053 | 0.0375 / 0.0076 | 0.0390 / 0.0068 | 0.0378 / 0.0070 |

### MRR and lift over the random baseline of the same view

| Row | View | stix | capec | intersection | union |
|---|---|---|---|---|---|
| SecureBERT2 (frozen) | all | 2.4x | 2.8x | 2.7x | 2.6x |
| | pool | 1.4x | 1.6x | 1.6x | 1.5x |
| GCN(SecureBERT2) | all | 1.4x | 1.2x | 1.2x | 1.2x |
| | pool | 1.0x | 1.0x | 0.9x | 1.0x |
| SecureBERT2 + GCN | all | 2.3x | 3.0x | 2.7x | 2.8x |
| | pool | 1.5x | 1.7x | 1.7x | 1.5x |
| SecRoBERTa (frozen) | all | 3.9x | 3.6x | 4.4x | 3.3x |
| | pool | 1.8x | 2.1x | 2.1x | 1.9x |
| GCN(SecRoBERTa) | all | 1.7x | 2.1x | 1.9x | 2.0x |
| | pool | 0.9x | 1.3x | 1.0x | 1.2x |
| SecRoBERTa + GCN | all | 3.1x | 3.4x | 3.4x | 3.2x |
| | pool | 1.5x | 1.9x | 1.7x | 1.8x |

(Random MRR, all candidates: 0.0065 / 0.0070 / 0.0065 / 0.0070; pool: 0.0389 / 0.0375 / 0.0390 / 0.0378.)

## 5. Reading

- A pool of 151 raises everything, random included (MRR 0.007 → 0.038), so the raw pool numbers (SecRoBERTa frozen 0.071–0.081) must not be read as better models. Only the lift is informative, and it is smaller on the pool: the encoders mostly tell attack-related CWEs from unrelated ones, and are weak at picking the technique's own CWEs among CAPEC-linked ones.
- SecRoBERTa stays ahead of SecureBERT2 on the pool (lift 1.8–2.1x against 1.4–1.6x) and on F1, the same direction as 10.5.
- The trained GCN sits at random on the pool, below frozen on the full view, again as in 10.4 and 10.5.
- Pool-view F1 is the first place F1 is not just a few pairs: SecRoBERTa frozen has 8–16 correct pairs and about 7–10x random F1 (0.046–0.061 against 0.005–0.008). SecureBERT2 and the GCN rows have 1–5 correct pairs, within noise of random.
- Random ranking has F1 0.000 on all CWEs because it never hits an exact one-to-one pair among 944 candidates.

## 6. Disclosures

- **Optimistic pool:** the pool comes from `union` gold, so pool scores are an upper-bound style view; only the lift over the random baseline on the same pool is interpreted. Gold is still read only by `evaluate`; training and the main (full) tables never use the pool.
- **Gold reads:** four more reads of `test`, one per source, each covering both views (twelve source reads in total with 10.4 and 10.5), plus one repeat of `union`.
- **Training untouched:** no node or edge was filtered; models are those of 10.3 and 10.5.
- **Pool coverage:** see section 3 (74 / 2 / 83 / 0 pool CWEs outside the own gold for `stix` / `capec` / `intersection` / `union`).
- **Recall ceiling, gold-set sizes, common query set:** unchanged from 10.4.
- **Fixed settings:** fusion weight 0.25 not re-tuned for the pool.
- **Random baseline:** 3 seeds, 768-d Gaussian vectors over the SecureBERT2 node set; its F1 values are tiny, so its std is dominated by one or two pairs.

## 7. Checks (gate 10.6)

- **Pool:** 151 CWEs, all gold targets of the four sources inside it (checked by the script and by the evaluator at run time).
- **Reproduction:** with no pool, the `*_full` rows of the new runs equal the 10.4 and 10.5 runs exactly (168 rows matched per source, max abs diff 0.0, `n_queries` identical).
- **Random baseline stable:** full-view MRR 0.0065–0.0070 and pool MRR 0.0373–0.0390 across seeds (union: 0.0072/0.0070/0.0068 and 0.0387/0.0374/0.0373).
- **Determinism:** the repeated `union` evaluation is identical in all 408 metric rows.
- **Tests:** `test_target_pool.py` (7), `test_unsup_summary.py` (+1); 574 unit tests pass.

## 8. Files

New: `scripts/make_target_pool.py`, `scripts/make_random_store.py`, `src/cga/pipelines/evaluate/evaluators/target_pool.py`, `configs/experiments/attack_cwe_ent_pool_test.yaml`, `tests/unit/pipelines/evaluate/test_target_pool.py`. Changed: `src/cga/pipelines/evaluate/evaluators/gold_alignment.py` and `gold_alignment_prf.py` (`target_pool`), `scripts/unsup_summary.py` (`attack_ent_pool` rows, `--lift`, `--out-name`, evaluation selection in `build_source_report`), `tests/unit/test_unsup_summary.py`, `dev_docs/sprints/sprint_10.md`. Data: `data/processed/attack_cwe_ent/target_pool_union.json`, `artifacts/embeddings/attack_cwe_ent/random-rand{42,43,44}/`.

## 9. Commands

```
I=data/processed/attack_cwe_ent
uv run python scripts/make_target_pool.py --ingest $I/20261026T224948Z-0f7e98dc \
  --check $I/20261026T224906Z-a0725a37 --check $I/20261026T224923Z-1b4b5b3d --check $I/20261026T224935Z-dee50c06 \
  --out $I/target_pool_union.json
for s in 42 43 44; do uv run python scripts/make_random_store.py --dataset attack_cwe_ent \
  --reference artifacts/embeddings/attack_cwe_ent/hf_mean_pool-SecureBERT2/20261026T225000Z-4f1f6686 --seed $s --label rand$s; done
for pair in stix=20261026T224906Z-a0725a37 capec=20261026T224923Z-1b4b5b3d \
            intersection=20261026T224935Z-dee50c06 union=20261026T224948Z-0f7e98dc; do
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_pool_test \
    --set evaluate.inputs.ingest_run=${pair#*=}
done
SR="--source-run stix=20261027T021941Z-01c5d7f6 --source-run capec=20261027T021953Z-fddc75ee --source-run intersection=20261027T022008Z-ec24ba1c --source-run union=20261027T022021Z-8c74b3b3"
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run 20261027T022021Z-8c74b3b3 --rows attack_ent_pool \
  --ranking-eval gold_pool --prf-eval prf_pool --lift --out-name sources_summary_pool.md $SR
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run 20261027T022021Z-8c74b3b3 --rows attack_ent_pool \
  --ranking-eval gold_full --prf-eval prf_full --lift --out-name sources_summary_full.md $SR
uv run pytest tests/unit -q
```
