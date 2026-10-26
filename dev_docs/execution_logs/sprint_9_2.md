# Sprint 9, Phase 9.2: Ablation matrix and paper table

Plan: [sprint_9.md](../sprints/sprint_9.md). Previous phase: [9.1](sprint_9_1.md). Decision: report only. 535 unit tests pass.

## 0. Outcome in brief

- **All 21 NCIT–DOID runs (priorities 1, 2 and 3 × seeds 42 / 43 / 44) were trained with the settings fixed in advance, and the gold test pairs were read once for the matrix** (run `20261026T043657Z-b3585c20`). No tuning run was made and no priority level was dropped.
- **No unsupervised variant beats frozen BERT (U0) on NCIT–DOID.** U0: MRR 0.0186, F1 0.0204. The best trained row is U1-I (BERT + projection, InfoNCE): MRR 0.0177, F1 0.0175. Every GCN row (U2 family) is far below U0 (MRR 0.0035 to 0.0093, F1 at most 0.0039).
- **Graph structure did not help gold-pair retrieval here.** U2 (triplet) MRR 0.0040 against U1 0.0160 (the planned U1-vs-U2 question). On the gold-free ATT&CK–CWE readout the opposite holds for structural agreement: U2 raises 1-hop structural hit@10 from 0.17 / 0.25 (U0) to 0.64 / 0.82 (graph_1 / graph_2), while textual agreement with BERT drops (graph_2 0.62 against 1.00).
- **InfoNCE beats triplet in both families** on every ranking metric (U1-I vs U1: MRR 0.0177 vs 0.0160; U2-I vs U2: 0.0093 vs 0.0040). For P/R/F1 the order differs for U1: triplet has the higher F1 (0.0186 vs 0.0175).
- **Absolute numbers are very low** (frozen BERT places the gold target first for 1.2% of the 3,280 sources). The matrix measures whether pseudo-pair training moves this floor; in this run it does not improve it.

## 1. What was run

Shared settings (unchanged from the plan): text = frozen `bert_base`; structure = DeepWalk; `rank_topk` k = 5; Adam 1e-3; 100 epochs, last epoch exported; triplet margin 1.0, InfoNCE temperature 0.1; seeds 42 / 43 / 44. `--set seed=S` seeds both `training_data` and `gcl_train`; one `training_data` run per seed and sampler variant.

| ID | Prio | Model id | Experiment config | Overrides |
|---|---|---|---|---|
| U0 | 1 | – | frozen `hf_mean_pool-bert_base` | – |
| U1 | 1 | `u1-s<seed>` | [unsup_u1.yaml](../../configs/experiments/unsup_u1.yaml) | – |
| U2 | 1 | `u2-s<seed>` | [unsup_u2.yaml](../../configs/experiments/unsup_u2.yaml) | – |
| U1-I | 2 | `u1-i-s<seed>` | unsup_u1 | `gcl_train.loss.name=infonce` |
| U2-I | 2 | `u2-i-s<seed>` | unsup_u2 | `gcl_train.loss.name=infonce` |
| U2-rand | 3 | `u2-rand-s<seed>` | unsup_u2 | `training_data.sampler.negatives=random` |
| U2-hard | 3 | `u2-hard-s<seed>` | unsup_u2 | `training_data.sampler.negatives=hard` (offset 10) |
| U2-mut | 3 | `u2-mut-s<seed>` | unsup_u2 | `training_data.sampler.positives=mutual` |

All rows are driven by [run_unsup_matrix.sh](../../scripts/run_unsup_matrix.sh) (about 2 h 20 min for the NCIT–DOID matrix on one GPU). ATT&CK–CWE trained U1 and U2 for the same three seeds.

The 20-epoch smoke models of Phase 9.1 used the same model ids (`u1-s42`, `u2-s42`); the 100-epoch runs replaced them as `LATEST`, so the 9.1 smoke configs now resolve to the matrix models.

Evaluation (once each):

| dataset | config | run |
|---|---|---|
| NCIT–DOID (3,280 test pairs) | [ncit_doid_unsup_test.yaml](../../configs/experiments/ncit_doid_unsup_test.yaml): `gold_alignment` (global, sibling filter `[test]`) + `gold_alignment_prf` (top-3, greedy one-to-one, threshold 0, no BioGITOM filters) | `20261026T043657Z-b3585c20` |
| ATT&CK–CWE | [attack_cwe_unsup_eval.yaml](../../configs/experiments/attack_cwe_unsup_eval.yaml): `neighborhood` only | `20261026T044750Z-e4abd55c` |

`summary.md` / `summary.csv` from [unsup_summary.py](../../scripts/unsup_summary.py) are in each run directory.

## 2. Paper table

Table X. Unsupervised ablation on NCIT–DOID (3,280 held-out gold test pairs; no gold pair used in training; all hyperparameters fixed in advance). Mean ± sample std over 3 seeds (U0 is deterministic). P/R/F1: greedy one-to-one over top-3 cosine candidates, threshold 0. #Gold = 3,280 for every row.

| Variant | Pseudo-pairs | Loss | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT (frozen) | – | – | 0.0116 | 0.0308 | 0.0186 | 0.0712 | 0.0119 | 0.0204 | 548 | 39 |
| BERT + projection | top-k | Triplet | 0.0100 ± 0.0004 | 0.0255 ± 0.0008 | 0.0160 ± 0.0005 | 0.0676 ± 0.0035 | 0.0108 ± 0.0006 | 0.0186 ± 0.0011 | 522.7 ± 4.7 | 35.3 ± 2.1 |
| BERT + projection | top-k | InfoNCE | 0.0107 ± 0.0000 | 0.0296 ± 0.0006 | 0.0177 ± 0.0001 | 0.0601 ± 0.0012 | 0.0103 ± 0.0002 | 0.0175 ± 0.0003 | 560.7 ± 7.6 | 33.7 ± 0.6 |
| BERT + GCN (text + structure) | top-k | Triplet | 0.0010 ± 0.0007 | 0.0059 ± 0.0007 | 0.0040 ± 0.0007 | 0.0004 ± 0.0007 | 0.0001 ± 0.0002 | 0.0002 ± 0.0003 | 797.0 ± 11.3 | 0.3 ± 0.6 |
| BERT + GCN (text + structure) | top-k | InfoNCE | 0.0033 ± 0.0005 | 0.0189 ± 0.0006 | 0.0093 ± 0.0003 | 0.0081 ± 0.0028 | 0.0025 ± 0.0009 | 0.0039 ± 0.0013 | 1034.0 ± 23.3 | 8.3 ± 2.9 |
| BERT + GCN | random neg. | Triplet | 0.0008 ± 0.0005 | 0.0049 ± 0.0006 | 0.0035 ± 0.0001 | 0.0018 ± 0.0015 | 0.0002 ± 0.0002 | 0.0004 ± 0.0003 | 378.3 ± 5.5 | 0.7 ± 0.6 |
| BERT + GCN | hard neg. | Triplet | 0.0007 ± 0.0002 | 0.0119 ± 0.0029 | 0.0062 ± 0.0006 | 0.0030 ± 0.0025 | 0.0006 ± 0.0005 | 0.0010 ± 0.0009 | 671.0 ± 5.2 | 2.0 ± 1.7 |
| BERT + GCN | mutual top-k | Triplet | 0.0021 ± 0.0005 | 0.0152 ± 0.0029 | 0.0084 ± 0.0012 | 0.0015 ± 0.0010 | 0.0007 ± 0.0005 | 0.0010 ± 0.0006 | 1557.0 ± 13.1 | 2.3 ± 1.5 |

Reading it (matrix questions from the plan):

| question | answer from this run |
|---|---|
| Does training on pseudo-pairs improve over frozen BERT (U1 vs U0)? | No: U1 MRR 0.0160 and U1-I 0.0177 against 0.0186. |
| Does graph structure add to U1 (U2 vs U1)? | No for gold retrieval: U2 0.0040 against U1 0.0160. |
| Triplet vs InfoNCE | InfoNCE is ahead on Hits and MRR with and without the graph; U1 triplet has the higher F1. |
| Easy vs random negatives (U2-rand vs U2) | Same level (MRR 0.0035 vs 0.0040); random predicts fewer pairs (378 vs 797). |
| Hard negatives (U2-hard vs U2) | Higher Hits@10 / MRR (0.0119 / 0.0062 vs 0.0059 / 0.0040), still far below U0. |
| Cleaner positives (U2-mut vs U2) | Higher Hits@10 / MRR (0.0152 / 0.0084), but 1,557 predicted pairs and only 2.3 correct. |

The U2-family differences are small in absolute terms (all below 0.01 MRR) and the stds are comparable to several of the gaps, so only "the GCN rows are far below U0 and U1" and "InfoNCE > triplet on MRR" are clear. U1's seed std is tiny (Hits@1 std 0.0000 for U1-I): the residual floor keeps the output close to the BERT vector.

**Possible reasons (not tested here).** The pseudo-pairs come from the same frozen BERT similarity that U0 is scored with, so there is no new gold-aligned signal; and U2 replaces the BERT vector by a 128-d learned projection plus a DeepWalk tower, with no residual floor, so any information it loses cannot be recovered. A residual / no-floor ablation of U2 would separate these, but it is not in this matrix.

## 3. ATT&CK–CWE (gold-free, 3 seeds for U1 / U2)

Run `20261026T044750Z-e4abd55c`, `neighborhood`, hit@10 (mean ± std; all gold columns are `—`):

| view | role | U0 | U1 | U2 |
|---|---|---|---|---|
| structural r1 | graph_1 | 0.1675 | 0.1494 ± 0.0057 | 0.6420 ± 0.0124 |
| structural r1 | graph_2 | 0.2497 | 0.2465 ± 0.0000 | 0.8221 ± 0.0102 |
| structural r2 | graph_1 | 0.9458 | 0.9228 ± 0.0028 | 0.9869 ± 0.0057 |
| structural r2 | graph_2 | 0.6388 | 0.6288 ± 0.0012 | 0.9968 ± 0.0000 |
| textual (BERT top-10) | graph_1 | 1.0000 | 0.9885 ± 0.0028 | 0.9212 ± 0.0085 |
| textual (BERT top-10) | graph_2 | 1.0000 | 0.9986 ± 0.0006 | 0.6152 ± 0.0021 |

These measure agreement with the graph's own neighbourhood and with BERT, not alignment quality, so they say that U2 encodes structure and no more.

## 4. Implementation change needed to train the InfoNCE rows

The 9.1 `infonce` loss builds an `N × N` logit matrix over all examples; the training loop is full-batch and `N` = 121,105, so the first U1-I attempt needed 54.6 GiB on a 16 GiB GPU. The loss in [losses.py](../../src/cga/pipelines/gcl_train/losses.py) now gives the same value and gradient with two exact rewrites (no change to the objective, the data or the settings):

- identical negative rows are collapsed to distinct ones and each logit is shifted by `log(count)` (the text negatives have 1,041 distinct rows of 121,105 and the graph negatives 11,036, because bottom-k negatives are shared across anchors);
- rows are processed in chunks of 2,048 under activation checkpointing.

Tests: the chunked value and gradient equal the unchunked ones, and a batch with repeated negatives equals a naive `N × N` reference. The earlier matrix attempts (stopped after the out-of-memory failure and after a chunked-only version proved too slow, about 1 h per 10 epochs of one InfoNCE row) produced no results that were used; all 21 final runs use the final loss. The older `infonce` unit tests pass unchanged.

## 5. Disclosures and limitations

- **Test-read disclosure.** The gold test pairs were read (a) for the three Phase 9.1 smoke evaluations (seed 42, 20 epochs; models since replaced) and (b) once for the final matrix above. No hyperparameter, epoch count, threshold or row was chosen from either read; the settings were fixed before the first run and nothing was re-run after seeing the results. The gold train pairs were never read by training, by the evaluator's sibling filter (`[test]`) or by the P/R/F1 pair set (`biogitom_filters: false`).
- Seeds: GCN kernels are not bit-reproducible on GPU; U0 is deterministic and reproduces the 9.1 value.
- U0 and the trained rows score the same 3,280 queries (no gold endpoint is among the 6 nodes without a DeepWalk vector).
- ATT&CK–CWE has no gold set, so no ATT&CK–CWE row is gold-scored; U1-I and the priority-2/3 rows were trained only on NCIT–DOID, as in the plan.
- One matrix, seed std from 3 seeds, no significance test.

## 6. Commands

```bash
uv run pytest tests/unit -q
bash scripts/run_unsup_matrix.sh ncit_doid_unsup 1 2 3      # 21 runs (priority 1, 2, 3)
bash scripts/run_unsup_matrix.sh attack_cwe_unsup 1         # U1 / U2 on ATT&CK-CWE
uv run cga run evaluate --dataset ncit_doid_unsup  --experiment ncit_doid_unsup_test
uv run cga run evaluate --dataset attack_cwe_unsup --experiment attack_cwe_unsup_eval
uv run python scripts/unsup_summary.py --dataset ncit_doid_unsup  --run 20261026T043657Z-b3585c20
uv run python scripts/unsup_summary.py --dataset attack_cwe_unsup --run 20261026T044750Z-e4abd55c
```

## 7. Training runs (`gcl_train`, `models/gcl/<dataset>/<model id>/<run>`; date 20261026)

NCIT–DOID, seeds 42 / 43 / 44:

| row | runs |
|---|---|
| U1 `u1` | T033436Z-5c1a29eb / T035624Z-00dca471 / T041551Z-8139b5b4 |
| U2 `u2` | T033704Z-05b36788 / T035912Z-cbc32f9e / T041833Z-d8436c58 |
| U1-I `u1-i` | T033741Z-78cdf78f / T035938Z-85829c98 / T041910Z-3436ef5a |
| U2-I `u2-i` | T034646Z-835bbe73 / T040821Z-b39e5cf5 / T042751Z-9e51cf31 |
| U2-rand `u2-rand` | T035019Z-2c01c6d1 / T041145Z-c367f0aa / T043152Z-29a38329 |
| U2-hard `u2-hard` | T035202Z-e4ed5535 / T041317Z-8f1d154b / T043406Z-7e6f9b37 |
| U2-mut `u2-mut` | T035435Z-85943051 / T041450Z-af2eed04 / T043624Z-744e7339 |

ATT&CK–CWE: `u1` T034258Z-486d53a4 / T034343Z-3733e6bf / T034439Z-2b78b943; `u2` T034319Z-9f3de7e7 / T034407Z-d967d093 / T034502Z-207468c4.

## 8. Files

New: `scripts/run_unsup_matrix.sh`, `configs/experiments/{ncit_doid_unsup_test,attack_cwe_unsup_eval}.yaml`.
Modified: `src/cga/pipelines/gcl_train/losses.py` (exact memory-bounded `infonce`), `scripts/unsup_summary.py` (paper table), `tests/unit/pipelines/gcl_train/test_losses.py`, `tests/unit/test_unsup_summary.py`.
