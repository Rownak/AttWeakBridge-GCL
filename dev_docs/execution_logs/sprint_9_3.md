# Sprint 9, Phase 9.3: Paper table with SapBERT

Plan: [sprint_9.md](../sprints/sprint_9.md). Previous phases: [9.1](sprint_9_1.md), [9.2](sprint_9_2.md) (the BERT table this phase repeats). Decision: report only. 537 unit tests pass.

## 0. Outcome in brief

- **Gate 9.3: GO.** U0-SapBERT reproduces both references exactly (section 3).
- **The 9.2 conclusion holds with a much stronger text encoder: no unsupervised variant beats frozen SapBERT on MRR or F1.** U0 (frozen SapBERT): Hits@1 0.3424, MRR 0.4475, F1 0.4136. The best trained row, "SapBERT + projection, triplet", reaches MRR 0.4276 and F1 0.3978 (−0.020 and −0.016). Training on pseudo-pairs made the frozen encoder slightly worse, not better.
- **One partial exception:** SapBERT + GCN with InfoNCE has the best Hits@10 of all rows (0.6973 ± 0.0083 against 0.6537 for U0), while its Hits@1 (0.2738) and MRR (0.4088) are below U0.
- **Graph structure again hurts gold retrieval:** projection (U1) vs GCN (U2) with triplet, MRR 0.4276 vs 0.1899. With InfoNCE the gap is small (0.4129 vs 0.4088).
- **InfoNCE vs triplet flips with the graph:** without the graph triplet is ahead (MRR 0.4276 vs 0.4129); with the GCN InfoNCE is far ahead (0.4088 vs 0.1899). In 9.2 (BERT) InfoNCE was ahead in both families.
- **Sampler variants (all U2, triplet):** random 0.2318, hard 0.1991, mutual 0.2759 MRR against 0.1899 for the top-k default; all three beat the default and none gets near U1 or U0.
- **Encoder effect:** the same rows are 0.19 to 0.43 MRR higher with SapBERT than with `bert_base` (section 2). Frozen SapBERT already carries almost all the quality; the matrix changes it by a few points at most (projection rows) or loses a lot (GCN rows).

## 1. What was implemented and run

- **Store:** `artifacts/embeddings/ncit_doid/hf_mean_pool-sapbert/20261009T182919Z-95011340` copied (no `shards/`) to `artifacts/embeddings/ncit_doid_unsup/hf_mean_pool-sapbert/` with [copy_embedding_store.py](../../scripts/copy_embedding_store.py). All checks pass: byte-identical files, manifest hashes (`AlignedMatrix.load`), node order against the `ncit_doid_unsup` ingest `20261026T024157Z-3bde93ef` (24,227 of 24,227), and `input_sha256` equals the ingest's `label_synonyms_definition` text hash for 24,227 of 24,227 nodes. The other run folder in that directory (`20261009T185331Z-390d8bc3`) holds no store files and was not used. DeepWalk is the store 9.1 copied.
- **Matrix script:** [run_unsup_matrix.sh](../../scripts/run_unsup_matrix.sh) takes `TEXT_EMBEDDER` and `PREFIX`. With a prefix it sets `training_data.inputs.text_embeddings.embedder_id`, the sampler id (`sap-rank_topk-k5[-negrandom|-neghard-off10|-posmutual]`) and the model id (`sap-<row>-s<seed>`), so the 9.2 `LATEST` pointers are untouched. Without them the commands are those of 9.2. The `training_data` manifest of the SapBERT lineage records `hf_mean_pool-sapbert` as its text input.
- **Summary script:** [unsup_summary.py](../../scripts/unsup_summary.py) gained `--prefix`, `--text-label`, and `--compare <run>` (+ `--compare-prefix`, `--compare-label`) which appends the two-encoder MRR / F1 table with the difference second − first. Two unit tests.
- **Evaluation config:** [ncit_doid_unsup_sapbert_test.yaml](../../configs/experiments/ncit_doid_unsup_sapbert_test.yaml): `sap-u0` (frozen `hf_mean_pool-sapbert`) and the 21 `sap-*` models; the same two evaluations as 9.2 (`gold_alignment` global with `sibling_filter_splits: [test]`; `gold_alignment_prf` top-3 greedy one-to-one, threshold 0, no BioGITOM filters).
- **Training:** 21 runs (7 rows × seeds 42 / 43 / 44) with the 9.2 fixed settings, nothing re-tuned: `rank_topk` k 5, Adam 1e-3, 100 epochs (last epoch), triplet margin 1.0, InfoNCE temperature 0.1, U1 `residual_adapter` / `mlp` λ 0.25 `alpha 1.0`, U2 `gcn` `alpha 0.5`. **SapBERT is frozen throughout; what is trained are the projection (U1) and the GCN (U2) on top of its vectors.** About 1 h 25 min on one GPU.
- **Evaluation:** once, run `20261026T131132Z-5d0dc63e` (3,280 gold test pairs).

## 2. Paper table (SapBERT)

Table X. Unsupervised ablation on NCIT–DOID with frozen SapBERT (3,280 held-out gold test pairs; no gold pair used in training; all hyperparameters fixed in advance). Mean ± sample std over 3 seeds (U0 is deterministic). P/R/F1: greedy one-to-one over top-3 cosine candidates, threshold 0, no train-label filter. #Gold = 3,280 for every row. SapBERT was pretrained on UMLS synonym pairs (see section 5).

| Variant | Pseudo-pairs | Loss | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct |
|---|---|---|---|---|---|---|---|---|---|---|
| SapBERT (frozen) | – | – | 0.3424 | 0.6537 | 0.4475 | 0.4663 | 0.3716 | 0.4136 | 2614 | 1219 |
| SapBERT + projection | top-k | Triplet | 0.3298 ± 0.0008 | 0.6139 ± 0.0005 | 0.4276 ± 0.0004 | 0.4541 ± 0.0007 | 0.3539 ± 0.0015 | 0.3978 ± 0.0012 | 2556.0 ± 7.2 | 1160.7 ± 5.0 |
| SapBERT + projection | top-k | InfoNCE | 0.3175 ± 0.0034 | 0.5925 ± 0.0044 | 0.4129 ± 0.0035 | 0.4204 ± 0.0077 | 0.3345 ± 0.0052 | 0.3725 ± 0.0062 | 2609.7 ± 7.2 | 1097.0 ± 17.1 |
| SapBERT + GCN (text + structure) | top-k | Triplet | 0.1106 ± 0.0073 | 0.3544 ± 0.0099 | 0.1899 ± 0.0072 | 0.1449 ± 0.0078 | 0.1001 ± 0.0051 | 0.1184 ± 0.0061 | 2266.3 ± 13.7 | 328.3 ± 16.6 |
| SapBERT + GCN (text + structure) | top-k | InfoNCE | 0.2738 ± 0.0122 | 0.6973 ± 0.0083 | 0.4088 ± 0.0119 | 0.3481 ± 0.0104 | 0.2886 ± 0.0099 | 0.3156 ± 0.0101 | 2719.0 ± 15.1 | 946.7 ± 32.3 |
| SapBERT + GCN | random neg. | Triplet | 0.1365 ± 0.0027 | 0.4287 ± 0.0074 | 0.2318 ± 0.0019 | 0.1723 ± 0.0013 | 0.1258 ± 0.0015 | 0.1454 ± 0.0015 | 2394.7 ± 12.3 | 412.7 ± 4.9 |
| SapBERT + GCN | hard neg. | Triplet | 0.1021 ± 0.0040 | 0.4102 ± 0.0070 | 0.1991 ± 0.0021 | 0.1490 ± 0.0067 | 0.1021 ± 0.0043 | 0.1212 ± 0.0052 | 2248.7 ± 9.1 | 335.0 ± 14.1 |
| SapBERT + GCN | mutual top-k | Triplet | 0.1689 ± 0.0021 | 0.4985 ± 0.0085 | 0.2759 ± 0.0016 | 0.2146 ± 0.0038 | 0.1645 ± 0.0047 | 0.1862 ± 0.0045 | 2515.0 ± 28.6 | 539.7 ± 15.5 |

BERT (9.2, run `20261026T043657Z-b3585c20`) against SapBERT (this run), MRR and F1, difference = SapBERT − BERT:

| Row | Pseudo-pairs | Loss | BERT MRR | SapBERT MRR | Δ MRR | BERT F1 | SapBERT F1 | Δ F1 |
|---|---|---|---|---|---|---|---|---|
| frozen encoder | – | – | 0.0186 | 0.4475 | +0.4289 | 0.0204 | 0.4136 | +0.3933 |
| + projection | top-k | Triplet | 0.0160 | 0.4276 | +0.4116 | 0.0186 | 0.3978 | +0.3792 |
| + projection | top-k | InfoNCE | 0.0177 | 0.4129 | +0.3952 | 0.0175 | 0.3725 | +0.3550 |
| + GCN (text + structure) | top-k | Triplet | 0.0040 | 0.1899 | +0.1860 | 0.0002 | 0.1184 | +0.1182 |
| + GCN (text + structure) | top-k | InfoNCE | 0.0093 | 0.4088 | +0.3995 | 0.0039 | 0.3156 | +0.3117 |
| + GCN | random neg. | Triplet | 0.0035 | 0.2318 | +0.2283 | 0.0004 | 0.1454 | +0.1451 |
| + GCN | hard neg. | Triplet | 0.0062 | 0.1991 | +0.1930 | 0.0010 | 0.1212 | +0.1202 |
| + GCN | mutual top-k | Triplet | 0.0084 | 0.2759 | +0.2674 | 0.0010 | 0.1862 | +0.1853 |

Reading it:

| question | answer from this run |
|---|---|
| Does training on pseudo-pairs improve over frozen SapBERT (U1 vs U0)? | No: MRR 0.4276 (triplet) and 0.4129 (InfoNCE) against 0.4475; F1 0.3978 / 0.3725 against 0.4136. |
| Does graph structure add to U1 (U2 vs U1)? | No for MRR / F1 (0.1899 vs 0.4276 triplet; 0.4088 vs 0.4129 InfoNCE). The GCN with InfoNCE is the only trained row above U0, on Hits@10 only. |
| Triplet vs InfoNCE | Triplet is ahead for the projection (MRR 0.4276 vs 0.4129); InfoNCE is far ahead for the GCN (0.4088 vs 0.1899). |
| Easy vs random negatives (U2-rand vs U2) | Random is better: MRR 0.2318 vs 0.1899, F1 0.1454 vs 0.1184. |
| Hard negatives (U2-hard vs U2) | Slightly better MRR (0.1991 vs 0.1899), same F1 (0.1212 vs 0.1184), within about 1.3 stds of the default's seed spread. |
| Cleaner positives (U2-mut vs U2) | Better: MRR 0.2759 vs 0.1899, F1 0.1862 vs 0.1184. |

The seed stds are small for the projection rows (≤ 0.0035 MRR) and larger for the GCN rows (up to 0.012), so the projection-versus-U0 gaps (−0.020 and −0.035 MRR) and the GCN-triplet gaps are well outside seed noise, while the hard-versus-default difference is not. One matrix, three seeds, no significance test.

## 3. Gate 9.3: U0 checks

| check | result |
|---|---|
| U0 of this run equals `sapbert` evaluated alone as an `embedding_store` (run `20261026T131922Z-56bb7133`, same 3,280 pairs) | All 4 ranking and 8 P/R/F1 / count values are identical. |
| With the default all-splits sibling filter the frozen-SapBERT ranking equals Sprint 5 / 6.3 A0 | Hits@1 0.3424, Hits@5 0.5677, Hits@10 0.6537, MRR 0.4475: equal to the stored A0 values (`dev_docs/execution_logs/sprint_6_3.md`). |
| The restricted sibling filter (`[test]`) does not change U0 here | Same ranking values as the all-splits filter. |
| P/R/F1 equals the Sprint 8 A0 row (`prf_test_greedy_nofilt`, runs `20261025T215047Z-4e19066f` / `…215058Z…`) | P 0.4663, R 0.3716, F1 0.4136, 2,614 predicted, 1,219 correct: identical. |

The Sprint 5 / 6.3 / 8 numbers were read from their stored evaluation runs, not recomputed.

## 4. Training runs (`gcl_train`, `models/gcl/ncit_doid_unsup/sap-<row>-s<seed>/<run>`; date 20261026)

| row | runs (seeds 42 / 43 / 44) |
|---|---|
| `sap-u1` | T120600Z-8b599764 / T122718Z-4b95ea1c / T125004Z-65951191 |
| `sap-u2` | T120841Z-c75f1964 / T123017Z-69fdb0f5 / T125250Z-f360d0ae |
| `sap-u1-i` | T120907Z-975caba5 / T123048Z-9b5e1b63 / T125318Z-d7b2b0dd |
| `sap-u2-i` | T121901Z-b8c514e5 / T124045Z-76939c07 / T130316Z-5b22d59e |
| `sap-u2-rand` | T122252Z-9ef54365 / T124459Z-fc9e5adf / T130731Z-e7eacc29 |
| `sap-u2-hard` | T122426Z-2a4101d3 / T124714Z-243fd836 / T130858Z-b76f42a3 |
| `sap-u2-mut` | T122611Z-46959010 / T124902Z-db11a518 / T131051Z-a07b06d8 |

Matrix evaluation: `20261026T131132Z-5d0dc63e` (`summary.md` and `summary.csv` in the run directory, including the comparison table).

## 5. Disclosures and limitations

- **SapBERT carries external alignment knowledge.** It was pretrained on UMLS synonym pairs. NCIT and DOID classes that share a UMLS concept were pulled together during that pretraining, so frozen SapBERT already knows part of the cross-vocabulary mapping, which is a likely reason for its very large lead over `bert_base`. No gold pair from this dataset is used by our training or by the evaluator's sibling filter or pair set, but the SapBERT rows must not be described as using no external alignment information.
- **Test reads.** Gold test pairs were read once for the 22 methods of the matrix, and once more for the single-method U0 check (`…131922Z…`). No setting, row, epoch count or threshold was chosen from any read; settings were fixed in Phase 9.2 and not adapted to SapBERT. Phase 9.1 / 9.2 reads are disclosed in their logs.
- The hyperparameters were set for `bert_base` in 9.2 and are not tuned for SapBERT, so the SapBERT rows could in principle do better with a different learning rate, epoch count, `λ` or `alpha`. That is out of scope for this phase.
- GCN kernels are not bit-reproducible on GPU, so trained rows report mean ± sample std over 3 seeds; U0 is exact.
- All 3,280 test pairs are scored for every method (no gold endpoint is among the 6 nodes without a DeepWalk vector).
- ATT&CK–CWE is not part of this phase (no SapBERT store for it).

## 6. Commands

```bash
uv run pytest tests/unit -q
uv run python scripts/copy_embedding_store.py --src artifacts/embeddings/ncit_doid/hf_mean_pool-sapbert/20261009T182919Z-95011340 \
    --dst-lineage artifacts/embeddings/ncit_doid_unsup/hf_mean_pool-sapbert \
    --ingest data/processed/ncit_doid_unsup/20261026T024157Z-3bde93ef --text-variant label_synonyms_definition
TEXT_EMBEDDER=hf_mean_pool-sapbert PREFIX=sap- bash scripts/run_unsup_matrix.sh ncit_doid_unsup 1 2 3
uv run cga run evaluate --dataset ncit_doid_unsup --experiment ncit_doid_unsup_sapbert_test
uv run python scripts/unsup_summary.py --dataset ncit_doid_unsup --run 20261026T131132Z-5d0dc63e \
    --prefix sap- --text-label SapBERT --compare 20261026T043657Z-b3585c20 --compare-label BERT
```

## 7. Files

New: `configs/experiments/ncit_doid_unsup_sapbert_test.yaml`.
Modified: `scripts/run_unsup_matrix.sh` (text embedder and prefix), `scripts/unsup_summary.py` (`--prefix`, `--text-label`, `--compare`), `tests/unit/test_unsup_summary.py`, `dev_docs/sprints/sprint_9.md` (ticks).
