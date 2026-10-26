# Sprint 8, Phase 8.2: P/R/F1 table for our variants next to BioGITOM

Plan: [sprint_8.md](../sprints/sprint_8.md). Evaluator and gate: [8.1](sprint_8_1.md). Analysis and risks: [biogitom_vs_srga_prf_comparability.md](../data_analysis/biogitom_vs_srga_prf_comparability.md). Nothing was trained and no earlier number changed.

## 0. Outcome in brief

- **No gate (report only).** The best comparable row, **A25 A19 (full-data fusion), filters on, threshold 0.0, `greedy_one_to_one`: P 0.6926 / R 0.5921 / F1 0.6384** (2,804 predicted, 1,942 correct, 3,280 gold). BioGITOM logs P 0.924 / R 0.911 / F1 0.917 (3,236 predicted, 2,989 correct, 3,280 gold). The gap is about 0.28 F1.
- **Check required by the plan (A0 under the same code): done.** The evaluator reproduces our own earlier numbers (section 5), so the gap is not an artefact of this code. It is a gap between our models and BioGITOM's reported result, subject to the caveats in section 6.
- **P/R/F1 and MRR tell the same story** (A19: F1 0.638 against global MRR 0.6385); the ordering of methods is the Sprint 5 / 6 one (A0 < A3 < A4/A11/A12 < A17 < A18 < A19).
- **The train-label filter matters a lot, as expected, and helps the no-floor models most.** On test (train-only lineage, greedy), filters off to on: A12 0.4384 → 0.5097 (+0.071), A15 +0.063, A19 +0.069, A11 +0.046, A0 +0.031. Unfiltered, A12 (0.4384) is clearly below A11 (0.4663); filtered, the two tie (0.5097 vs 0.5125, within noise) in the train-only lineage, and A25 A12 is ahead of A25 A11 (0.5424 vs 0.5227).
- **Greedy one-to-one lifts precision far above top-1**, as risk 2 of the analysis said: A0 test filters on, P 0.527 (greedy) vs 0.371 (top1); F1 0.445 vs 0.369.
- **The val-chosen threshold is nearly flat** (section 4): +0.003 to +0.011 F1 on test, mostly a precision-for-recall swap.

## 1. What was run

| item | file / run |
|---|---|
| Val, train-only lineage, 12 methods (A0, A1, A3, A4, A11, A12, A14, A15, A17, A18, A19, A20), 4 evaluations (filters on/off × greedy/top1) | [ncit_doid_srga_prf.yaml](../../configs/experiments/ncit_doid_srga_prf.yaml), run `20261025T213718Z-b6f8d965` |
| Threshold sweep, val | [srga_prf_threshold.py](../../scripts/srga_prf_threshold.py) (A12 3 seeds, A19 = 0.5 A12 + 0.5 A11, grid 0.00 to 0.95 step 0.05, filters off and on) |
| Test (once), train-only lineage `ncit_doid_srga`: A0, A1, A3, A4, A11, A12, A14, A15, A17, A18, A19; 4 + 4 evaluations | [ncit_doid_srga_prf_test.yaml](../../configs/experiments/ncit_doid_srga_prf_test.yaml), run `20261025T214058Z-62fc1374` |
| Test (once), full-data lineage `ncit_doid_srga_full`: A25 A3, A11, A12 (3 seeds), A25 A19 | [ncit_doid_srga_prf_full_test.yaml](../../configs/experiments/ncit_doid_srga_prf_full_test.yaml), run `20261025T215058Z-4e19066f` |
| Table printer | [srga_prf_table.py](../../scripts/srga_prf_table.py) (mean ± sample std over seeds) |

Deviation: the plan names one test config; a config is read per dataset lineage and the model ids differ, so the full-data lineage has its own file. The `*_thr` evaluations (threshold ablation) sit in both test configs. The full-data test was launched twice by mistake (the first launch was also still queued in a background shell), so `artifacts/evaluations/ncit_doid_srga_full/` holds two identical runs (`20261025T215047Z-4e19066f` and `20261025T215058Z-4e19066f`); the tables use the second.

Seeds 42 / 43 / 44 for A3, A4, A11, A12, A14, A15 and A25 models. A0, A1 and the fusions (A17 to A20, A25 A19) have no seed spread. A4 ± is over seeds as in Sprint 5.

## 2. Val table (211 pairs, `n_gold` = 211, recall ceiling 0.9953)

P / R / F1 of `greedy_one_to_one` at threshold 0.0 (mean over seeds), and F1 of `top1`.

| method | no filter P / R / F1 | filters P / R / F1 | top1 F1, no filter | top1 F1, filters |
|---|---|---|---|---|
| A0 SapBERT | 0.3143 / 0.3128 / 0.3135 | 0.3350 / 0.3270 / 0.3309 | 0.2945 | 0.3150 |
| A1 nb dir | 0.3333 / 0.3318 / 0.3325 | 0.3689 / 0.3602 / 0.3645 | 0.3373 | 0.3580 |
| A3 MLP | 0.3444 / 0.3428 / 0.3436 ± 0.0055 | 0.3750 / 0.3697 / 0.3723 ± 0.0050 | 0.3500 | 0.3756 |
| A4 relational | 0.3429 / 0.3412 / 0.3420 ± 0.0190 | 0.4013 / 0.3918 / 0.3965 ± 0.0153 | 0.3436 | 0.3943 |
| A11 GCN + floor | 0.3683 / 0.3665 / 0.3674 ± 0.0110 | 0.3837 / 0.3807 / 0.3822 ± 0.0095 | 0.3642 | 0.3775 |
| A12 GCN no floor | 0.4016 / 0.3997 / 0.4006 ± 0.0337 | 0.4550 / 0.4471 / 0.4510 ± 0.0280 | 0.3959 | 0.4429 |
| A14 GAT + floor | 0.3683 / 0.3665 / 0.3674 ± 0.0167 | 0.3907 / 0.3870 / 0.3889 ± 0.0215 | 0.3563 | 0.3772 |
| A15 GAT no floor | 0.3841 / 0.3823 / 0.3832 ± 0.0180 | 0.4428 / 0.4344 / 0.4386 ± 0.0267 | 0.3816 | 0.4330 |
| A17 ens. of A11 | 0.3762 / 0.3744 / 0.3753 | 0.3971 / 0.3934 / 0.3952 | 0.3705 | 0.3848 |
| A18 ens. of A12 | 0.4524 / 0.4502 / 0.4513 | 0.5340 / 0.5213 / 0.5276 | 0.4513 | 0.5203 |
| **A19 fusion** | **0.5238 / 0.5213 / 0.5226** | **0.5694 / 0.5640 / 0.5667** | 0.5226 | 0.5653 |
| A20 grand ens. | 0.5048 / 0.5024 / 0.5036 | 0.5481 / 0.5403 / 0.5442 | 0.4988 | 0.5333 |

`n_predicted` is 206 to 210 everywhere (the filters remove 0 to 4 predictions); the ± is the sample std of F1 over seeds. With 211 pairs one pair is 0.005 of recall, so differences under about 0.02 are not meaningful on val.

## 3. Test tables (3,280 pairs, read once)

Recall ceiling of a one-to-one assignment: 3,245 unique sources and targets → **0.9893**; `n_unique_sources` = 3,245, so `top1` predicts 3,245 pairs when unfiltered.

### 3.1 Train-only lineage, threshold 0.0, `greedy_one_to_one`

| method | no filter P / R / F1 | filters P / R / F1 | n_predicted / n_correct (filters) |
|---|---|---|---|
| A0 SapBERT | 0.4663 / 0.3716 / 0.4136 | 0.5270 / 0.3845 / 0.4446 | 2,393 / 1,261 |
| A1 | 0.5125 / 0.4174 / 0.4601 | 0.5752 / 0.4305 / 0.4924 | 2,455 / 1,412 |
| A3 | 0.4698 / 0.3991 / 0.4316 ± 0.0054 | 0.5484 / 0.4223 / 0.4771 ± 0.0060 | 2,526 / 1,385 |
| A4 | 0.4951 / 0.4173 / 0.4529 ± 0.0141 | 0.6000 / 0.4453 / 0.5112 ± 0.0112 | 2,434 / 1,461 |
| A11 | 0.5086 / 0.4305 / 0.4663 ± 0.0003 | 0.5876 / 0.4544 / 0.5125 ± 0.0020 | 2,536 / 1,490 |
| A12 | 0.4556 / 0.4225 / 0.4384 ± 0.0079 | 0.5536 / 0.4722 / 0.5097 ± 0.0057 | 2,797 / 1,549 |
| A14 | 0.4985 / 0.4276 / 0.4604 ± 0.0044 | 0.5731 / 0.4489 / 0.5034 ± 0.0048 | 2,569 / 1,472 |
| A15 | 0.4420 / 0.4074 / 0.4240 ± 0.0086 | 0.5338 / 0.4484 / 0.4874 ± 0.0097 | 2,755 / 1,471 |
| A17 | 0.5220 / 0.4418 / 0.4785 | 0.6023 / 0.4668 / 0.5259 | 2,542 / 1,531 |
| A18 | 0.5280 / 0.4912 / 0.5089 | 0.6358 / 0.5497 / 0.5896 | 2,836 / 1,803 |
| **A19** | **0.5752 / 0.5305 / 0.5519** | **0.6706 / 0.5784 / 0.6211** | 2,829 / 1,897 |

### 3.2 Train-only lineage, `top1` (BioGITOM's "@1" variant), F1 (P / R in the run)

| method | no filter | filters |
|---|---|---|
| A0 | 0.3436 (P 0.3455, R 0.3418) | 0.3689 |
| A3 | 0.3800 | 0.4137 |
| A4 | 0.3925 | 0.4308 |
| A11 | 0.3998 | 0.4354 |
| A12 | 0.4088 | 0.4698 |
| A14 | 0.4044 | 0.4388 |
| A15 | 0.3930 | 0.4520 |
| A17 / A18 | 0.4077 / 0.4684 | 0.4435 / 0.5373 |
| **A19** | **0.4962** (P 0.4989, R 0.4936) | **0.5516** (P 0.5551, R 0.5482) |

(A1: 0.3831 / 0.4103.) `top1` gives about one prediction per source, so P ≈ R.

### 3.3 Full-data lineage (A25: the same 1,406 training pairs as BioGITOM)

| method | greedy, no filter P / R / F1 | **greedy, filters P / R / F1** | top1 F1, no filter / filters |
|---|---|---|---|
| A0 SapBERT | 0.4663 / 0.3716 / 0.4136 | 0.5404 / 0.3875 / 0.4513 | 0.3436 / 0.3733 |
| A25 A3 | 0.4699 / 0.4021 / 0.4334 ± 0.0040 | 0.5603 / 0.4263 / 0.4842 ± 0.0020 | 0.3840 / 0.4210 |
| A25 A11 | 0.5121 / 0.4348 / 0.4703 ± 0.0049 | 0.6038 / 0.4608 / 0.5227 ± 0.0053 | 0.4038 / 0.4428 |
| A25 A12 | 0.4780 / 0.4465 / 0.4617 ± 0.0033 | 0.5904 / 0.5016 / 0.5424 ± 0.0027 | 0.4271 / 0.4964 |
| **A25 A19** | **0.5844 / 0.5427 / 0.5628** | **0.6926 / 0.5921 / 0.6384** | **0.5079 / 0.5702** |

The filtered A0 row differs between lineages (0.4446 vs 0.4513) only because the filter's "train-only" set differs: the full-data lineage has the 211 val pairs as train pairs.

## 4. Threshold sweep (val only) and its single application to test

Grid and procedure: section 1. Per method on val, mean F1 (A12 over 3 seeds), greedy:

| method | filters | F1 at 0.0 | best F1 (threshold) | P / R at the best |
|---|---|---|---|---|
| A12 | off | 0.4006 | 0.4009 (0.55) | 0.4054 / 0.3965 |
| A12 | on | 0.4510 | **0.4559 (0.60)** | 0.4918 / 0.4250 |
| A19 | off | 0.5226 | 0.5251 (0.50) | 0.5288 / 0.5213 |
| A19 | on | 0.5667 | **0.5700 (0.55)** | 0.5918 / 0.5498 |

Up to 0.45 every threshold gives the same set (all kept cosines are higher); above 0.70 recall collapses (A19, filters on: P 0.65 / R 0.32 at 0.70, P 1.00 / R 0.005 at 0.90). The best-F1 gain over 0.0 is +0.005 (A12) and +0.003 (A19), well inside the seed std of A12 (0.028), so **the val data do not identify a useful threshold**; the curve is a plateau followed by a precision-for-recall trade. The labelled rows use the filters-on optimum of each method (A12 0.60, A19 0.55). The full-data lineage has no val, so it reuses the thresholds chosen on the train-only models.

Test, applied once, greedy, filters on (headline is still threshold 0.0):

| model | threshold | P / R / F1 | change in F1 vs 0.0 |
|---|---|---|---|
| A12 train-only | 0.60 | 0.6027 / 0.4551 / 0.5186 ± 0.0072 | +0.0089 |
| A19 train-only | 0.55 | 0.6973 / 0.5640 / 0.6236 | +0.0025 |
| A25 A12 | 0.60 | 0.6446 / 0.4849 / 0.5534 ± 0.0013 | +0.0110 |
| A25 A19 | 0.55 | 0.7149 / 0.5750 / 0.6374 | −0.0010 |

Filters off: A12 0.4384 → 0.4477 (+0.009), A19 0.5519 → 0.5555 (+0.004), A25 A12 0.4617 → 0.4730, A25 A19 0.5628 → 0.5645. The threshold raises precision (A25 A19: 0.693 → 0.715) and lowers recall (0.592 → 0.575); F1 moves by about 0.01 at most, and for A25 A19 it does not move. It does not close any part of the gap to BioGITOM.

## 5. The comparison rows and the A0 check

**Comparable to BioGITOM** (full-data lineage, filters on, threshold 0.0, `greedy_one_to_one`, top-3):

| system | P | R | F1 | predicted / correct / gold |
|---|---|---|---|---|
| BioGITOM (logs below) | 0.924 | 0.911 | 0.917 | 3,236 / 2,989 / 3,280 |
| **A25 A19 (ours, best)** | 0.6926 | 0.5921 | 0.6384 | 2,804 / 1,942 / 3,280 |
| A25 A12 | 0.5904 ± 0.0044 | 0.5016 ± 0.0015 | 0.5424 ± 0.0027 | 2,787 / 1,645 / 3,280 |
| A25 A11 | 0.6038 ± 0.0061 | 0.4608 ± 0.0048 | 0.5227 ± 0.0053 | 2,503 / 1,511 / 3,280 |
| A25 A3 | 0.5603 ± 0.0023 | 0.4263 ± 0.0020 | 0.4842 ± 0.0020 | 2,496 / 1,398 / 3,280 |
| A0 SapBERT | 0.5404 | 0.3875 | 0.4513 | 2,352 / 1,271 / 3,280 |

Unfiltered, next to them (so the reader sees what the train-label filter contributes): A25 A19 0.5844 / 0.5427 / **0.5628**; A25 A12 0.4780 / 0.4465 / 0.4617; A25 A11 0.5121 / 0.4348 / 0.4703; A25 A3 0.4699 / 0.4021 / 0.4334; A0 0.4663 / 0.3716 / 0.4136. The filter is worth +0.076 F1 to A25 A19 and +0.081 to A25 A12; neither system comes within 0.27 of BioGITOM's F1 with it off or on.

BioGITOM's "@1": P 0.9237 / R 0.9138 / F1 0.9187. Ours (A25 A19, filters on): 0.5742 / 0.5662 / 0.5702 (unfiltered 0.5106 / 0.5052 / 0.5079).

Source of the BioGITOM figures: `Experiments/ncit2doid/Negative_Number_Ajdustment/Results_log_ncit2doid_train_100.md` (l.2468 to 2474) and `Experiments/ncit2doid/Similarity Metrics/Results_log_ncit2doid_Similarity.md` (l.3291), as read in [biogitom_ncit_doid_evaluation_protocol.md](../data_analysis/biogitom_ncit_doid_evaluation_protocol.md) section 5. I did not re-read those logs here. The "main" `k=3` log gives P 0.931 / R 0.906 / F1 0.919 instead; the ordering of the conclusions does not depend on which is cited.

**A0 check (plan: check A0 under the same code before reading a gap as a method gap).**
- Val: unfiltered `top1` recall 0.2938 = Sprint 5 A0 val Hits@1 (62 / 211), exactly.
- Test: unfiltered `top1` recall 0.3418 (1,121 / 3,280) against the Sprint 5 test global Hits@1 0.3424 (1,123). The 2-pair difference is expected: the ranking evaluator removes a query's other gold targets from the pool, `top1` here does not.
- Every other method also lands where its Sprint 5 / 6 Hits@1 and MRR order puts it, and F1 of the best method (0.638) is close to its MRR (0.6385).
- So the evaluator is consistent with the earlier ranking numbers, and the gap is between our models and BioGITOM's reported figures. It is not a bug of this code. What the check cannot show is that BioGITOM's own pipeline is replicated; that code was not run here.

## 6. Caveats (restated from the analysis)

1. **BioGITOM's test-scored experiment logs.** Its experiment logs score test across negative counts, similarity metrics and `k`; whether settings were chosen on test is not stated. Its headline may be mildly optimistic. Ours is selected on val.
2. **The DeepOnto formula is unverified.** DeepOnto's `AlignmentEvaluator.f1` is not in that repo. The logged counts match the standard formulas (2,989 / 3,236 / 3,280 gives 0.924 / 0.911 / 0.917), and this evaluator uses them.
3. **The train-label filter uses `train.tsv` at inference.** It is applied to every method in the "filters on" rows and reported beside the unfiltered rows. It helps methods that memorise their train targets most (A12, A15), so compare filtered with filtered and read the unfiltered rows as the label-free number.
4. **Greedy one-to-one is post-processing** and lifts precision above what the embeddings give (top1 rows).
5. **Training pairs.** Only the full-data (A25) lineage matches BioGITOM's 1,406 training pairs; the train-only rows are conservative.
6. **Score:** cosine here, `1 / (1 + L2)` in BioGITOM. For unit vectors the rankings are identical; only a threshold above 0 differs, and the 0.0 headline is unaffected.
7. **The fusion rows have no seed std** (one ensemble). Single-model ± is over three seeds, sample std; training is not bit-reproducible.
8. **Test reads:** test was read once per configuration above (train-only: 8 + 4 evaluations; full-data: 8 + 4). The thresholds come from val only. No selection used test.
9. The threshold grid is coarse (0.05), and the val optimum is not significant (section 4).
10. `ruff` is not installed, so lint was not run.

## 7. Files

- Created: `configs/experiments/ncit_doid_srga_prf{,_test,_full_test}.yaml`, `scripts/srga_prf_table.py`, `scripts/srga_prf_threshold.py`, this log.
- Modified: `README.md`, `dev_docs/sprints/sprint_8.md` (boxes ticked).
- Untracked artifacts: evaluate runs `20261025T213718Z-b6f8d965` (val), `20261025T214058Z-62fc1374` (test, train-only), `20261025T215058Z-4e19066f` and a duplicate `20261025T215047Z-4e19066f` (test, full-data).

## 8. Commands

```
uv run pytest tests/unit -q
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_prf
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_prf_test
uv run cga run evaluate --dataset ncit_doid_srga_full --experiment ncit_doid_srga_prf_full_test
uv run python scripts/srga_prf_table.py --dataset ncit_doid_srga --run 20261025T214058Z-62fc1374
uv run python scripts/srga_prf_table.py --dataset ncit_doid_srga_full --run 20261025T215058Z-4e19066f
uv run python scripts/srga_prf_threshold.py --split val --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}" --fuse A19=A11:0.5,A12:0.5 --report A12 --report A19
```
