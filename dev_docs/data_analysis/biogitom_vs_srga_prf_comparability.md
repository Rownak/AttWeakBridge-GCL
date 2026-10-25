# BioGITOM vs our NCIT–DOID protocol: can we report comparable P / R / F1?

Sources: [biogitom_ncit_doid_evaluation_protocol.md](biogitom_ncit_doid_evaluation_protocol.md) (BioGITOM side, taken from its saved logs); our `gold_alignment` evaluator, the `ncit_doid_srga*` dataset configs and the Sprint 5–6 logs.
Nothing was run for this report. BioGITOM's F1 comes from DeepOnto, which is not in that repo, so its exact formula is unverified.

## 1. Verdict

**The test split is the same. The evaluation protocol is materially different.** We report ranking metrics (MRR, Hits@k). BioGITOM reports P/R/F1 of a **global one-to-one assignment**. Our methods can produce that output without retraining, but we have **not** produced it yet, so no comparable number exists today. It needs one new analysis script (section 5).

## 2. Split comparison

| aspect | BioGITOM | ours |
|---|---|---|
| Gold files | `refs_equiv/train.tsv`, `test.tsv` (3,280) | same files |
| Test pairs | 3,280 | 3,280 (identical to Sprint 4; A0 test numbers were bit-identical across lineages) |
| Train pairs used | all 1,406 | train-only lineage: 1,195 (211 carved as val; 1,174 usable). **Full-data lineage `ncit_doid_srga_full` (A25): all 1,406 (1,383 usable)** |
| Validation | internal 70/30 split of train pairs (for the gated network) | 211 pairs carved from train by source, used for all selection |
| Test used for tuning | Experiment logs show P/R/F1 on test across negative counts (50/100), similarity metrics and `k` (3/10). Whether choices were made on them is not stated | Selection on val only. A few unplanned test reads are disclosed in the logs (A13, A16) |
| Ontologies in the model | both embedded and passed through the GNN in full, including test nodes (transductive, no test labels) | same (transductive, features only) |
| Seeds | partly seeded; negative sampling is not | 3 fixed seeds for trained models |

Only the full-data lineage (A25 models) uses the same training pairs as BioGITOM. The train-only models (A3, A11, A12, A19 as reported in Sprint 5–6) see 15% fewer pairs.

## 3. Evaluation protocol comparison

| aspect | BioGITOM | ours |
|---|---|---|
| Unit | the set of predicted `(src, tgt)` pairs | one ranked list per gold pair |
| Output | top-3 per source, then **greedy one-to-one** assignment (sorted by score, a source and a target each used once), score threshold 0.0 | full ranking, no assignment |
| Search space | every test source against all DOID nodes (exact L2 search) | every DOID node (global), minus the other gold targets of the same source. Candidate mode: 101 per pair |
| Pre-filters | drops predictions whose source or target is in `train.tsv` but not in `test.tsv`; keeps only test sources | none. Train-only DOID nodes stay in the pool |
| Score | 1 / (1 + L2 distance) | cosine |
| Correct | the exact pair is in `test.tsv` | gold target ranked first / at k |
| Headline | P 0.924, R 0.911, F1 0.917 to 0.919 (different logs give different numbers; no 0.918 found) | MRR, Hits@1/5/10 |

## 4. What P / R / F1 requires, and whether each method can give it

**Required.** A set of `(NCIT, DOID)` pairs, scored as P = correct / |predictions|, R = correct / 3,280, F1 = harmonic mean. With threshold 0.0 nothing is rejected on score, so the number of predictions is set by the assignment (about one per test source), not by confidence. P and R therefore sit close together. This is not a precision/recall trade-off in the usual sense.

**Can our methods produce it fairly?**
- **A0, A1, A3, A4, A11–A16.** Yes. Each gives node embeddings, so the test-source × DOID cosine matrix gives the pair list.
- **A17–A20, A25.** Yes. The fusion is a weighted mean of member cosines, so its matrix is the same weighted mean.
- **Cosine vs `1/(1+L2)`.** For unit vectors both rank identically. This only matters if a score threshold above 0 is used. A12 and A15 vectors are not unit-length before normalisation, so cosine is the fair choice.
- **A21–A24 (re-ranker).** Not fitted (probe gate failed). Nothing to report.
- **Fairness within our own table.** Equal. All methods get the same pool, filters and assignment.

## 5. Leakage and comparability risks

1. **The train-only filter uses train labels at inference.** BioGITOM removes predictions whose endpoint is a train-only node, which shrinks the pool using `train.tsv`. We currently do not. The filter probably helps models that memorise train targets (A12: train MRR 0.99), so it should be applied to every method and reported both with and without it.
2. **Greedy one-to-one is a post-processing step.** It can lift precision beyond what the embeddings alone give. Report ranking-only top-1 as well (P@1, R@1, F1@1), which BioGITOM also logs (0.924 / 0.914 / 0.919).
3. **Recall ceiling.** One-to-one cannot recover 1:n gold pairs. Report the number of unique test sources and targets, since the ceiling is below 3,280 when they repeat. I did not check those counts.
4. **Training pairs differ.** Compare BioGITOM only with the full-data lineage (A25). Train-only numbers are conservative.
5. **Hyperparameter selection.** BioGITOM's experiment logs score test across settings, so its headline may be mildly optimistic. Our numbers are selected on val.
6. **Metric formula.** DeepOnto's `AlignmentEvaluator.f1` is not available. The logs match the standard formulas (2,989 / 3,236 / 3,280 gives 0.924 / 0.911 / 0.917), so the standard formulas are a safe assumption.
7. **Other BioGITOM runs differ.** The main log (k=3) gives P 0.931, R 0.906, F1 0.919. Pick one run and cite it.
8. **Transductive features.** Both sides use test nodes as features only. No difference.

## 6. Minimal changes to report comparable metrics

No retraining. Add one analysis script, in the style of the existing `scripts/srga_*` tools. It reuses `model_run_dir()` and `add_fusions()` from `srga_paired_ranks.py`.

1. Load `train.tsv` and `test.tsv` for the pair sets and their entities.
2. For each method, build the cosine matrix for the **test sources** against the **DOID nodes**.
3. Apply BioGITOM's two filters (drop train-only endpoints, keep test sources only). Also run the unfiltered variant.
4. Take the top-3 per source, sort by score, and greedily assign one-to-one with threshold 0.0.
5. Score against `test.tsv`: P, R, F1. Also report top-1 per source (P@1, R@1, F1@1).
6. Run it for A0, A3, A11, A12, A19 on the **full-data lineage** (A25 models) and on the train-only lineage. Report the mean over seeds for single models.
7. Optional: sweep a score threshold on **val** and report the best-F1 threshold, to show a precision/recall trade-off that the threshold-0 protocol hides.

Expected effect is unknown. Our global MRR (0.63 for A19 on test) is a different quantity from F1, so none of the Sprint 5–6 numbers can be compared with BioGITOM's 0.92 as they stand.
