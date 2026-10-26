# BioGITOM on NCIT–DOID: how P / R / F1 are produced

Sources (all under `benchmark/biogitom/`): `Docs/BioGITOM_Usage.md`, `run_biogitom.py`,
`Tasks/ncit2doid/ncit2doid.py` (the executed task script), `Tasks/ncit2doid/Results_log_ncit2doid.md`,
`Scripts/Evaluation/evaluate_global_metrics.py`, `Scripts/Preprocessing/train_generation.py`,
`Experiments/ncit2doid/**/Results_log_*.md`. Line numbers refer to `ncit2doid.py`.

## 1. Files used

| Role | File | Where |
|---|---|---|
| Ontologies | `ncit.owl`, `doid.owl` in `Datasets/ncit2doid/` | l.159, 162 (`Ontology(...)`) |
| Training signal | `refs_equiv/train.tsv` → positives, plus sampled negatives, saved as `{task}_train.csv` | l.201–202, 1243–1270, 1305 |
| Gold standard | `refs_equiv/test.tsv` (3,280 mappings) | l.205, 1568, `evaluate_predictions` |
| Ranking candidates | `refs_equiv/test.cands.tsv` | l.208, 1663; used only for MRR/Hits@k, **not** for P/R/F1 |
| Predictions | `Tasks/ncit2doid/Results/ncit2doid_top_{k}_mappings.tsv` (`k = 3`, l.134), then `ncit2doid_matching_results.tsv` | l.218, 224 |

`Usage.md` documents the same outputs: `{task}_all-predictions` (all candidates with scores) and
`{task}_matching_results` (filtered), scored by `Scripts/Evaluation/evaluate_global_metrics.py`
(Usage.md, "Outputs and Results Visualization").

## 2. Split used

A **fixed train/test split from the dataset**: `refs_equiv/train.tsv` for fitting, `refs_equiv/test.tsv`
for scoring. The full `full.tsv` is never read. Details:

- The GIT encoder is trained on `train.tsv` positives plus random negatives (l.1259–1270, 1000 epochs).
  The gated network then trains on the same pairs with an internal 70/30 `train_test_split`
  (`test_size=0.3`, l.1313). That "validation" set is a split of **training pairs**, not of test data.
- Both ontologies are embedded and passed through the GNN **in full**, including test entities
  (transductive; the test pairs' labels are never used, but their nodes and edges are present).
- Scoring is restricted to test-relevant entities by two filters (docstring l.822–876,
  code l.937–945): (1) drop predictions whose source or target appears in `train.tsv` but not in
  `test.tsv`; (2) keep only predictions whose `SrcEntity` is in `test.tsv`.
  Example from the log (`Similarity Metrics/Results_log_ncit2doid_Similarity.md`, `top_k=10`):
  70,650 rows → 45,915 after removing train-only URIs → 26,463 after keeping test sources.

## 3. How predictions are generated and filtered

1. Final embeddings for every concept are saved, then `filter_ignored_class` (l.632) removes classes that
   DeepOnto's `get_ignored_class_index` marks as ignored (e.g. `owl:Thing`, deprecated).
2. `topk_faiss_l2` (l.781): exact `IndexFlatL2` search of **every source** against **all target** embeddings
   (a global search, not the `test.cands` pools). Score = `1 / (1 + L2 distance)`; top-3 per source.
3. `evaluate_predictions` (l.913): apply the two filters above, sort all remaining rows by score, and take a
   **greedy one-to-one assignment**: a pair is kept if neither its source nor its target is already used and
   `score >= threshold` (threshold defaults to `0.0`, so nothing is rejected on score; l.955–963).
4. The kept pairs are written to `ncit2doid_matching_results.tsv` and scored.

## 4. Metric computation

- Pairs are `(SrcEntity, TgtEntity)` tuples. A prediction is **correct only if the exact pair is in
  `test.tsv`** (l.974–978). `correct = |preds ∩ refs|`.
- P / R / F1 come from DeepOnto `AlignmentEvaluator.f1(preds, refs)` (l.980). That library is **not
  vendored in this repo**, so its exact formula cannot be inspected here. The logs are consistent with the
  standard definitions: P = correct / |preds|, R = correct / |refs|, F1 = harmonic mean.
- `evaluate_global_metrics.py` (the CLI in Usage.md) differs slightly: it scores `matching_results.tsv`
  directly after `remove_ignored_mappings`, and does **not** apply the train-only/test-source filters. The
  logged numbers come from the in-notebook `evaluate_predictions`.
- A separate "@1" metric exists (`evaluate_topk`, l.999, and `evaluate_top1_metrics` in the script): top-1
  per source, matched by source → target dict.

## 5. Does this plausibly give P 0.924 / R 0.911 / F1 0.918?

**P and R: yes, reproduced in the logs; F1: 0.917, not 0.918.**

- `Experiments/ncit2doid/Negative_Number_Ajdustment/Results_log_ncit2doid_train_100.md` (l.2468–2474) and
  `Experiments/ncit2doid/Similarity Metrics/Results_log_ncit2doid_Similarity.md` (l.3291): correct = 2,989,
  predictions = 3,236, references = 3,280 → P = 0.924, R = 0.911, **F1 = 0.917**. Recomputing from those counts by
  hand gives P = 0.9237, R = 0.9113, F1 ≈ 0.9175, so the repo's number is correct to its rounding.
- The "main" task log, `Tasks/ncit2doid/Results_log_ncit2doid.md` (l.2386–2391, `k=3`), gives
  2,973 / 3,193 / 3,280 → **P 0.931, R 0.906, F1 0.919**, i.e. different from the quoted figures.
- `Similarity Metrics` log, "@1" evaluation: P@1 0.9237, R@1 0.9138, F1@1 0.9187 (0.919).
- I found no occurrence of F1 = 0.918 anywhere in the repo. The quoted 0.918 may come from the paper or
  another run; that cannot be verified here.

The magnitude is plausible for this protocol, for three reasons, and some of it flatters the result:

1. **Global search, but only test sources scored.** Each test source is ranked against all ~8k DOID classes, which
   is harder than the `test.cands` pools (the DOID class count after filtering is not stated in the logs I read; the
   raw file has 8,465), so ~91% is a real result for that setup.
2. **Precision denominator is shaped by filtering.** Predictions are restricted to the 3,280-source test set and
   to a one-to-one assignment, so there are only 3,236 predictions for 3,280 references; this keeps P and R
   close to each other and both high. It is a legitimate Bio-ML convention (`test.tsv` is the reference), but
   it is not a "predict over the whole ontologies, score against `full.tsv`" protocol.
3. **Transductive leakage is limited to features, not labels.** Test nodes are in the training graph, with
   no label use. I found nothing that feeds `test.tsv` into training.

## 6. Not verifiable from the repository

- `Datasets/` is absent (downloaded by `download_data.py`), and so are `Tasks/ncit2doid/Data` and `Results`.
  The pipeline cannot be re-run here, so all numbers above come from saved logs, which were produced on Colab
  (`/content/gdrive/...`).
- `AlignmentEvaluator.f1`, `get_ignored_class_index`, and `EntityMapping/ReferenceMapping` are DeepOnto code,
  not in the repo.
- Which log run the quoted 0.924/0.911/0.918 came from; the hyperparameters differ between logs
  (negatives per pair: 50 vs 100; `k` = 3 vs 10).
- Random seeds are set (`random.seed(42)`, `train_test_split(random_state=42)`), but `Scripts/Preprocessing/train_generation.py`
  (negative sampling via `np.random.choice`) contains no seed, so exact reproduction is uncertain.
