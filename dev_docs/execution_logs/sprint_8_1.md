# Sprint 8, Phase 8.1: BioGITOM-style P/R/F1 evaluator

Plan: [sprint_8.md](../sprints/sprint_8.md). Analysis: [biogitom_vs_srga_prf_comparability.md](../data_analysis/biogitom_vs_srga_prf_comparability.md). No model was trained and no earlier number changed.

## 0. Outcome in brief

- **Gate 8.1: GO.** Tests are green (496 pass, 16 new) and the A0 val smoke run reproduces the Sprint 5 numbers: `n_gold` = 211 and unfiltered `top1` recall = **0.2938** (62 / 211), exactly the Sprint 5 A0 val Hits@1. `top1` F1 is 0.2945.
- New evaluator kind `gold_alignment_prf`; every option is default-off, so no earlier run or config changes.

## 1. What was implemented

- [gold_alignment_prf.py](../../src/cga/pipelines/evaluate/evaluators/gold_alignment_prf.py) (registered in `evaluators/__init__.py`). Options: `split` (`val`|`test`), `top_k` (3), `threshold` (0.0), `biogitom_filters` (false), `assignment` (`greedy_one_to_one`|`top1`).
- Steps, as module-level helpers so scripts reuse them: `top_k_candidates` (cosine, stable sort over id-sorted targets, so ties break by node id), `apply_filters`, `assign_pairs` (sorted by `(-score, source, target)`, so the result does not depend on row order), `score_pairs`.
- Emits `precision`, `recall`, `f1`, `n_predicted`, `n_correct`, `n_gold`, `n_unique_sources`, `n_unique_targets` per method; `view_id` = the assignment; the options go into the scope provenance.
- Decisions:
  - **`n_gold` counts every de-duplicated gold pair of the split**, including pairs whose endpoint a method lacks, so recall cannot be inflated. The target pool is each method's own `graph_2` nodes (all methods share the same nodes in practice).
  - **Filter (a)**: "train-only" = nodes of the `train` pairs minus nodes of the scored split's gold pairs. For `val` the train pairs are those left after the val carve (they are the `train` rows of the ingest). Filter (b) (keep only sources of the split) is a no-op by construction, because only sources of the split are scored; it is kept for fidelity.
  - **`top1` with filters**: filters apply to the top-3 rows first, then the best remaining row per source is kept. So a source whose best target is train-only falls back to its next row, as in the greedy path.
  - **Score is cosine**, as in the plan. A negative cosine is below the default threshold 0.0 and is never predicted; this only matters for a source whose top-3 are all negative (not seen with these models: unfiltered `top1` predicts every source).
- README: SRGA section lists the evaluator, configs and scripts.

## 2. Tests

[test_gold_alignment_prf.py](../../tests/unit/pipelines/evaluate/test_gold_alignment_prf.py), a hand-built 4 × 4 fixture with known values: greedy (P=R=F1=0.5), top1 (0.75), threshold (P 2/3, R 0.5), filters (a train-only target flips top1 from 2 to 3 correct), `apply_filters` on a foreign / train-only source, shared target and source with two targets (recall ceiling), missing endpoint counts as unpredicted gold, id tie-break, row-order independence, empty predictions, a `fusion` method equals its explicit weighted-mean score matrix, `val` split, invalid options, provenance.

**Deviation from the plan text:** the plan says a source with two gold targets caps recall under one-to-one "but not under `top1`". With `top1` defined as one target per source (BioGITOM's `evaluate_topk`) that source is capped under both. What differs is a **target shared by two sources**: greedy keeps one of them, `top1` keeps both. The tests assert both cases.

## 3. Smoke run

`configs/experiments/ncit_doid_srga_prf_smoke.yaml`, A0 only, val, evaluate run `20261025T213624Z-4ecb7f3e`:

| evaluation | P | R | F1 | n_predicted | n_correct | n_gold |
|---|---|---|---|---|---|---|
| top1, no filter | 0.2952 | **0.2938** | 0.2945 | 210 | 62 | 211 |
| top1, filters | 0.3173 | 0.3128 | 0.3150 | 208 | 66 | 211 |
| greedy, no filter | 0.3143 | 0.3128 | 0.3135 | 210 | 66 | 211 |
| greedy, filters | 0.3350 | 0.3270 | 0.3309 | 206 | 69 | 211 |

`n_predicted` is 210, not 211, because the 211 val pairs have 210 unique sources (one source has two gold targets), so at most 210 predictions are made and the recall ceiling is 210 / 211 = 0.9953. With the filters `n_predicted` drops further (206 to 208) when every kept row of a source was train-only.

## 4. Files

- Created: `src/cga/pipelines/evaluate/evaluators/gold_alignment_prf.py`, `tests/unit/pipelines/evaluate/test_gold_alignment_prf.py`, `configs/experiments/ncit_doid_srga_prf_smoke.yaml`, this log.
- Modified: `src/cga/pipelines/evaluate/evaluators/__init__.py`, `README.md`, `dev_docs/sprints/sprint_8.md`.

## 5. Commands

```
uv run pytest tests/unit -q
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_prf_smoke
```
