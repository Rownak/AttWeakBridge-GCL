# Sprint 8: BioGITOM-comparable precision, recall and F1 for the NCIT–DOID variants

## Objective

BioGITOM reports P/R/F1 of a global one-to-one assignment on NCIT–DOID, not MRR. Our Sprint 4 to 6 results are ranking metrics, so none of them can be
compared with BioGITOM as they stand. This sprint adds a P/R/F1 evaluator that follows BioGITOM's protocol and runs it over the existing models.
The test split is the same (`refs_equiv/test.tsv`, 3,280 pairs). No model is trained and no earlier number changes.

Analysis and risks: [biogitom_vs_srga_prf_comparability.md](../data_analysis/biogitom_vs_srga_prf_comparability.md).
Models and numbering (A0 to A25) are those of Sprints 5 and 6: [sprint_5.md](sprint_5.md), [sprint_6.md](sprint_6.md).

## Conventions for every phase

- Google-style docstrings, tests in `tests/unit/`, `uv run pytest tests/unit -q` green.
- Each phase ends with `dev_docs/execution_logs/sprint_8_N.md`: method, results tables with run ids, decision and caveats.
- Commit prefix `[Sprint-8.N]`.
- Every new option is default-off, so earlier runs and configs are unchanged.

## Decision gates (summary)

| After | Go if | Else |
|---|---|---|
| 8.1 | Tests green and the A0 val smoke run reproduces the Sprint 5 Hits@1 | Fix the evaluator before 8.2. |
| 8.2 | None (report only) | Check A0 under the same code before reading any gap to BioGITOM as a method gap. |

---

## Phase 8.1: BioGITOM-style P/R/F1 evaluator

**Why.** BioGITOM reports P/R/F1 of a global one-to-one assignment, not MRR. The test split is the same (`test.tsv`, 3,280 pairs),
the protocol is not. Analysis: [biogitom_vs_srga_prf_comparability.md](../data_analysis/biogitom_vs_srga_prf_comparability.md).

**Capability:** a new evaluator kind `gold_alignment_prf` that turns any method's scores into a predicted pair set and scores it against
the gold pairs. `gold_alignment` is unchanged. Every method kind (embedding store, `gcl_run`, `fusion`) works, because the evaluator only
reads the aligned matrix.

**Protocol (mirrors BioGITOM `ncit2doid.py`, `evaluate_predictions`).**

```
1. Scores cos(z_a, z_d) for every source a of the split against every target d (global search; test.cands.tsv is not used).
2. Top-k per source (k = 3), sorted by score.
3. Optional filters (`biogitom_filters`): (a) drop a prediction whose source or target is in train.tsv but not in the scored split's gold file;
   (b) keep only sources that occur in the scored split.
4. Greedy one-to-one: sort all rows by score; keep a pair if neither endpoint is used and score >= threshold (default 0.0).
5. P = correct / |predicted|, R = correct / |gold|, F1 = 2PR / (P + R); correct = exact gold pair.
6. @1 variant: top-1 target per source, no assignment (BioGITOM's `evaluate_topk`).
```

- [ ] `evaluate/evaluators/gold_alignment_prf.py` (new, registered in `evaluators/__init__.py`). Options: `split` (`val`|`test`), `top_k` (3),
      `threshold` (0.0), `biogitom_filters` (bool), `assignment` (`greedy_one_to_one`|`top1`).
  - Ties break by node id. Gold pairs with an endpoint missing from a method count as unpredicted, so recall cannot be inflated.
  - For `val`, "train" in filter (a) means the train pairs left after the val carve.
  - Emits `precision`, `recall`, `f1`, `n_predicted`, `n_correct`, `n_gold`, `n_unique_sources`, `n_unique_targets` (the recall ceiling of
    a one-to-one assignment when a source has several gold targets).
- [ ] Tests (`tests/unit/pipelines/evaluate/test_gold_alignment_prf.py`):
  - a hand-built 4×4 score matrix with known P/R/F1 for `greedy_one_to_one`, `top1`, each filter and a threshold;
  - a source with two gold targets caps recall below 1 under one-to-one but not under `top1`;
  - missing endpoints count as unpredicted gold; ties break by node id;
  - a `fusion` method gives the same numbers as its explicit weighted-mean score matrix.
- [ ] Smoke run on **val with A0 only** (`configs/experiments/ncit_doid_srga_prf_smoke.yaml`): counts are consistent (`n_gold` = 211) and
      `top1` F1 is close to the A0 Hits@1 from Sprint 5 (0.2938 when no filter is applied).
- [ ] README: add the evaluator and its command to the SRGA section.

**Training objective / negatives:** none (no training).
**Acceptance:** tests green; the smoke run reproduces the A0 sanity checks above.
**Decision:** none (infrastructure).

---

## Phase 8.2: P/R/F1 table for our variants next to BioGITOM

**Capability:** run the 8.1 evaluator over the existing models and produce the comparison. No model is trained and no earlier number changes.

- [ ] Val (`configs/experiments/ncit_doid_srga_prf.yaml`): A0, A1, A3, A4, A11, A12, A14, A15, A17–A20; filters on and off,
      `greedy_one_to_one` and `top1`.
- [ ] Threshold sweep, val only (`scripts/srga_prf_threshold.py`): score threshold over a fixed grid for A19 and A12, P/R curve and best-F1
      threshold. Threshold 0.0 stays the comparable headline; the swept value is a separate, labelled row, applied once to test.
- [ ] Test, once (`configs/experiments/ncit_doid_srga_prf_test.yaml`), filters on and off:
  - train-only lineage `ncit_doid_srga`: A0, A1, A3, A4, A11, A12, A14, A15, A17, A18, A19 (trained models: mean ± std over seeds);
  - full-data lineage `ncit_doid_srga_full` (the same 1,406 training pairs as BioGITOM): A25 for A3, A11, A12 and A19.
- [ ] Execution log `sprint_8_2.md` with the "comparable to BioGITOM" rows: full-data lineage, filters on, threshold 0.0,
      `greedy_one_to_one`, next to BioGITOM's logged P 0.924 / R 0.911 / F1 0.917 (2,989 / 3,236 / 3,280), naming the log cited. The unfiltered rows
      sit beside them so the reader sees how much of any gap comes from the train-label filter.

**Ablation:**
- filters on vs off (A12 is expected to gain most, since it memorises its train targets);
- `greedy_one_to_one` vs `top1`;
- threshold 0.0 vs the val-chosen threshold.

**Acceptance:**
- a val table and a test table (P, R, F1, `@1` variant, `n_predicted`, `n_correct`, recall ceiling);
- the BioGITOM comparison rows as above;
- the caveats of the analysis note restated: BioGITOM's test-scored experiment logs, the unverified DeepOnto formula, and the train-label filter.

**Decision:** report only (no gate). If every method is far below BioGITOM's F1, check A0 under the same code before calling it a method gap.

---

