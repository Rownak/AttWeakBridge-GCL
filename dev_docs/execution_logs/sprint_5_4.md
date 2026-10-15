# Sprint 5, Phase 5.4: Final test evaluation, λ curve, and stratified analysis

Plan: [sprint_5.md](../sprints/sprint_5.md). Previous phases: [5.1](sprint_5_1.md), [5.2](sprint_5_2.md),
[5.3](sprint_5_3.md). Lineage `ncit_doid_srga`; test = 3,280 gold pairs (unchanged since Sprint 4), val = 211.

## 0. Sprint 5 summary

- **SapBERT is beaten on held-out test pairs, in both modes.** The selected variant, A3, adds a trained 263k-parameter
  MLP correction beside frozen SapBERT. Test MRR is 0.4957 ± 0.0024 global (A0 0.4475, +0.048) and 0.5687 ± 0.0022
  candidates (A0 0.5193, +0.049), over 3 seeds.
- **The selection rule held, and it picked the weaker of two close variants.**
  - Gate 5.3 kept A3 because, on 211 val pairs, A4 (the full relational graph adapter) was not better (0.4791 vs 0.4784).
  - On the 3,280 test pairs, the A4 reference is higher: 0.5084 ± 0.0104 global and 0.5949 ± 0.0070 candidates.
  - It also fixes twice as many queries whose SapBERT confuser is a near neighbour of the gold target.
  - Per the test discipline, A4 is **not** retroactively selected. The result is recorded as evidence that val was too
    small to resolve the structural gain (section 4).
- **The λ floor works** (A9): λ = 0 reproduces SapBERT exactly, and the curve peaks above it.

## 1. What was done

| task | how |
|---|---|
| Selected variant | Per gate 5.3 (failed), the best of A1/A3 on val is **A3**. The runner-up is **A1**. **A4** was added as a declared structural reference *before* any test number was read (not a selection candidate). |
| Test readout (once) | Added A3 (3 seeds) and A4 (3 seeds) to [ncit_doid_srga_compare.yaml](../../configs/experiments/ncit_doid_srga_compare.yaml), which already held A0, BERT-base, the λ=0 floor and A1. Evaluate run `20261015T195321Z-1eb91f39` (`gold_val` + `gold_test`). |
| A9 λ curve | Retrained A3 (seed 42, same τ/epochs) at λ ∈ {0, 0.1, 0.5, 1} (λ = 0.25 is the existing seed-42 run); val only, [ncit_doid_srga_lambda.yaml](../../configs/experiments/ncit_doid_srga_lambda.yaml), evaluate run `20261015T195804Z-a89add3c`. |
| Stratified report | New [scripts/srga_stratified.py](../../scripts/srga_stratified.py) (analysis only, no pipeline change). [scripts/srga_paired_ranks.py](../../scripts/srga_paired_ranks.py) now also returns each method's top confuser. Both reuse the evaluator's query builder and ranking, and their MRRs equal `metrics.parquet` (A0 0.4475, A3 seed 42 0.4930, A4 seed 42 0.5159). |
| Plot | [configs/plots/ncit_doid_srga_compare.yaml](../../configs/plots/ncit_doid_srga_compare.yaml) (`bar`: test Hits@K, test MRR, val MRR; faceted by mode). Plot run `20261015T195637Z-4a7356a1`, pinned to the test-readout evaluate run. |
| README | New section "SapBERT-residual graph adapter (`ncit_doid_srga`)" before "Configuration"; existing sections unchanged. |

No source code in `src/` changed in this phase.

## 2. Model architecture and equations (selected variant A3, reference A4)

Both export `z_n = [ x̂_n ; sqrt(λ) · h_n / ||h_n|| ]`, where `x̂` is frozen SapBERT (unit, 768-d). Scoring is cosine
on `z`, which ranks by `x̂_a·x̂_d + λ·ĥ_a·ĥ_d` up to per-node norms. λ = 0.25.

```
A3 (no graph):    h_n = Linear(256,256)( Dropout( LayerNorm( ReLU( Linear(768,256)(x̂_n) ) ) ) )         263,168 params
A4 (relational):  a_n = [x̂_n ; p_n ; c_n ; x̂_n ⊙ p_n ; log(1+#children) ; log(1+#parents)]
                  u_n = W_in a_n ;  r_n = W_par·mean_parents(u) + W_chi·mean_children(u)
                  h_n = MLP_2( LayerNorm(u_n + r_n) )                                                     1,050,368 params
```

Loss for both (Phase 5.2), over 1,174 train pairs `(a, p)`, with τ = 0.05:

```
L = −mean log softmax_{d ∈ DOID_trainable \ other gold of a}( cos(z_a, z_d)/τ )[p]
    − 0.5 · mean log softmax_{n ∈ NCIT_trainable \ other gold of p}( cos(z_p, z_n)/τ )[a]
```

Training: Adam (lr 1e-3, weight decay 1e-4), 50 epochs, dropout 0.2, seeds 42/43/44.

## 3. Results

### 3.1 Test (3,280 pairs, the sprint's one test readout)

Mean ± sample std over 3 seeds where trained. Evaluate run `20261015T195321Z-1eb91f39`.

**global**

| method | role | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|---|
| A0 SapBERT | bar | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| λ = 0 floor | check | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| A1 nb, λ=0.25 dir | runner-up | 0.3817 | 0.6235 | 0.7122 | 0.4922 |
| **A3 MLP** | **selected** | **0.3784 ± 0.0035** | **0.6359 ± 0.0020** | **0.7295 ± 0.0069** | **0.4957 ± 0.0024** |
| A4 relational | reference | 0.3904 ± 0.0129 | 0.6511 ± 0.0097 | 0.7386 ± 0.0109 | 0.5084 ± 0.0104 |
| BERT-base | text control | 0.0116 | 0.0229 | 0.0308 | 0.0186 |

**candidates** (101 per pair)

| method | role | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|---|
| A0 SapBERT | bar | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| λ = 0 floor | check | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| A1 nb, λ=0.25 dir | runner-up | 0.4277 | 0.7052 | 0.8210 | 0.5557 |
| **A3 MLP** | **selected** | **0.4325 ± 0.0034** | **0.7355 ± 0.0034** | **0.8450 ± 0.0015** | **0.5687 ± 0.0022** |
| A4 relational | reference | 0.4589 ± 0.0096 | 0.7617 ± 0.0044 | 0.8714 ± 0.0026 | 0.5949 ± 0.0070 |
| BERT-base | text control | 0.0229 | 0.0628 | 0.1006 | 0.0616 |

Per-seed test MRR (global / candidates):
- A3: 0.4930 / 0.5667, 0.4965 / 0.5684, 0.4976 / 0.5710.
- A4: 0.5159 / 0.6000, 0.4966 / 0.5869, 0.5128 / 0.5978.

Paired, global, ranks of the gold target (same seed):

| comparison | wins | losses | ties |
|---|---|---|---|
| A1 vs A0 | 1480 | 521 | 1279 |
| A3 vs A0 (mean ± std) | 1495.3 ± 22.1 | 556.7 ± 30.2 | 1228 |
| A4 vs A0 (mean ± std) | 1589.0 ± 35.6 | 695.3 ± 70.0 | 996 |
| A3 vs A4 (seeds 42 / 43 / 44) | 880 / 988 / 935 | 1267 / 1215 / 1238 | 1133 / 1077 / 1107 |

The val numbers in the same run equal Phases 5.1–5.3 exactly (A0 0.4125, A1 0.4555, A3 0.4784 ± 0.0042,
A4 0.4791 ± 0.0183), so the test readout uses the same models that were selected on val.

### 3.2 A9 λ curve (A3, seed 42, val, global)

| λ | 0 | 0.1 | 0.25 | 0.5 | 1 |
|---|---|---|---|---|---|
| val MRR | 0.4125 (= A0) | 0.4698 | 0.4756 | **0.4846** | 0.4840 |
| val Hits@1 | 0.2938 (= A0) | 0.3602 | 0.3460 | 0.3697 | 0.3697 |
| val Hits@10 | 0.6398 (= A0) | 0.7062 | 0.7393 | 0.7393 | 0.7062 |

- **λ = 0 equals A0** in every metric. The floor holds for the trained adapter too: with λ = 0, `z` does not depend on
  the adapter.
- **The curve peaks above 0** and is flat from 0.25 to 1. MRR spans 0.4756–0.4846, a 0.009 range, about 2 seed stds of A3.
- λ = 0.25 was fixed in 5.2 from the training-free 5.1 grid and was not re-tuned for the trained adapter. λ = 0.5 looks
  marginally better on val. That difference is within noise, and it was not carried to test.

Model runs, `srga-mlp-l<λ>-s42` (dated 20261015): λ 0 `T195700Z-67682566`, 0.1 `T195715Z-ec44e36b`, 0.25
`T165407Z-7619df52`, 0.5 `T195732Z-8154298f`, 1 `T195747Z-973caa9f`.

### 3.3 Stratified win/loss on test (global), vs A0

The strata are defined by two questions:
- whether SapBERT ranks the gold target first ("right": 1,123 queries; "wrong": 2,157);
- whether SapBERT's top confuser (its best-ranked non-target candidate) is within 2 hops of the gold target in DOID
  (undirected ingest edges): "near" 1,343, "far" 1,937.

The counts are defined at rank 1:
- **fixed:** A0 rank > 1 and variant rank = 1;
- **broken:** A0 rank = 1 and variant rank > 1;
- "rank better/worse" compares the full ranks.

All counts are the mean ± std over 3 seeds.

**A3 (selected)**

| stratum | n | fixed | broken | unchanged | rank better | rank worse |
|---|---|---|---|---|---|---|
| SapBERT right, confuser near | 442 | 0 | 57.7 ± 4.0 | 384.3 | 0 | 57.7 |
| SapBERT right, confuser far | 681 | 0 | 79.3 ± 0.6 | 601.7 | 0 | 79.3 |
| SapBERT wrong, confuser near | 901 | 84.0 ± 7.5 | 0 | 817.0 | 599.0 ± 13.5 | 169.7 ± 15.0 |
| SapBERT wrong, confuser far | 1256 | 171.0 ± 7.2 | 0 | 1085.0 | 896.3 ± 9.9 | 250.0 ± 11.8 |
| all | 3280 | 255.0 ± 12.3 | 137.0 ± 3.6 | 2888.0 | 1495.3 ± 22.1 | 556.7 ± 30.2 |

**A4 (reference)**

| stratum | n | fixed | broken | unchanged | rank better | rank worse |
|---|---|---|---|---|---|---|
| SapBERT right, confuser near | 442 | 0 | 84.7 ± 9.1 | 357.3 | 0 | 84.7 |
| SapBERT right, confuser far | 681 | 0 | 174.3 ± 21.9 | 506.7 | 0 | 174.3 |
| SapBERT wrong, confuser near | 901 | 171.3 ± 4.0 | 0 | 729.7 | 670.3 ± 15.1 | 157.3 ± 12.3 |
| SapBERT wrong, confuser far | 1256 | 245.3 ± 11.5 | 0 | 1010.7 | 918.7 ± 21.0 | 279.0 ± 27.7 |
| all | 3280 | 416.7 ± 15.2 | 259.0 ± 31.0 | 2604.3 | 1589.0 ± 35.6 | 695.3 ± 70.0 |

What the strata show:
- **A3 fixes 255 and breaks 137 at rank 1** (net +118, which is the Hits@1 gain 0.3424 → 0.3784). It breaks 12% of the
  queries SapBERT got right.
- **A3's fixes do not target structural confusers.** It fixes 9.3% of "wrong, near" and 13.6% of "wrong, far". The fixes
  are not concentrated where the confuser is a DOID neighbour of the target, as expected for a model with no graph.
- **A4 fixes "wrong, near" queries twice as often as A3** (171 vs 84, 19.0% vs 9.3%). Where the confuser is far, its gain
  over A3 is smaller (245 vs 171). This is the pattern the design expected from structure: A4 separates a gold target
  from its own parent/child/sibling confusers.
- **A4 also breaks more:** 259 (23% of "SapBERT right") vs A3's 137, mostly in the "right, far" stratum (174). Its net
  rank-1 gain is +158 vs A3's +118, and its rank-better : rank-worse ratio is 2.3 vs A3's 2.7.

## 4. Acceptance and decision

| acceptance item | result |
|---|---|
| Test tables (Hits@1/5/10, MRR; global and candidates; mean ± std) | done (3.1) |
| Selected variant's test MRR > A0 in both modes | **yes**: 0.4957 ± 0.0024 > 0.4475 (global); 0.5687 ± 0.0022 > 0.5193 (candidates). All 3 seeds are above A0. |
| Fixes concentrated in "SapBERT wrong", few breaks in "SapBERT right" | **partly.** By definition, fixes can only occur in "wrong". Breaks are not few: 137 (12% of "right"), against 255 fixes (ratio 1.9 : 1). |
| λ curve peaks above λ = 0, and λ = 0 equals A0 | **yes** (3.2) |

**Sprint 5 result:**
- **A3**, a SapBERT-residual adapter trained with full-softmax InfoNCE and no graph input, beats SapBERT on held-out
  NCIT–DOID test pairs: +0.048 MRR global and +0.049 candidates over 3 seeds, with 2.7 per-query rank improvements per
  rank loss.
- It also edges out the training-free structural baseline A1 (+0.004 global, +0.013 candidates).

**On structure, the evidence is mixed and stated plainly:**
- On val (211 pairs), the graph adapter A4 showed no gain over A3, and gate 5.3 stopped there.
- On test (3,280 pairs), A4 is ahead of A3 by +0.013 MRR (global) and +0.026 (candidates). It wins the head-to-head on all
  three seeds and doubles the fixes on structurally near confusers.
- A4 was not selected, so this is a post-hoc observation, not the sprint's result. The likely reason it was not selected
  is that 211 val pairs cannot resolve a gain of about 0.01–0.02 MRR (A4's val seed std was 0.018).

## 5. Limitations and suggestions for a next sprint (not done)

- **The val split is too small for selection at this effect size.** A next sprint could use a larger val, for example
  30% of train sources, or k-fold over train sources, and re-run the A3 vs A4 decision without touching test again.
- **A4 is unstable across seeds** (test std 0.010 vs A3's 0.002) and breaks twice as many SapBERT-correct queries.
  Candidates: early stopping on val, more dropout, or a smaller `W_in`.
- **λ was never tuned for the trained adapters** (the A9 curve is flat from 0.25 to 1).
- Not attempted (out of scope per the plan): per-node confidence gate (A8), DeepWalk input (A7), Variant B, the anchor
  re-ranker (design §8), and A2 score fusion.
- Training is transductive: test nodes take part in message passing as features only, never in a loss term.
- Runs are not bit-reproducible on this machine (Phase 5.2). `ruff` is not installed, so lint was not run. Unit tests:
  439 pass (no `src/` change in this phase).

## 6. Files

- Modified: `configs/experiments/ncit_doid_srga_compare.yaml` (A3, A4 added), `scripts/srga_paired_ranks.py` (returns top
  confusers), `README.md` (new SRGA section), `dev_docs/sprints/sprint_5.md` (Phase 5.4 boxes ticked).
- Created: `configs/experiments/ncit_doid_srga_lambda.yaml`, `configs/plots/ncit_doid_srga_compare.yaml`,
  `scripts/srga_stratified.py`, this log.
- Untracked artifacts:
  - evaluate runs `20261015T195321Z-1eb91f39` (test readout) and `20261015T195804Z-a89add3c` (λ curve);
  - plot run `20261015T195637Z-4a7356a1` (`artifacts/plots/ncit_doid_srga/.../figures/{test_hits_at_k,test_mrr,val_mrr}.png`);
  - 4 new `srga-mlp-l<λ>-s42` models.

## 7. Commands

```
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_compare
uv run cga run plot --dataset ncit_doid_srga --config configs/plots/ncit_doid_srga_compare.yaml --set plot.inputs.evaluate_run=20261015T195321Z-1eb91f39

# λ curve (λ in {0, 0.1, 0.5, 1}; 0.25 = srga-mlp-l0.25-s42)
uv run cga run gcl_train --dataset ncit_doid_srga --set seed=42 --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=mlp --set gcl_train.model.lambda=0.5 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.model.model_id=srga-mlp-l0.5-s42
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_lambda

uv run python scripts/srga_paired_ranks.py --dataset ncit_doid_srga --split test --group A1=srga-nb-l0.25-dir --group A3="srga-mlp-l0.25-s{seed}" --group A4="srga-rel-l0.25-s{seed}"
uv run python scripts/srga_stratified.py --dataset ncit_doid_srga --split test --group A3="srga-mlp-l0.25-s{seed}" --group A4="srga-rel-l0.25-s{seed}"
```
