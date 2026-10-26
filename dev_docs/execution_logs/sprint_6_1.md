# Sprint 6, Phase 6.1: Score-level fusion of A11 and A12, and seed ensembles (A17–A20)

Plan: [sprint_6.md](../sprints/sprint_6.md). Background: [sprint_5_6.md](sprint_5_6.md). Lineage `ncit_doid_srga`; val = 211 pairs, test = 3,280. **No new training**: every member is a Sprint 5 model (seeds 42/43/44).

## 0. Outcome in brief

- **Gate 6.1: GO.** The selected fusion **A19 (w = 0.5)** beats the best single member on val (0.6278 vs A12 0.5169) and its own seed ensemble A18 (0.5730; paired CI of the difference [+0.025, +0.089], excludes 0).
- **Test (read once): A19 reaches global MRR 0.6293 / Hits@10 0.8777 and candidates MRR 0.6799 / Hits@10 0.9390**, against the A12 bars of 0.5361 / 0.7725 and 0.6209 / 0.9034. This is at or above the reference target (about 0.62 / 0.86) in both modes. The target's protocol is still unverified (see sprint_6.md), so treat it as a reference point.
- **The gain is mostly complementarity, plus some seed averaging.** Averaging three A12 seeds gives +0.063 test MRR (A18 vs the A12 mean); averaging three A11 seeds gives only +0.009 (A17 vs A11). Mixing A11 and A12 adds +0.030 over A18 and +0.097 over A17.
- **A20 (adding the GAT variants) does not help:** val 0.6172 vs A19 0.6278, CI [−0.028, +0.007] includes 0, so by the preference for fewer members A19 is kept.
- The fusion keeps part of A11's safety: it breaks 151 of 1,123 SapBERT-correct test queries (A12: 412, A18: 315, A11: 98, A17: 82).

## 1. Method and equations

A new `kind: fusion` for `MethodSpec` ([methods.py](../../src/cga/pipelines/evaluate/methods.py)). Each member's vectors are L2-normalised, scaled by √weight and concatenated:

```
z_n = [ √w_1 · ẑ_n^(1) ; √w_2 · ẑ_n^(2) ; ... ]        ẑ^(i) = z^(i) / ||z^(i)||
z_a · z_d = Σ_i w_i cos_i(a, d),   ||z_n||² = Σ_i w_i
cos(z_a, z_d) = Σ_i w_i cos_i(a, d) / Σ_i w_i           (weighted mean of the members' cosines)
```

The evaluator normalises vectors again, so the fused score is exactly that weighted mean. Members may have different widths (A11 exports 1,024-d, A12 256-d).

- A17 = 3 × A11 seeds, equal weights. A18 = 3 × A12 seeds. A20 = A11, A12, A14, A15 × 3 seeds (12 members).
- **A19(w) = w · A18 + (1 − w) · A17**, written as 6 members with weights w/3 (A12 seeds) and (1 − w)/3 (A11 seeds). w = 0 and w = 1 reproduce A17 and A18 (the grid's end points have the same val numbers as the A17 and A18 rows).
- Member hyperparameters are unchanged from Sprint 5. The only things chosen here are w (5-point grid) and which of A17–A20 to carry forward, both on val.

## 2. What was implemented

- [methods.py](../../src/cga/pipelines/evaluate/methods.py): `kind: fusion` with `members: [{kind, model_id | embedder_id, ref, weight}]`; `fuse_matrices()` does the arithmetic.
  - Zero-weight members are dropped at config time (never resolved). A negative weight, all-zero weights, no members, or a nested fusion raises `ConfigError`.
  - Members load through the existing path as `method:<id>:m<i>`, so each is recorded in `manifest.inputs`. `method_run_id` is the member run ids joined with `+`.
  - Members are restricted to the shared nodes; an empty intersection raises `ConfigError`.
  - Default-off: the `embedding_store` and `gcl_run` paths are unchanged.
- [scripts/srga_fusion_bootstrap.py](../../scripts/srga_fusion_bootstrap.py) (new): paired bootstrap over queries (1,000 resamples, 95% percentile CI, fixed seed) of the MRR difference, with wins/losses and the SapBERT-correct break count. A group of single models is represented by the per-query mean reciprocal rank over seeds, so its MRR equals the mean-over-seeds tables.
- [scripts/srga_paired_ranks.py](../../scripts/srga_paired_ranks.py): `add_fusions()` helper. `srga_stratified.py` and `srga_pool_diagnostic.py` gain `--fuse NAME=GROUP[:w],...` (and `--only` for the pool script) so they score a fusion as a one-entry group. The stratified script no longer produces a NaN std for a one-entry group (shown as 0.0).
- Configs: [ncit_doid_srga_fusion.yaml](../../configs/experiments/ncit_doid_srga_fusion.yaml) (val: A0, per-seed A11/A12/A14/A15, A17, A18, A20), [ncit_doid_srga_fusion_grid.yaml](../../configs/experiments/ncit_doid_srga_fusion_grid.yaml) (A19, five weights), [ncit_doid_srga_fusion_test.yaml](../../configs/experiments/ncit_doid_srga_fusion_test.yaml) (test: A0, A17, A18, A19 at w = 0.5; both modes).
- Tests (460 pass; 6 new in [test_methods.py](../../tests/unit/pipelines/evaluate/test_methods.py)):
  - the fused cosine equals the weighted mean on a 3-member fixture with different widths and scales, and ‖z‖² = Σw;
  - a single member equals itself;
  - a zero weight drops a member (an unresolvable one is never loaded);
  - zero-sum, negative, empty and nested configs raise;
  - mismatched node sets use the intersection, and an empty intersection raises;
  - `method_run_id` joins the run ids and the member inputs are recorded.
  - The existing `embedding_store` and `gcl_run` tests pass unchanged.

## 3. Results

### 3.1 Val (211 pairs, global), evaluate runs `20261016T141009Z-d5e7ae49` (A17, A18, A20) and `20261016T141542Z-28788437` (A19 grid)

A11 and A12 are the mean over their 3 seeds.

| variant | what it is | MRR | Hits@1 | Hits@5 | Hits@10 | breaks of 62 SapBERT-correct |
|---|---|---|---|---|---|---|
| A0 | SapBERT | 0.4125 | 0.2938 | 0.5498 | 0.6398 | 0 |
| A11 | GCN + floor | 0.4924 | 0.3633 | 0.6509 | 0.7472 | 5.0 |
| A12 | GCN, no floor | 0.5169 | 0.3949 | 0.6651 | 0.7520 | 20.7 |
| A17 | seed ensemble of A11 | 0.5018 | 0.3697 | 0.6635 | 0.7488 | 4 |
| A18 | seed ensemble of A12 | 0.5730 | 0.4502 | 0.7346 | 0.8246 | 14 |
| **A19 (w = 0.5)** | **0.5 A18 + 0.5 A17** | **0.6278** | **0.5213** | 0.7630 | 0.8389 | 5 |
| A20 | A11 + A12 + A14 + A15 | 0.6172 | 0.4976 | 0.7630 | 0.8389 | 6 |

A19 weight curve (w = weight of A18):

| w | 0 (= A17) | 0.25 | **0.5** | 0.75 | 1 (= A18) |
|---|---|---|---|---|---|
| MRR | 0.5018 | 0.5911 | **0.6278** | 0.6088 | 0.5730 |
| Hits@1 | 0.3697 | 0.4692 | 0.5213 | 0.4976 | 0.4502 |
| Hits@10 | 0.7488 | 0.8436 | 0.8389 | 0.8294 | 0.8246 |

The curve is a single hump with its maximum in the middle of the grid, and both ends reproduce A17 and A18, so the optimum is not at an edge.

Paired bootstrap on val (MRR difference, 95% CI over queries; wins / losses of the first side):

| comparison | diff | CI | wins / losses |
|---|---|---|---|
| A17 vs A11 (seed averaging, floor) | +0.0095 | [−0.0009, +0.0206] includes 0 | 80 / 45 |
| A18 vs A12 (seed averaging, no floor) | +0.0561 | [+0.0358, +0.0778] excludes 0 | 106 / 46 |
| **A19 vs A18** | +0.0547 | [+0.0245, +0.0885] excludes 0 | 77 / 28 |
| A19 vs A17 | +0.1259 | [+0.0865, +0.1692] excludes 0 | 94 / 31 |
| A20 vs A19 | −0.0106 | [−0.0278, +0.0074] includes 0 | 40 / 39 |
| A20 vs A12 | +0.1003 | [+0.0700, +0.1366] excludes 0 | 121 / 37 |
| A19 vs A0 | +0.2153 | [+0.1702, +0.2680] excludes 0 | 127 / 18 |
| A19 vs A0 (A17, A18, A20) | +0.0894, +0.1606, +0.2047 | all exclude 0 | |
| A19 (w = 0.5) vs w = 0.25 | +0.0367 | [+0.0153, +0.0593] excludes 0 | 47 / 34 |
| A19 (w = 0.5) vs w = 0.75 | +0.0189 | [−0.0073, +0.0457] includes 0 | 54 / 28 |

w = 0.5 and w = 0.75 are within noise of each other; w = 0.5 is the grid maximum and is kept (the grid is coarse).

### 3.2 Selection

Best by val MRR: A19 (w = 0.5). A20 is within the CI of A19 and has twice the members, so A19 is preferred. **Selected: A19 (w = 0.5), 6 members.**

### 3.3 Test readout (3,280 pairs, once), evaluate run `20261016T141926Z-961181e2`

The A0, A11 and A12 rows are the existing Sprint 5 results (same models).

**global**

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| A11 (Sprint 5, 3-seed mean) | 0.3982 | 0.6732 | 0.7708 | 0.5234 |
| A12 (Sprint 5, 3-seed mean) | 0.4079 | 0.6952 | 0.7725 | 0.5361 |
| A17 | 0.4058 | 0.6802 | 0.7817 | 0.5325 |
| A18 | 0.4677 | 0.7631 | 0.8360 | 0.5990 |
| **A19 (w = 0.5)** | **0.4960** | **0.8006** | **0.8777** | **0.6293** |

**candidates** (101 per pair)

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| A11 (Sprint 5) | 0.4440 | 0.7613 | 0.8710 | 0.5852 |
| A12 (Sprint 5) | 0.4742 | 0.8103 | 0.9034 | 0.6209 |
| A17 | 0.4460 | 0.7680 | 0.8765 | 0.5892 |
| A18 | 0.5226 | 0.8500 | 0.9277 | 0.6649 |
| **A19 (w = 0.5)** | **0.5415** | **0.8622** | **0.9390** | **0.6799** |

Paired bootstrap on test (global MRR): A17 vs A11 +0.0091 [+0.0065, +0.0118]; A18 vs A12 +0.0630 [+0.0574, +0.0687]; A19 vs A18 +0.0302 [+0.0221, +0.0381]; A19 vs A17 +0.0968 [+0.0870, +0.1063]; A19 vs A0 +0.1817 [+0.1696, +0.1935]. All exclude 0. The CI is over queries only; it does not include the variation of training seeds, which is why the ensemble rows have no ± (one ensemble per variant).

### 3.4 Stratified fixed / broken on test (global, vs A0, rank-1 definition)

Strata: SapBERT right near 442 / far 681; wrong near 901 / far 1,256.

| variant | fixed | broken | fixed, wrong-near | fixed, wrong-far | broken, right-near | broken, right-far |
|---|---|---|---|---|---|---|
| A11 (5.5, mean) | 281.0 | 98.0 | 79.7 | 201.3 | 34.0 | 64.0 |
| A12 (5.5, mean) | 626.7 | 411.7 | 278.7 | 348.0 | 149.3 | 262.3 |
| A17 | 290 | 82 | 74 | 216 | 32 | 50 |
| A18 | 726 | 315 | 318 | 408 | 113 | 202 |
| **A19** | **655** | **151** | 258 | 397 | 62 | 89 |

Net rank-1 gain (fixed − broken): A11 +183, A12 +215, A17 +208, A18 +411, **A19 +504**. A19 improves the rank of 1,810 queries and worsens 368 (4.9 : 1; A12 was 1.6 : 1). It keeps 81% of A18's "wrong, near" fixes (258 vs 318) while breaking less than half as many SapBERT-correct queries (151 vs 315).

### 3.5 Held-out-pool diagnostic (test; MRR, gold target always kept)

| model | all | held-out only | trainable only |
|---|---|---|---|
| A0 | 0.4475 | 0.5195 | 0.5421 |
| A17 | 0.5325 | 0.6045 | 0.6558 |
| A18 | 0.5990 | 0.6888 | 0.6903 |
| A19 | 0.6293 | 0.7105 | 0.7348 |

The gain holds on the held-out-only pool (A19 +0.191 over A0), and A19 is also best on the trainable-only pool, so the lift is not the held-out-exclusion shortcut.

## 4. Gate 6.1 and what the ablations say

| condition | result |
|---|---|
| best fusion beats the best single member on val MRR, CI excludes 0 | A19 0.6278 vs A12 0.5169 (best single variant), and vs A18 with CI [+0.025, +0.089]: **yes** |

**Gate 6.1: GO.** A19 (w = 0.5) is the first stage for Phase 6.2 and an input to 6.3.

- **A17 / A18 vs a single seed.** Averaging seeds helps the no-floor model a lot (+0.056 val, +0.063 test) and the floor model little (+0.010 val, +0.009 test). A12 memorises its training pairs (train MRR 0.99), and its residual noise differs across seeds, which averaging removes. A11's floor already keeps it close to SapBERT, so there is less to average. This is an interpretation; it was not tested separately.
- **A19 vs A17 and A18 (complementary?).** Yes. A11 rarely breaks SapBERT-correct queries and fixes few; A12 fixes more but breaks more. The mean of their cosines keeps most of A12's fixes and about half of its breaks.
- **A20 vs A19 (do GAT variants add?).** No: −0.011 val, inside the CI. The GAT models behave like the GCN ones (Phase 5.6), so they add members but little diversity.

## 5. Limitations and caveats

- **Weight chosen on val.** w was picked on 211 pairs from a 5-point grid. The test MRR (0.6293) is close to the val MRR (0.6278), so there is no sign of over-fitting, but w = 0.5 and 0.75 are indistinguishable on val.
- **One ensemble per variant on test.** A17–A19 are deterministic functions of 3 fixed seeds each, so there is no seed std. The test CI covers query sampling only.
- **Test reads.** Test was read for A0, A17, A18, A19 (evaluator, stratified and pool scripts). `srga_fusion_bootstrap.py --split test` also printed the Sprint 5 rows A11 and A12, which are already reported. No other group was passed.
- **Cost.** A19 stores 6 models' embeddings (a 1,280-d fused vector in the first-stage layout) and needs no retraining.
- **Target comparison.** The reference 0.62 / 0.86 protocol is still unverified (sprint_6.md open question). A19 exceeds it in both modes, but that is not evidence of equivalence with the published model.
- Ensembling adds no training information. The A12 half still breaks SapBERT-correct queries (§3.4).
- The first A19 grid run (`T141237Z`) was started in the same shell as the A17/A18/A20 run and is a duplicate of the rerun `T141542Z`, which is the one reported.
- `ruff` is not installed, so lint was not run.

## 6. Files

- Modified: `src/cga/pipelines/evaluate/methods.py`, `tests/unit/pipelines/evaluate/test_methods.py`, `scripts/{srga_paired_ranks,srga_stratified,srga_pool_diagnostic}.py`, `dev_docs/sprints/sprint_6.md` (Phase 6.1 boxes ticked).
- Created: `scripts/srga_fusion_bootstrap.py`, `configs/experiments/ncit_doid_srga_fusion{,_grid,_test}.yaml`, this log.
- Untracked artifacts (evaluate runs, 20261016): `T141009Z-d5e7ae49` (val: A17, A18, A20), `T141237Z-28788437` and `T141542Z-28788437` (A19 grid; the second is reported), `T141926Z-961181e2` (test).

## 7. Commands

```
uv run pytest tests/unit -q

uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_fusion
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_fusion_grid
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_fusion_test   # test, once

G='--group A11=srga-gcn-l0.25-s{seed} --group A12=gcn-full-e200-s{seed} --group A14=srga-gat-l0.25-s{seed} --group A15=gat-full-e100-s{seed}'
uv run python scripts/srga_fusion_bootstrap.py --split val $G --fuse A17=A11 --fuse A18=A12 --fuse A19=A11:0.5,A12:0.5 --fuse A20=A11,A12,A14,A15 --compare A17:A11 --compare A18:A12 --compare A19:A17 --compare A19:A18 --compare A20:A19 --compare A19:A0

G='--group A11=srga-gcn-l0.25-s{seed} --group A12=gcn-full-e200-s{seed}'
uv run python scripts/srga_stratified.py --split test $G --fuse A17=A11 --fuse A18=A12 --fuse A19=A11:0.5,A12:0.5
uv run python scripts/srga_pool_diagnostic.py --split test $G --fuse A17=A11 --fuse A18=A12 --fuse A19=A11:0.5,A12:0.5 --only A17 A18 A19
```
