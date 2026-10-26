# Sprint 6, Phase 6.3: More supervision for the final model, and a data-scaling curve (A25)

Plan: [sprint_6.md](../sprints/sprint_6.md). Previous phases: [6.1](sprint_6_1.md) (fusion, A19), [6.2](sprint_6_2.md) (stopped at the probe gate). The recipes are the frozen Sprint 5 ones; nothing was re-tuned.

## 0. Outcome in brief

- **Gate 6.3: GO.** On test, every full-data (train + val) model beats its train-only counterpart by more than the larger of the two seed stds, so data **is** still a limit at this size, and the full-data models become the Sprint 6 final models. The gains are small, except for A12.
- **Test, global MRR (mean ± std over 3 seeds; read once):**

| model | train-only | full-data (A25) | change | larger seed std |
|---|---|---|---|---|
| A3 MLP | 0.4957 ± 0.0024 | 0.5003 ± 0.0006 | +0.0046 | 0.0024 |
| A11 GCN + floor | 0.5234 ± 0.0030 | 0.5271 ± 0.0030 | +0.0037 | 0.0030 |
| A12 GCN, no floor | 0.5361 ± 0.0073 | 0.5545 ± 0.0024 | +0.0184 | 0.0073 |
| A19 fusion (6.1) | 0.6293 (one ensemble) | **0.6385** (one ensemble) | +0.0092 | no seed std |

- **The A11 margin is thin** (+0.0037 against a 0.0030 std), and the A3 and A11 changes are each inside about 1.5 seed stds. Only A12 and the fusion move clearly.
- **The scaling curve predicted the A11 change and under-predicted A12's.** Val slope per doubling of pairs: A11 +0.020, A12 +0.045 MRR. For +17.8% pairs that is +0.0046 (A11) and +0.0106 (A12); observed on test +0.0037 and +0.0184.
- **A0 test equals Sprint 5 bit-for-bit** (8 of 8 values), so the stores, test pairs and query set are unchanged and the comparison is paired on identical queries.
- The final first stage is the full-data fusion (**A25 A19: global MRR 0.6385, Hits@10 0.8832; candidates MRR 0.6889, Hits@10 0.9448**).

## 1. What was implemented

- **`pair_fraction` / `pair_seed`** (`gcl_train.train.*`, default `1.0` / `0`, so every earlier run is unchanged): `sample_pairs_by_source()` in [losses.py](../../src/cga/pipelines/gcl_train/losses.py).
  - Samples `round(fraction × n)` of the distinct **source nodes** with `random.Random(seed)` and keeps all their pairs (a source with two targets is never half-sampled).
  - [pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py) applies it to the positive pairs before `build_full_softmax_data`. The `gold_pairs` mask is **not** reduced, so a dropped pair is never scored as a false negative, and its endpoints stay ordinary candidates.
  - `n_positive_pairs` (distinct pairs trained on, `infonce_full` only) is added to the manifest summary.
- **Lineage `ncit_doid_srga_full`:** [configs/datasets/ncit_doid_srga_full.yaml](../../configs/datasets/ncit_doid_srga_full.yaml) = `ncit_doid_srga` without `val_split`; own id, so no Sprint 4/5 artifact or `LATEST` pointer was touched.
  - Ingest `20261016T205126Z-c2ddb846` (`ingest.split.fractions=[1.0,0.0,0.0]`): alignments train 1,406 + test 3,280. Its `nodes.parquet` and `edges.parquet` equal the `ncit_doid_srga` ones (24,227 nodes, 32,874 edges).
  - The three embedding stores (sapbert `20261009T182919Z-95011340`, bert_base `20261009T191700Z-1278c23a`, deepwalk `20261011T134446Z-d5aa89ed`) were copied **without `shards/`** (`vectors.npy`, `index.parquet`, `status.parquet`, `manifest.json`, `stats.json`, `LATEST`); the copy was instant.
    - `vectors.npy`, `index.parquet` and `status.parquet` are byte-identical to the `ncit_doid_srga` copies, `AlignedMatrix.load` (which checks the manifest hashes) passes, and the `node_id` order equals the new ingest's.
    - For sapbert and bert_base, `input_sha256` matches the ingest's `label_synonyms_definition` `text_sha256` for 24,227 of 24,227 nodes. The deepwalk store has no text hash (0 matches is expected); it is verified by byte equality and node order only.
  - `training_data` run `20261016T205359Z-0266e30b` (`supervised-negatives_per_positive5`): 6,915 examples, **1,383 of 1,406 train pairs usable** (23 skipped), 6,515 held-out nodes (test endpoints only; 420 fewer than the 6,935 of the train-only lineage). The train-only lineage used 1,174 pairs, so the full-data pair count is **+17.8%** (a ratio of 1.178).
- **Scripts** (analysis only, read artifacts):
  - [srga_scaling_curve.py](../../scripts/srga_scaling_curve.py) (new): mean ± std per recipe and size, and the log-linear fit with its standard error and predicted gain.
  - `srga_paired_ranks.py`, `srga_stratified.py`, `srga_pool_diagnostic.py`, `srga_fusion_bootstrap.py`: a group pattern may be `lineage::model_id{seed}` (`model_run_dir()` in `srga_paired_ranks.py`), so a full-data model is ranked on the same queries as its train-only counterpart.
- **Configs:** [ncit_doid_srga_scaling.yaml](../../configs/experiments/ncit_doid_srga_scaling.yaml) (val, 24 models) and [ncit_doid_srga_full_test.yaml](../../configs/experiments/ncit_doid_srga_full_test.yaml) (test; A25 A3, A11, A12 and the rebuilt A19).
- README: the SRGA section lists the new option, lineage, scripts and the cross-lineage group syntax.
- **Tests** (480 pass; 7 new): `pair_fraction` is deterministic for a seed, moves whole sources, keeps the dropped pair in the gold mask, defaults to the old behaviour (`1.0` returns the de-duplicated pairs), and rejects a fraction outside `(0, 1]`; an `infonce_full` run with `pair_fraction = 0.5` trains on 1 pair against 2 by default; the full-data config loads, has no `val_split`, and equals the train-only config in everything else.

## 2. Data-scaling curve (val, 211 pairs, global; mean ± sample std over seeds 42 / 43 / 44)

24 models: `scale-gcn-p{25,50,75,100}-s<seed>` (A11 recipe: GCN + floor, λ 0.25, 50 epochs) and `scale-gcn-nofloor-p…` (A12 recipe: no floor, 200 epochs), all other hyper-parameters as Sprint 5; `pair_seed` = the training seed. Evaluate run `20261016T205911Z-f549bb59`.

| recipe | size | positive pairs (seeds 42/43/44) | val MRR | val Hits@10 |
|---|---|---|---|---|
| A11 | 25% | 293 / 294 / 294 | 0.4560 ± 0.0134 | 0.6746 ± 0.0243 |
| A11 | 50% | 588 / 587 / 587 | 0.4754 ± 0.0167 | 0.7330 ± 0.0197 |
| A11 | 75% | 880 / 880 / 880 | 0.4890 ± 0.0143 | 0.7425 ± 0.0119 |
| A11 | 100% | 1,174 | 0.4938 ± 0.0078 | 0.7425 ± 0.0145 |
| A12 | 25% | 293 / 294 / 294 | 0.4205 ± 0.0192 | 0.6367 ± 0.0119 |
| A12 | 50% | 588 / 587 / 587 | 0.4665 ± 0.0080 | 0.6951 ± 0.0109 |
| A12 | 75% | 880 / 880 / 880 | 0.4831 ± 0.0023 | 0.7156 ± 0.0082 |
| A12 | 100% | 1,174 | 0.5146 ± 0.0180 | 0.7393 ± 0.0264 |

- The 100% rows retrain the Sprint 5 recipes and agree with Phase 5.5 (A11 0.4924 ± 0.0031, A12 0.5169 ± 0.0209): the code change is neutral at `pair_fraction = 1.0`, within the run-to-run noise of these runs (training is not bit-reproducible here).
- **Fit `MRR = a + b · ln(pairs)`** over all 12 (seed, size) points per recipe:

| recipe | b (MRR per ln-pair) | SE(b) | MRR per doubling | predicted gain for +17.8% pairs | range (b ± 1.96 SE) |
|---|---|---|---|---|---|
| A11 | +0.0281 | 0.0067 | +0.0195 | **+0.0046** | [+0.0024, +0.0068] |
| A12 | +0.0647 | 0.0074 | +0.0448 | **+0.0106** | [+0.0082, +0.0130] |

- Both curves are still rising at 100%; the slope is steeper for the no-floor model (it relies on the supervised signal more: with 25% of the pairs it is 0.036 below A11, with 100% it is above it). The A12 curve is not smooth (0.4831 at 75% to 0.5146 at 100%, with a large 100% std), so its straight-line fit is a rough guide.
- The fit's range reflects the scatter around the line, not the seed noise of one new model or the val-to-test transfer.

## 3. A25 retraining and test readout

Models on `ncit_doid_srga_full` (seeds 42 / 43 / 44, frozen recipes, no val readout and no new selection):

| variant | recipe | model ids | run ids (s42 / s43 / s44) |
|---|---|---|---|
| A3 | MLP adapter, floor λ 0.25, 50 epochs | `full-mlp-l0.25-s<seed>` | `T205427Z-2e8c121d` / `T205526Z-751f4b79` / `T205622Z-ef72ce61` |
| A11 | GCN adapter, floor λ 0.25, 50 epochs | `full-gcn-l0.25-s<seed>` | `T205445Z-d2fbf4e3` / `T205542Z-bca7f910` / `T205638Z-ed6929fa` |
| A12 | GCN adapter, no floor, 200 epochs | `full-gcn-e200-s<seed>` | `T205503Z-fdd8c6cc` / `T205559Z-71e16931` / `T205655Z-ae10dcb3` |
| A19 | 0.5 · mean of the three A12 + 0.5 · mean of the three A11 (6.1 fusion) | method `A25-A19` | members above |

(Dates 20261016. The `infonce_full` loss, τ 0.05, Adam lr 1e-3, weight decay 1e-4 and `reverse_weight` 0.5 are as in Sprints 5.2–5.6.)

Test (3,280 pairs, read once), evaluate run `20261016T210220Z-f96e78e5` on `ncit_doid_srga_full`. Train-only rows are the existing Sprint 5 / 6.1 results.

**global**

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| A3 train-only | 0.3784 ± 0.0035 | 0.6359 ± 0.0020 | 0.7295 ± 0.0069 | 0.4957 ± 0.0024 |
| **A3 full-data** | 0.3825 ± 0.0026 | 0.6393 ± 0.0018 | 0.7275 ± 0.0075 | 0.5003 ± 0.0006 |
| A11 train-only | 0.3982 ± 0.0054 | 0.6732 ± 0.0027 | 0.7708 ± 0.0030 | 0.5234 ± 0.0030 |
| **A11 full-data** | 0.4021 ± 0.0030 | 0.6784 ± 0.0060 | 0.7769 ± 0.0030 | 0.5271 ± 0.0030 |
| A12 train-only | 0.4079 ± 0.0102 | 0.6952 ± 0.0044 | 0.7725 ± 0.0040 | 0.5361 ± 0.0073 |
| **A12 full-data** | 0.4264 ± 0.0059 | 0.7143 ± 0.0006 | 0.7923 ± 0.0024 | 0.5545 ± 0.0024 |
| A19 train-only (6.1) | 0.4960 | 0.8006 | 0.8777 | 0.6293 |
| **A19 full-data** | **0.5076** | **0.8040** | **0.8832** | **0.6385** |

**candidates** (101 per pair)

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| A3 train-only | 0.4325 ± 0.0034 | 0.7355 ± 0.0034 | 0.8450 ± 0.0015 | 0.5687 ± 0.0022 |
| **A3 full-data** | 0.4376 ± 0.0024 | 0.7364 ± 0.0045 | 0.8452 ± 0.0050 | 0.5726 ± 0.0021 |
| A11 train-only | 0.4440 ± 0.0060 | 0.7613 ± 0.0029 | 0.8710 ± 0.0025 | 0.5852 ± 0.0025 |
| **A11 full-data** | 0.4475 ± 0.0034 | 0.7654 ± 0.0049 | 0.8703 ± 0.0046 | 0.5887 ± 0.0032 |
| A12 train-only | 0.4742 ± 0.0134 | 0.8103 ± 0.0023 | 0.9034 ± 0.0045 | 0.6209 ± 0.0088 |
| **A12 full-data** | 0.4864 ± 0.0074 | 0.8225 ± 0.0012 | 0.9103 ± 0.0033 | 0.6341 ± 0.0038 |
| A19 train-only (6.1) | 0.5415 | 0.8622 | 0.9390 | 0.6799 |
| **A19 full-data** | **0.5537** | **0.8634** | **0.9448** | **0.6889** |

Per-seed global MRR (42 / 43 / 44): A3 0.4997 / 0.5008 / 0.5002; A11 0.5295 / 0.5237 / 0.5282; A12 0.5542 / 0.5570 / 0.5523.

**A0 check:** the 8 test values of the SapBERT row (4 global, 4 candidates) are exactly equal to evaluate run `20261015T233935Z-038f363f` of Sprint 5.6.

### 3.1 Paired comparison against the train-only counterparts (test, global, same 3,280 queries)

Paired bootstrap (1,000 resamples over queries; each side is the per-query reciprocal rank averaged over its seeds, so the fusion rows compare one ensemble to one ensemble):

| comparison | MRR (full / train-only) | diff | 95% CI | wins / losses / ties |
|---|---|---|---|---|
| A3 full vs train-only | 0.5003 / 0.4957 | +0.0046 | [+0.0006, +0.0085] | 1,133 / 1,056 / 1,091 |
| A11 full vs train-only | 0.5271 / 0.5234 | +0.0037 | [+0.0002, +0.0072] | 1,033 / 907 / 1,340 |
| A12 full vs train-only | 0.5545 / 0.5361 | +0.0184 | [+0.0109, +0.0257] | 1,410 / 1,142 / 728 |
| A19 full vs train-only | 0.6385 / 0.6293 | +0.0092 | [+0.0018, +0.0161] | 755 / 619 / 1,906 |
| A19 full vs A0 | 0.6385 / 0.4475 | +0.1909 | [+0.1784, +0.2031] | 1,838 / 345 / 1,097 |

Same-seed counts of the per-seed models (full-data wins vs train-only wins on the queries where the ranks differ; seeds 42 / 43 / 44, from `srga_paired_ranks.py`):

| model | full-data wins | train-only wins |
|---|---|---|
| A3 | 956 / 959 / 894 | 855 / 888 / 921 |
| A11 | 924 / 820 / 789 | 670 / 796 / 783 |
| A12 | 1,085 / 1,120 / 1,115 | 981 / 960 / 966 |

The A3 and A11 changes therefore rest on margins of tens of queries per seed: A11 seed 44 is a near tie (789 vs 783) and **A3 seed 44 goes the other way** (894 vs 921). Only A12 is ahead on every seed by a margin of about 100-150 queries.

Broken SapBERT-correct queries (of 1,123; mean over seeds): A3 137 → 124, A11 98 → 91, A12 412 → 396, A19 151 → 147. More data does not remove the A12 breaks.

## 4. Gate 6.3 and what it says

| condition | A3 | A11 | A12 | A19 (fusion) |
|---|---|---|---|---|
| full-data − train-only (test global MRR) | +0.0046 | +0.0037 | +0.0184 | +0.0092 |
| larger seed std of the two | 0.0024 | 0.0030 | 0.0073 | none (one ensemble per side) |
| change > larger std | yes | yes (by 0.0007) | yes | n/a; query-level CI [+0.0018, +0.0161] excludes 0 |

**Gate 6.3: GO.** Full-data models beat their train-only counterparts on test by more than the larger seed std, for all three recipes, and the fusion is ahead with a CI that excludes 0. The full-data models are kept as the final Sprint 6 models.

The data-size conclusion, stated plainly: **adding the 211 val pairs (+17.8%) helps, but not by much.** The expected gain from the curve is about +0.005 (GCN with floor) to +0.011 (no floor) MRR for this much data; the observed test gains are +0.004 / +0.018 (+0.005 for the MLP, +0.009 for the fusion). Data is a real but modest lever at this size: doubling the pairs would be worth about +0.02 MRR for A11 and +0.045 for A12 by the fits, and it does not close the gap on its own. The curve has not flattened at 100% for either recipe.

Caveats on the claim: the A11 and A3 effects are each within about 1.5 seed stds; the extrapolation is log-linear from a 4-point curve measured on val (211 pairs) and applied to a different split; and A12's test gain (+0.0184) is above the fit's range [+0.0082, +0.0130], most likely because its 100% val point is noisy (std 0.018) rather than evidence for a steeper slope.

## 5. Limitations and caveats

- **No independent readout for the full-data models:** val is training data, so there is no val number, and the single test read is the only check. This is by design (plan: no val readout, no new selection).
- **Seed noise:** the A11 comparison has a margin of 0.0007 over its seed std, and training is not bit-reproducible (GCN/GAT kernels), so a new set of seeds could move A3 and A11 either way by about that much.
- **The scaling fit is rough** (4 points per recipe, 3 seeds, val only) and its standard error covers scatter around the line, not extrapolation risk.
- **The fusion's `±` is absent:** one ensemble per side; the CI covers query sampling only (as in 6.1).
- **The 23 unusable train pairs** (sources or targets outside the universe) are skipped by `training_data`, as in the train-only lineage (21 of 1,195); 1,383 of 1,406 train pairs train.
- **Phase 6.2 stopped at its probe gate**, so the optional extra row that applies 6.2's val-fitted weights to the full-data first stage was not run (nothing to apply).
- `pair_fraction` samples whole sources, so the realised pair count differs a little by seed (293 / 294 / 294 at 25%).
- `ruff` is not installed, so lint was not run.
- Disclosure of test reads: test was read for A0 and the A25 models (A3, A11, A12 and the fusion), plus the paired analyses of those same groups against their train-only counterparts. The scaling runs were evaluated on val only.

## 6. Files

- Modified: `src/cga/pipelines/gcl_train/{losses,pipeline}.py`, `tests/unit/pipelines/gcl_train/{test_losses,test_pipeline}.py`, `scripts/{srga_paired_ranks,srga_stratified,srga_pool_diagnostic,srga_fusion_bootstrap}.py`, `README.md`, `dev_docs/sprints/sprint_6.md` (Phase 6.3 boxes ticked).
- Created: `configs/datasets/ncit_doid_srga_full.yaml`, `configs/experiments/ncit_doid_srga_{scaling,full_test}.yaml`, `scripts/srga_scaling_curve.py`, `tests/unit/test_full_data_dataset_config.py`, this log.
- Untracked artifacts (all 20261016):
  - `data/processed/ncit_doid_srga_full/20261016T205126Z-c2ddb846`; `artifacts/embeddings/ncit_doid_srga_full/` (three stores, no shards); `artifacts/training_data/ncit_doid_srga_full/supervised-negatives_per_positive5/20261016T205359Z-0266e30b`;
  - 24 scaling models `scale-gcn-p<25|50|75|100>-s<seed>` and `scale-gcn-nofloor-p…` under `models/gcl/ncit_doid_srga/` (they do not move any existing `LATEST`);
  - 9 A25 models under `models/gcl/ncit_doid_srga_full/` (table in §3);
  - evaluate runs `20261016T205911Z-f549bb59` (scaling, val, `ncit_doid_srga`) and `20261016T210220Z-f96e78e5` (A25 test, `ncit_doid_srga_full`).

## 7. Commands

```
uv run pytest tests/unit -q

# Scaling curve (one of 24 shown: A11 recipe at 50% of the source nodes' pairs, seed 43). A12 recipe: add residual_floor=false and epochs 200.
uv run cga run gcl_train --dataset ncit_doid_srga --set seed=43 --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=gcn --set gcl_train.model.lambda=0.25 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.train.pair_fraction=0.5 --set gcl_train.train.pair_seed=43 --set gcl_train.model.model_id=scale-gcn-p50-s43
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_scaling
uv run python scripts/srga_scaling_curve.py --evaluate-run 20261016T205911Z-f549bb59

# Full-data lineage
uv run cga run ingest --dataset ncit_doid_srga_full --set "ingest.split.fractions=[1.0,0.0,0.0]"
#   copy artifacts/embeddings/ncit_doid_srga/<store>/<run>/{vectors.npy,index.parquet,status.parquet,manifest.json,stats.json} and <store>/LATEST
#   to artifacts/embeddings/ncit_doid_srga_full/ for hf_mean_pool-sapbert, hf_mean_pool-bert_base and deepwalk-deepwalk (no shards/), then verify.
uv run cga run training_data --dataset ncit_doid_srga_full --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5

# A25 (seed 43 shown). A11: adapter=gcn, epochs 50. A12: adapter=gcn, residual_floor=false, epochs 200. A3: adapter=mlp, epochs 50.
uv run cga run gcl_train --dataset ncit_doid_srga_full --set seed=43 --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=gcn --set gcl_train.model.lambda=0.25 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.model.model_id=full-gcn-l0.25-s43

# Test, once
uv run cga run evaluate --dataset ncit_doid_srga_full --experiment ncit_doid_srga_full_test
G='--group A3=srga-mlp-l0.25-s{seed} --group A3F=ncit_doid_srga_full::full-mlp-l0.25-s{seed} --group A11=srga-gcn-l0.25-s{seed} --group A11F=ncit_doid_srga_full::full-gcn-l0.25-s{seed} --group A12=gcn-full-e200-s{seed} --group A12F=ncit_doid_srga_full::full-gcn-e200-s{seed}'
uv run python scripts/srga_paired_ranks.py --split test $G
uv run python scripts/srga_fusion_bootstrap.py --split test $G --fuse A19=A11:0.5,A12:0.5 --fuse A19F=A11F:0.5,A12F:0.5 --compare A3F:A3 --compare A11F:A11 --compare A12F:A12 --compare A19F:A19 --compare A19F:A0
```
