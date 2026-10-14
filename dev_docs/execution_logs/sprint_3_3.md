# Sprint 3.3 — Sampler sweep (repeat of Phase 2.1) (execution log)

GCN + SapBERT + DeepWalk, 100 epochs, `alpha=0.5`, one `training_data` run per sampler point.
Inputs as in Phase 3.2 (ingest `20261011T134334Z-6d45b8af`, `hf_mean_pool-sapbert`, DeepWalk `d5aa89ed`).

## Task 1 — Sweep specs

- [configs/sweeps/ncit_doid_rank_topk_k.yaml](../../configs/sweeps/ncit_doid_rank_topk_k.yaml): `rank_topk`, `k: [5, 10, 15]`.
- [configs/sweeps/ncit_doid_percentile_pct.yaml](../../configs/sweeps/ncit_doid_percentile_pct.yaml): `percentile`, zipped
  `top_pct`/`bottom_pct: [0.02, 0.05, 0.1]`.
- Both set `dataset: ncit_doid` and pin `training_data.inputs.text_embeddings.embedder_id: hf_mean_pool-sapbert` under
  `fixed:` (the pipeline default is SecRoBERTa). `fixed:` values stay out of `sweep_point_id`, so ids are
  `rank_topk-k5`, `percentile-top_pct0.02-bottom_pct0.02`, etc.

## Task 2 — Run the sweeps and train

```
uv run cga sweep --spec configs/sweeps/ncit_doid_rank_topk_k.yaml
uv run cga sweep --spec configs/sweeps/ncit_doid_percentile_pct.yaml
uv run cga run gcl_train --dataset ncit_doid --set gcl_train.inputs.training_data_sampler_id=<id> \
  --set gcl_train.model.model_id=gcn-<id> --set gcl_train.train.epochs=100      # per point
```

All 6 `training_data` points succeeded. `gcl_train` succeeded for 4 of 6:

| Sweep point | `n_examples` | `training_data` | `gcl_train` | Peak RAM (gcl_train) |
|---|---|---|---|---|
| `rank_topk` k=5 | 96,880 | rank_topk sweep: 250 s total, 2.2 GB | 33 s | 1.45 GB |
| `rank_topk` k=10 | 193,760 | (same sweep) | 44 s | 1.48 GB |
| `rank_topk` k=15 | 290,640 | (same sweep) | 59 s | 1.52 GB |
| `percentile` 2% | 2,635,136 | percentile sweep: 821 s total, 12.0 GB | 38 min | 18.4 GB |
| `percentile` 5% | 6,568,464 | (same sweep) | **CUDA OOM** (3.13 GiB request, 16 GB GPU) | 44.8 GB at failure |
| `percentile` 10% | 13,136,928 | (same sweep) | **CUDA OOM** (6.27 GiB request) | 48.3 GB at failure |

### Limitation: percentile points at 5% and 10% could not be trained

`percentile` keeps a *percentage of all train-anchor pairs* (12,604 NCIT x 6,772 DOID ~ 85M pairs), so example counts
are 27x to 136x those of `rank_topk` k=5, against 68k at most in Sprint 2 (`attack_cwe` is far smaller). `gcl_train`
puts the whole example set on the GPU, so 5% and 10% run out of memory, and even 2% took 38 min.
Decision (developer): skip 5% and 10% and evaluate the other 4 points. The failed runs left log-only directories
(`models/gcl/ncit_doid/gcn-percentile-top_pct0.05-...`, `...0.1-...`: no weights, no `LATEST`); `evaluate` ignores them.
Not done: example mini-batching in `gcl_train`, or a rescaled percentile grid (e.g. 0.2%-1%).

Spec note: the sprint text says "8 `gcl_run` methods", but the specified grids give 3 + 3 = 6 points (4 after the skip).

## Task 3 — Experiment and plot configs

- [configs/experiments/ncit_doid_sampler_sweep_compare.yaml](../../configs/experiments/ncit_doid_sampler_sweep_compare.yaml):
  4 `gcl_run` methods (the 5% and 10% entries are left out, with a comment); views and cluster settings identical to
  Phase 3.2 (`structural_r1/r2`, `textual` vs `hf_mean_pool-bert_base`, `ks [1,5,10,20]`, `n_clusters [2,4,8,16,32]`).
- [configs/plots/ncit_doid_sampler_sweep_compare.yaml](../../configs/plots/ncit_doid_sampler_sweep_compare.yaml):
  precision@k, MRR and silhouette line plots plus loss curves.

```
uv run cga run evaluate --dataset ncit_doid --experiment ncit_doid_sampler_sweep_compare
uv run cga run plot --dataset ncit_doid --config configs/plots/ncit_doid_sampler_sweep_compare.yaml
```
evaluate run `20261012T103219Z-cacb819c` (48 min, peak RAM 2.72 GB); plot run `20261012T112028Z-5be566af`.

## Task 4 — Reporting table

`mrr_textual` and `precision@10_textual`: `textual` view, mean over NCIT and DOID. `silhouette_mean`: mean over both
graphs and all five `n_clusters`. `final_loss`: last row of `metrics.jsonl`. Sorted by `mrr_textual`:

| sweep point | mrr_textual | precision@10_textual | silhouette_mean | final_loss | n_examples |
|---|---|---|---|---|---|
| rank_topk k=5   | **0.3750** | **0.1360** | 0.1555 | 0.0567 | 96,880 |
| rank_topk k=10  | 0.3725 | 0.1348 | 0.1649 | 0.0775 | 193,760 |
| rank_topk k=15  | 0.3683 | 0.1323 | 0.1706 | 0.0922 | 290,640 |
| percentile 2%   | 0.3290 | 0.1132 | **0.2135** | 0.2151 | 2,635,136 |
| percentile 5%, 10% | not run (GPU OOM) | | | | |

(Extra, not a selection metric: `structural_r1` MRR averaged over both graphs: 0.508 / 0.481 / 0.460 / 0.322.)

**Winner (on `mrr_textual`): `rank_topk`, `k=5`**, the same as `attack_cwe`'s (`rank_topk`, `k=5`).

### Reading
- `rank_topk`: `mrr_textual`, `precision@10` and `structural_r1` MRR all fall monotonically as `k` grows
  (0.3750 -> 0.3725 -> 0.3683), the same direction as `attack_cwe`. The gaps are small (0.007 between k=5 and k=15),
  so within single-seed noise k=5 and k=10 are close; k=5 is also the cheapest.
- `percentile` 2% is clearly worst on the textual and structural views (0.329 vs 0.375) while using 27x more examples,
  and only wins on silhouette (0.214), as in Sprint 2 where silhouette tracked larger sample sets. Silhouette is a
  sanity check, not a selection metric.
- Because 5% and 10% were not trained, the `percentile` trend over its grid is not established. The comparison
  `rank_topk` vs `percentile` rests on the 2% point alone.
- `final_loss` rises with `n_examples` and is not comparable across points.
- k=5 reproduces the Phase 3.2 GCN (mean MRR 0.375 vs 0.3745 from that table), as a consistency check.
- Single seed, 100 epochs; proxy relevance sets (sprint caveat), so a relative comparison only.

## Files created / modified

- New: the two sweep specs, `configs/experiments/ncit_doid_sampler_sweep_compare.yaml`,
  `configs/plots/ncit_doid_sampler_sweep_compare.yaml`, this log.
- Modified: `dev_docs/sprints/sprint_3.md` (ticks).
- Generated (not committed): `artifacts/training_data/ncit_doid/<6 sampler ids>/`, `models/gcl/ncit_doid/gcn-<id>/`,
  `artifacts/evaluations|plots/ncit_doid/`.
