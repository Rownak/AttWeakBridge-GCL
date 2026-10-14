# Sprint 3.4 — `alpha` sweep (repeat of Phase 2.2) (execution log)

GCN + SapBERT + DeepWalk, sampler fixed at the Phase 3.3 winner (`rank_topk`, `k=5`), 100 epochs,
`alpha in {0, 0.25, 0.5, 0.75, 1}`. `alpha=0` is graph-loss-only, `alpha=1` is text-loss-only.

## Task 1 — Sweep spec

[configs/sweeps/ncit_doid_alpha.yaml](../../configs/sweeps/ncit_doid_alpha.yaml): `pipeline: gcl_train`,
`dataset: ncit_doid`; `gcl_train.loss.alpha` zipped with `gcl_train.model.model_id`
(`gcn-alpha0`, `gcn-alpha0.25`, `gcn-alpha0.5`, `gcn-alpha0.75`, `gcn-alpha1`, using the `%g` rule); `fixed:` sets
`training_data_sampler_id: rank_topk-k5` and `train.epochs: 100`.

## Task 2 — Dry-check

Expanded the spec with `expand_sweep` and resolved each point through `build_config` (same layering `cga sweep` uses)
before launching. Every point resolved to the expected `alpha`, `model_id`, `rank_topk-k5` and 100 epochs, and each
`model_id` is the `gcn-alpha<%g>` string the experiment config references. Note `sweep_point_id` is
`alpha0-model_idgcn-alpha0` (the zipped `model_id` is part of the id); it is only a manifest label and does not
affect artifact paths, which come from `model_id`.

## Task 3 — Run, evaluate, plot

```
uv run cga sweep --spec configs/sweeps/ncit_doid_alpha.yaml
uv run cga run evaluate --dataset ncit_doid --experiment ncit_doid_alpha_sweep_compare
uv run cga run plot --dataset ncit_doid --config configs/plots/ncit_doid_alpha_sweep_compare.yaml
```
- Sweep: 5/5 points succeeded, 85 s total (about 16 s per model), peak RAM 1.60 GB.
- evaluate run `20261014T142800Z-4741f26f`: 42 min, peak RAM 2.77 GB.
- plot run `20261014T151008Z-5be566af` (manifest confirms it read that evaluate run).
- New configs: [experiments](../../configs/experiments/ncit_doid_alpha_sweep_compare.yaml),
  [plots](../../configs/plots/ncit_doid_alpha_sweep_compare.yaml) (same shape as the Sprint 2 alpha files).

### Results

`mrr_textual` / `precision@10_textual`: `textual` view, mean over NCIT and DOID. `silhouette_mean`: mean over both
graphs and all `n_clusters`. `structural_r1`: MRR and precision@10 averaged over both graphs (context, not a
selection metric). Sorted by `mrr_textual`:

| alpha | mrr_textual | precision@10_textual | silhouette_mean | structural_r1 MRR / p@10 | final_loss |
|---|---|---|---|---|---|
| **0.25** | **0.3759** | **0.1387** | 0.1503 | 0.565 / 0.690 | 0.0450 |
| 0.5 (default) | 0.3756 | 0.1359 | **0.1515** | 0.509 / 0.610 | 0.0571 |
| 0.0 (graph-only) | 0.3687 | 0.1381 | 0.1266 | **0.600 / 0.782** | 0.0009 |
| 0.75 | 0.3687 | 0.1310 | 0.1466 | 0.419 / 0.479 | 0.0532 |
| 1.0 (text-only) | 0.3176 | 0.1069 | 0.1230 | 0.248 / 0.244 | 0.0105 |

`final_loss` is recorded but not used for selection: the objective changes with `alpha`.

**Selected on `mrr_textual`: `alpha = 0.25`.**

### Ablation readout

- An intermediate `alpha` beats both ends on the selection metric: 0.25 and 0.5 give 0.3759 and 0.3756 against 0.3687
  (`alpha=0`) and 0.3176 (`alpha=1`). The same holds for `precision@10_textual` against `alpha=1`; against `alpha=0`
  it is a near-tie (0.1387 vs 0.1381).
- The margin of 0.25 over 0.5 (0.0003) and of both over `alpha=0` (0.007) is small; with a single seed treat
  0.25 and 0.5 as tied. For reference, the default `alpha=0.5` re-trained here scored 0.3756 vs 0.3750 in
  Phase 3.3, so run-to-run variation is of the order 0.001.
- `alpha=1` (text-only) is clearly worst on everything: dropping the graph loss hurts both the structural and
  the textual neighborhoods.
- `alpha=0` is best on the structural view (r1 p@10 0.782) but still scores 0.369 on the textual view, because
  the SapBERT vectors enter as the encoder's input features, so graph-only training does not remove text signal.
  Structural r1 quality falls steadily as `alpha` rises (0.600 -> 0.565 -> 0.509 -> 0.419 -> 0.248), so the choice of
  `alpha` trades structure against text; the textual selection metric favors a small amount of the graph loss.
- Not compared against `attack_cwe`'s alpha results here; that belongs in the Phase 3.5 write-up.
- Single seed and proxy relevance sets (sprint caveat): relative comparison only.

## Files

New: `configs/sweeps/ncit_doid_alpha.yaml`, `configs/experiments/ncit_doid_alpha_sweep_compare.yaml`,
`configs/plots/ncit_doid_alpha_sweep_compare.yaml`, this log. Modified: `dev_docs/sprints/sprint_3.md` (ticks).
Generated (not committed): `models/gcl/ncit_doid/gcn-alpha*`, `artifacts/evaluations|plots/ncit_doid/`.
