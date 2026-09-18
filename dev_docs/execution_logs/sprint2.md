# Sprint 2 — Sampler sweep and comparison tooling

## Phase 2.1 — Sampler sweep infrastructure and comparison

### What was implemented

Built the two §2 prerequisites from `dev_docs/sweep_experiment_plan.md` (`sampler_id`-partitioned
`training_data` `LATEST`, then `cga sweep`), added tests for both, and wrote the two sweep spec
files plus the `sampler_sweep_compare` evaluate experiment. Confirmed two design points with the
developer before implementing rather than guessing: the `sampler_id` scheme
(`{sampler.name}-{key params}`, mirroring `build_model_id`) and `cga sweep`'s scope (sweeps exactly
one pipeline stage per spec, matching the plan's spec format — chaining `training_data -> gcl_train`
in one command was explicitly out of scope). A third question (how to keep 8 `gcl_train` runs, all
sharing the same encoder, from clobbering one `model_id`'s `LATEST`) was resolved the same way:
explicit `--set gcl_train.model.model_id=gcn-<sweep_point_id>` per run, no new CLI mechanism.

The developer then ran all 8 `training_data` + `gcl_train` (100 epochs) combinations and the
`sampler_sweep_compare` `evaluate` run themselves (`artifacts/evaluations/attack_cwe/
20260918T023545Z-e334ec67`). Fixed one bug surfaced by that run: `expand_sweep`'s `sweep_point_id`
sorted swept keys alphabetically, so `cga sweep`'s printed id for the percentile spec
(`percentile-bottom_pct...-top_pct...`) disagreed with `build_sampler_id`'s actual `LATEST`
directory name (`percentile-top_pct...-bottom_pct...`, param order declared in
`_SAMPLER_ID_PARAMS`). `_combination_id` now takes the spec file's own declared key order instead
of sorting, and `configs/experiments/sampler_sweep_compare.yaml`'s `model_id`s were corrected to
match. Also added `configs/plots/sampler_sweep_compare.yaml` (precision@10, mrr, silhouette line
figures + loss curves) — no new renderer code needed, `plot` is generic over `metrics.parquet`.

### Reporting table (Phase 2.1's last checklist item)

Pulled directly from the `evaluate` run's `metrics.json` (`textual` view, averaged over
`graph_1`/`graph_2` for `mrr`/`precision@10`; `clusters`, averaged over both roles and all five
`n_clusters` values, for `silhouette_mean`), each `gcl_train` run's `metrics.jsonl` last line
(`final_loss`, all 8 completed the full 100 epochs), and each `training_data` run's `stats.json`
(`n_examples`). Sorted by `mrr_textual` descending:

| sweep point | mrr_textual | precision@10_textual | silhouette_mean | final_loss | n_examples |
|---|---|---|---|---|---|
| rank_topk k=5   | **0.4351** | **0.1715** | 0.2094 | 0.0792 | 4,545  |
| rank_topk k=10  | 0.4320 | 0.1699 | 0.2328 | 0.1153 | 9,090  |
| percentile 2%   | 0.4187 | 0.1595 | 0.2224 | 0.1045 | 13,635 |
| percentile 8%   | 0.4154 | 0.1533 | 0.2729 | 0.1924 | 54,540 |
| percentile 5%   | 0.4128 | 0.1556 | 0.2622 | 0.1590 | 34,542 |
| rank_topk k=20  | 0.4119 | 0.1643 | 0.2521 | 0.1629 | 18,180 |
| rank_topk k=30  | 0.4116 | 0.1581 | 0.2672 | 0.1999 | 27,270 |
| percentile 10%  | 0.3937 | 0.1524 | 0.2738 | 0.2140 | 68,175 |

**Best sampler config: `rank_topk`, `k=5`** — highest `mrr_textual` (0.4351) and
`precision@10_textual` (0.1715) of all 8 points, and the lowest `final_loss` (0.0792, expected given
by far the fewest examples/epoch). `rank_topk k=10` is a close second on both textual metrics with
roughly double the examples. Both `rank_topk` results beat every `percentile` point on the primary
selection metric (`mrr_textual`); `percentile`'s advantage is confined to `silhouette_mean`
(sanity-check-only per the plan, not a selection criterion) and it only wins there as `top_pct`/
`bottom_pct` grow larger (10% > 8% > 5% > 2%) — i.e. as the sampler pulls in progressively easier,
less-informative pairs, which also tracks with the plan's framing of `silhouette` as a check against
degenerate embeddings rather than a target to maximize. Overall pattern for `rank_topk`: `mrr_textual`
decreases monotonically as `k` increases (0.4351 -> 0.4320 -> 0.4119 -> 0.4116) — smaller `k` keeps
only the most confident (closest/farthest) pairs, which appears to matter more for this task than
having more training examples.

### How it was completed

- **`sampler_id`-partitioned `training_data` `LATEST`** (`src/cga/pipelines/training_data/pipeline.py`):
  added `build_sampler_id(sampler_cfg)`, mirroring `gcl_train.pipeline.build_model_id`. Defaults to
  `{name}-{key params}` using a small `_SAMPLER_ID_PARAMS` map (`rank_topk` -> `k`; `percentile` ->
  `top_pct`, `bottom_pct`); an explicit `training_data.sampler.sampler_id` overrides it. `LATEST` now
  lives at `artifacts/training_data/{dataset_id}/{sampler_id}/LATEST`, one level deeper than before.
  Updated both write sites that need to agree on this path: `TrainingDataPipeline.run`'s
  `write_latest` call, and `cli._run_root`'s `training_data` branch (which now imports
  `build_sampler_id` the same way it already imports `build_embedder_id`/`build_model_id` for the
  other two pipeline-specific branches).
- **`gcl_train` now reads a specific `sampler_id`'s lineage**: since `training_data`'s `LATEST` moved
  one level deeper, `gcl_train.LoadStep` needed a way to say *which* `sampler_id`'s `training_data`
  run to resolve `latest`/an explicit run_id against. Added
  `gcl_train.inputs.training_data_sampler_id` (default `rank_topk-k30`, matching
  `training_data.yaml`'s own default sampler config) alongside the existing `training_data_run`.
- **`cga sweep --spec <file>`** (`src/cga/core/sweep.py`, new + `src/cga/cli.py`): `SweepSpec.from_file`
  parses `pipeline`/`dataset`/`sweep:` (dotted-path -> value-array map) plus an optional `zip:` list
  marking paired axes, validating equal-length zipped arrays and that every zipped key is itself
  present in `sweep:`. `expand_sweep` treats the zipped keys as one paired axis and crosses it
  (cartesian product, via `itertools.product`) against every other key, producing one `SweepPoint`
  per combination with a deterministic `sweep_point_id` (`name`-leaf keys render bare and sort first,
  e.g. `rank_topk-k20`, `percentile-bottom_pct0.05-top_pct0.05`) and duplicate-id detection (two
  distinct combinations rendering to the same id is a `ConfigError`, not a silent collision).
  `cli.cmd_sweep` builds each combination's config through the same `build_config` layering
  `cmd_run` uses (base -> dataset -> pipeline, then that point's overrides applied last — no
  `--experiment`/`--config` composition for a sweep spec, since the plan's example specs don't need
  it), runs it via a new shared `_run_pipeline` helper, and continues past a `CgaError` from any one
  combination (logs it, keeps going) rather than aborting the whole grid; the command's own exit
  code is non-zero if any combination failed. Refactored `cmd_run` to also use `_run_pipeline`, so
  there is exactly one place a `RunContext` gets built and run.
- **`sweep_point_id` in the manifest**: every pipeline's manifest step assigns
  `ctx.manifest.summary = {...}` wholesale (verified in `training_data`, `gcl_train`, `ingest`), so
  setting a `summary["sweep_point_id"]` key before the pipeline's steps run would be silently
  discarded by that later assignment. Added a dedicated `Manifest.sweep_point_id: str | None = None`
  field instead (`src/cga/core/manifest.py`), included in `to_dict()`; `cmd_sweep`/`_run_pipeline`
  sets it directly on `ctx.manifest`, immune to any step reassigning `summary`.
- **`configs/sweeps/rank_topk_k.yaml`** — `rank_topk.k ∈ {5, 10, 20, 30}`, `sampler.name` fixed to
  `[rank_topk]` (single-value array, not omitted, so the sweep spec is self-describing about which
  sampler it's testing).
- **`configs/sweeps/percentile_pct.yaml`** — `percentile.{top_pct,bottom_pct} ∈ {0.02, 0.05, 0.08, 0.1}`,
  paired via `zip:` (equal to each other, not crossed independently — they interact, per the plan).
  Per the Phase 2.1 checklist's clarification: the model is trained on *one* sampler at a time
  (`rank_topk` or `percentile`, never both in the same `training_data` run), so these are two
  separate sweep specs rather than one spec crossing sampler name against both parameter sets.
- **`configs/experiments/sampler_sweep_compare.yaml`** — same shape as the existing
  `full_grid_compare.yaml`, 8 `gcl_run` methods (`gcn_rank_topk_k{5,10,20,30}`,
  `gcn_percentile_p{02,05,08,10}`), each method's `model_id` explicitly `gcn-<sweep_point_id>`
  matching what each `gcl_train --set gcl_train.model.model_id=...` run will be launched with;
  `neighborhood` + `cluster_quality` evaluations reused verbatim from `full_grid_compare.yaml`
  (same `textual` reference embedder, same `n_clusters` grid), so the two experiments' metrics stay
  directly comparable.
- **`configs/plots/sampler_sweep_compare.yaml`** (new) — `precision_at_k`, `mrr`, `silhouette` line
  figures matching the `sampler_sweep_compare` evaluate experiment's `evaluation_id`s, plus
  `loss_curves.enabled: true`. No new renderer/plot pipeline code was needed: `plot` only ever reads
  `metrics.parquet` through a `select`/`x`/`hue`/`facet` config, so a new figure set is purely
  config, same as `neighborhood_compare.yaml`.

### Files modified or created

- `src/cga/pipelines/training_data/pipeline.py` — `build_sampler_id`, `sampler_id`-partitioned
  `write_latest` call
- `src/cga/pipelines/gcl_train/pipeline.py` — `LoadStep` reads `training_data_sampler_id`
- `src/cga/cli.py` — `training_data` branch of `_run_root` uses `build_sampler_id`; new shared
  `_run_pipeline` helper; new `cmd_sweep`; new `sweep` subparser; module docstring usage block
- `src/cga/core/sweep.py` (new) — `SweepSpec`, `SweepPoint`, `expand_sweep`, `sweep_point_hash`
- `src/cga/core/manifest.py` — `Manifest.sweep_point_id` field, included in `to_dict()`
- `configs/pipelines/gcl_train.yaml` — `inputs.training_data_sampler_id: rank_topk-k30`
- `configs/sweeps/rank_topk_k.yaml`, `configs/sweeps/percentile_pct.yaml` (new)
- `configs/experiments/sampler_sweep_compare.yaml` (new)
- `tests/unit/core/test_sweep.py` (new) — spec parsing, cross vs. zip expansion, deterministic ids,
  duplicate-id detection
- `tests/unit/pipelines/training_data/test_pipeline.py` — `build_sampler_id` unit tests, updated
  `_run_training_data`/`test_training_data_pipeline_writes_latest` for the new nested `LATEST` path,
  new `test_training_data_pipeline_partitions_latest_by_sampler_id` (two `k` values keep separate
  `LATEST`s)
- `tests/unit/pipelines/gcl_train/test_pipeline.py` — fixture updated for the nested
  `training_data` lineage path and the new `training_data_sampler_id` config key
- `tests/unit/cli/test_cli.py` — `test_sweep_runs_every_combination_and_partitions_latest`,
  `test_sweep_continues_past_a_failed_combination` (failure isolation: one bad `dataset.adapter`
  value in the grid doesn't stop the other point from running or writing `LATEST`)

### Challenges, ambiguity, assumptions

- **`training_data`'s `LATEST` nesting change is a breaking path change** for any existing artifacts
  under the old flat `artifacts/training_data/{dataset_id}/LATEST` — those are now orphaned relative
  to the new `{dataset_id}/{sampler_id}/LATEST` lookup. Nothing was migrated or deleted; a fresh
  `training_data` run under the new code writes to the new nested path immediately. Flagged here
  rather than silently handled, in case old artifacts need to be re-pointed or re-run.
- **`gcl_train.inputs.training_data_sampler_id`'s default (`rank_topk-k30`)** was chosen to match
  `training_data.yaml`'s own current default sampler config (`rank_topk`, `k: 30`), so an
  unmodified `gcl_train` run continues to resolve the same lineage it did before this change,
  with no behavior change for the existing single-sampler workflow.
- **Sweeping `training_data.sampler.name` as a single-value array** (`[rank_topk]` /
  `[percentile]`) in each spec, rather than omitting it and relying on `training_data.yaml`'s
  default, was a deliberate choice so each spec file is self-describing about which sampler it
  sweeps without having to open `configs/pipelines/training_data.yaml` to check.

### CLI commands to test and run current phase

```
uv run pytest tests/unit/core/test_sweep.py -q                          # 11 passed
uv run pytest tests/unit/pipelines/training_data -q
uv run pytest tests/unit/pipelines/gcl_train -q
uv run pytest tests/unit/cli -q
uv run pytest tests/unit -q                                             # full suite, 368 passed
```

Not yet run (next action item — real compute, not blocked): the actual sweeps and comparison —

```
uv run cga sweep --spec configs/sweeps/rank_topk_k.yaml
uv run cga sweep --spec configs/sweeps/percentile_pct.yaml

# Per sweep point (8x), fixed at GCN + SecRoBERTa + DeepWalk, 100 epochs:
uv run cga run gcl_train --dataset attack_cwe \
  --set gcl_train.inputs.training_data_sampler_id=<sweep_point_id> \
  --set gcl_train.model.model_id=gcn-<sweep_point_id> \
  --set gcl_train.train.epochs=100

uv run cga run evaluate --dataset attack_cwe --experiment sampler_sweep_compare
```
