# Sampler Sweep — Experiment Design

## Objective

Compare `training_data` sampler parameters (`rank_topk.k`, `percentile.top_pct`/`bottom_pct`) to
find the best downstream GCL alignment, holding the embedding/architecture combo fixed at
**GCN, SecRoBERTa (text) + DeepWalk (graph)**.

Planning only — nothing implemented yet. See
[samplers.py](../src/cga/pipelines/training_data/samplers.py),
[training_data/pipeline.py](../src/cga/pipelines/training_data/pipeline.py), and
[target_architecture.md §5.3](target_architecture.md).

---

## 1. What to sweep

| Sampler | Sweep parameters | Notes |
|---|---|---|
| `rank_topk` | `k` | Rank-zipped pairing keeps noise low even at small `k`. |
| `percentile` | `top_pct`, `bottom_pct` | Interact — sweep as a paired axis, not independently. |

`supervised` is out of scope: it needs `alignments.parquet`, which `attack_cwe` doesn't have, so
it no-ops there.

### Ranges

Anchor to what's structurally valid, not round numbers:
[`_topk_bottomk_indices`](../src/cga/pipelines/training_data/samplers.py#L42) clamps
`k = min(k, len(order))`, so **check `stats.json`'s per-`graph_role` train-pool counts first** and
keep the sweep below that size — a `k` past the pool size silently clamps and duplicates a cell.

- `rank_topk.k ∈ [5, 10, 20, 30, 50]`
- `percentile.{top_pct, bottom_pct} ∈ {0.02, 0.04, 0.08}` (paired, equal to each other)
- `k` and `(top_pct, bottom_pct)` aren't directly comparable — pick each sampler's own best cell,
  then compare the two winners.

8 `training_data` runs total (5 + 3), each fanning out to one `gcl_train` (GCN,
SecRoBERTa+DeepWalk, 400 epochs) and one `evaluate` run.

---

## 2. Sweep mechanism: `cga sweep`

**Decision: a thin `cga sweep` command reading an array-valued spec, not one config file per
value.** One-file-per-value is not good MLOps practice — it hides the grid across N near-identical
files with no single place showing what was swept. Rejected alternatives:

- *Shell loop over `--set`*: no new code, but the value array lives in a throwaway script, and
  leaves `LATEST`-collision bookkeeping (below) manual.
- *Wait for Hydra `--multirun`*: the eventual right home for this (§11 of the architecture doc), but
  adopting it well means reworking `core/config.py`'s single-config-in/out contract — a
  project-sized integration not worth blocking this sweep on.

`cga sweep` borrows Hydra's `key=v1,v2,v3` override shape on purpose, so migrating later changes
only the expansion engine, not the spec files.

### Spec format

```yaml
# configs/sweeps/rank_topk_k.yaml
pipeline: training_data
dataset: attack_cwe
sweep:
  training_data.sampler.name: [rank_topk]
  training_data.sampler.k: [5, 10, 20, 30, 50]
```

```yaml
# configs/sweeps/percentile_pct.yaml
pipeline: training_data
dataset: attack_cwe
sweep:
  training_data.sampler.name: [percentile]
  zip: [training_data.sampler.top_pct, training_data.sampler.bottom_pct]   # paired by index
  training_data.sampler.top_pct: [0.02, 0.04, 0.08]
  training_data.sampler.bottom_pct: [0.02, 0.04, 0.08]
```

Keys are dotted config paths (same shape as `--set`), crossed by default; `zip:` marks a set of
keys as paired-by-index instead of crossed.

### What it does

```bash
uv run cga sweep --spec configs/sweeps/rank_topk_k.yaml
```

Per expanded combination: build its `--set` list → compute a short `sweep_point_id` from the swept
values (e.g. `rank_topk-k20`) → run the existing `build_config → RunContext.create → pipeline.run`
path, passing `sweep_point_id` through so `write_latest` partitions by it → record
`sweep_point_id` + swept values in that run's `manifest.json` → continue past a failed combination
(log it, don't abort the grid).

No change to `core/config.py`; `cga sweep` is a CLI-layer loop over the existing per-run path.

### Prerequisite: partition `training_data`'s `LATEST` by sweep point

`artifacts/training_data/{dataset_id}/LATEST` is currently flat — 8 sweep points would clobber
each other the same way the encoder grid's `gcl_train` runs did before `model_id`-partitioning.
Add a `sampler_id`-partitioned `LATEST` (mirroring
[`build_model_id`](../src/cga/pipelines/gcl_train/pipeline.py#L44)), computed from
`{sampler.name}-{key params}`. Small, surgical change to `cli._run_root` and
`TrainingDataPipeline.run`'s `write_latest` call — required before `cga sweep` can run this
pipeline without collisions.

### Migrating to Hydra later

`sweep:`'s key→values maps directly onto Hydra's `key=v1,v2,v3` multirun syntax; `zip:` has a
direct equivalent in a custom sweeper if still needed. `cga sweep` is scoped to disappear into
`cga run --multirun` with no changes to `configs/sweeps/*.yaml`.

---

## 3. Comparison summary

Run `evaluate` with all 8 sweep-point models as **methods** in one
`configs/experiments/sampler_sweep_compare.yaml`, same shape as
[`full_grid_compare.yaml`](../configs/experiments/full_grid_compare.yaml) — one `method_id` per
sweep point (e.g. `gcn_rank_topk_k20`). One `metrics.parquet`, pivoted on `metric_value` with
`method_label` as rows, is the comparison table; no separate ad hoc CSV.

---

## 4. Metrics that drive sampler selection

- **`normalized_precision_at_k` / `mrr` on the `textual` view** — primary selection metric; scored
  against a reference independent of the sampler.
- **`cluster_quality` (silhouette)** — sanity check only, flags degenerate embeddings; not a
  primary criterion.
- **Structural views (`structural_r1`/`r2`)** — regression check only. Trained models already trade
  away structural fidelity vs. raw DeepWalk; don't judge the sampler on recovering that.
- **`gcl_train`'s `metrics.jsonl`** (final loss, monotonicity) — cheap pre-filter to discard broken
  cells before spending an `evaluate` run on them, not a selection metric itself.

**Report**: one table, rows = sweep point, columns = `{mrr_textual, precision@10_textual,
silhouette_mean, final_loss, n_examples}`, sorted by `mrr_textual`. Direct `metrics.parquet` pivot
joined against `training_data`'s `stats.json` and `gcl_train`'s `metrics.jsonl` tail.

---

## 5. MLflow / Hydra / DVC timing

**Run the sweep now; don't wait.** §11 of the architecture doc exists so this doesn't gate on those
integrations — `RunContext`/manifests already track runs, `configs/` is already Hydra-shaped,
immutable run dirs are already DVC-shaped. Integrating those tools later is a swap of
`core/config.resolve()` plus an added tracker, not a rework of how sweeps are expressed. Waiting
risks scope creep: MLflow/Hydra/DVC integration is its own project-sized task that shouldn't block
a time-sensitive comparison whose conclusions it won't change.

**Do first**: the two §2 prerequisites — the `sampler_id`-partitioned `LATEST`, and `cga sweep`
itself. Both are small, both are exactly the seams §11 describes (no filename-encoded params, no
pickle, pure `RunContext`/manifest usage), and both fold into Hydra later with no spec-file
changes.

---

## Open questions / next iteration

- [ ] Confirm `k` / `top_pct` candidates against real `stats.json` train-pool counts before locking
      the grid.
- [ ] Implement, in order: (1) `sampler_id`-partitioned `LATEST`, (2) `cga sweep`.
- [ ] Pin down `cga sweep`'s CLI surface: flag name (`--spec` assumed), whether
      `--experiment`/`--config` compose with a sweep spec, and the `ArtifactRef` syntax for a
      downstream pipeline to reference one sweep point's `LATEST`.
