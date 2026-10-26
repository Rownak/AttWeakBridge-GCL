# Sprint 1 Phase 1.10a — Per-run training loss curves

## What was implemented

Added to the `plot` pipeline: one loss-vs-epoch figure per `gcl_train` run, read directly from that
run's `metrics.jsonl` (§6.8: `{epoch, loss, loss_text, loss_graph, lr, elapsed_s}` per epoch) rather
than from `metrics.parquet`. Per the developer, this is not a cross-run comparison — every `model_id`
under the dataset's `models/gcl/{dataset_id}/` tree is discovered automatically and gets its own PNG,
independent of the `evaluate`/`metrics.parquet`-driven figures already in `plot`. Motivated by having
just finished training all 8 `{gcn,gat} x {secroberta,securebert2} x {deepwalk,node2vec}` combinations
to 400 epochs each and wanting to see each one's convergence.

## How it was completed

- **`pipelines/plot/loss_curves.py`** (new) — `LossCurve` (epoch/loss/loss_text/loss_graph lists),
  `read_loss_curve(run_dir)` (parses `metrics.jsonl` line by line), `render_loss_curve(curve,
  output_path, *, title)` (three lines — combined/graph/text — on one log-scale-y axes; log scale
  because loss spans about an order of magnitude in the first ~50 epochs and then flattens, so a
  linear axis buries the interesting early dynamics), and `discover_gcl_runs(gcl_dataset_dir)`
  (iterates `model_id` subdirectories, reads each one's `LATEST` via the existing `read_latest`
  helper, skips a `model_id` with no `LATEST` or whose resolved run has no `metrics.jsonl`). Kept
  fully independent of `renderers.py`'s `FigureSpec`/`select` machinery — there is no
  `metrics.parquet` row to select, since training losses were deliberately excluded from
  `evaluate`'s output back in Phase 1.7/1.8 (§6.8: "Authoritative metrics live in the `evaluate`
  run's `metrics.json`, not here").
- **`pipelines/plot/pipeline.py`** — new `LossCurvesStep`, inserted between `render` and
  `validate_and_manifest`. Gated by `plot.loss_curves.enabled` (default `false`); when enabled, it
  builds `models/gcl/{dataset_id}/`, calls `discover_gcl_runs`, and for every discovered run resolves
  it through `ctx.resolve(f"gcl_run:{model_id}", ...)` (so it lands in `manifest.inputs[]` with its
  concrete run_id, same traceability contract as `evaluate.methods`) before rendering
  `figures/loss_curve_{model_id}.{ext}`. `LoadStep` was loosened: the existing "empty `figures` is a
  `ConfigError`" check now only fires when `loss_curves.enabled` is *also* false, so a loss-curves-only
  run (no `evaluate` run needed at all) is a legitimate, minimal invocation rather than an error.
  `ValidateAndManifestStep`'s summary gained `loss_curve_model_ids`.
- **`configs/pipelines/plot.yaml`** — added `loss_curves: {enabled: false}`.
  **`configs/plots/neighborhood_compare.yaml`** — set `loss_curves.enabled: true`, so the paper run
  now also emits every discovered `gcl_train` run's loss curve alongside the existing
  `precision_at_k`/`silhouette` figures.

## Files modified or created

- `src/cga/pipelines/plot/loss_curves.py` (new)
- `src/cga/pipelines/plot/pipeline.py` — `LossCurvesStep`, loosened `LoadStep` empty-figures check,
  `loss_curve_model_ids` in the manifest summary
- `configs/pipelines/plot.yaml`, `configs/plots/neighborhood_compare.yaml`
- `tests/unit/pipelines/plot/test_loss_curves.py` (new), `test_pipeline.py` (extended)
- `dev_docs/sprints/sprint_1.md` — Phase 1.10a checkboxes marked complete

## Challenges, ambiguity, assumptions

- **Confirmed with the developer before implementing** (not guessed): (1) figures are per-run only,
  never a cross-run comparison, so no config-driven run *selection* was needed — just discovery; (2)
  this belongs as a new step inside the existing `plot` pipeline rather than a separate pipeline or a
  retrofit of `evaluate`'s `metrics.parquet` schema, since training losses genuinely aren't evaluation
  metrics.
- **Real discovery found 10 runs, not 8** — alongside the 8 requested `{gcn,gat} x {secroberta,
  securebert2} x {deepwalk,node2vec}` combinations, two older standalone `gcn`/`gat` runs from Phase
  1.9's verification are still live under `models/gcl/attack_cwe/{gcn,gat}/LATEST`. This is correct,
  not a bug: `discover_gcl_runs` finds every `model_id` with a successful run, by design, and those
  two runs are genuine (if superseded) trained models with their own `metrics.jsonl`. Nothing was
  deleted — removing artifacts wasn't asked for, consistent with the note left in Phase 1.9's log
  about the analogous stale flat `LATEST`.
- **The stale flat `models/gcl/{dataset}/LATEST` file** (pre-model_id-partition, noted in Phase 1.9's
  log) is correctly ignored by `discover_gcl_runs`: it iterates directories under
  `gcl_dataset_dir`, and a loose `LATEST` file there is not itself a directory, so it's skipped
  without special-casing.

## CLI commands to test and run

```
uv run pytest tests/unit/pipelines/plot -q                    # 26 passed
uv run pytest tests/unit -q                                    # full suite, 351 passed

uv run cga run plot --dataset attack_cwe --config configs/plots/neighborhood_compare.yaml
uv run cga inspect artifacts/plots/attack_cwe/<run_id>
```

Verified end-to-end against the real trained `gcl_train` runs: writes 10
`figures/loss_curve_{model_id}.png` files (8 requested combinations + 2 older standalone `gcn`/`gat`
runs), each a combined/graph/text loss curve over 400 epochs on a log-scale y-axis, decreasing
sharply in the first ~50 epochs and flattening thereafter; `cga inspect` lists all 10
`gcl_run:{model_id}` manifest inputs alongside the existing `evaluate_run` input, each traceable back
through its own `training_data`/`embed`/`ingest` chain to raw file hashes.
