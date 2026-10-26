# Sprint 1 Phase 1.10 — `plot` pipeline

## What was implemented

A sixth, thin pipeline that turns an `evaluate` run's long-form `metrics.parquet` (§6.10) into
figures, kept separate from `evaluate` so re-plotting never re-scores and one evaluation run can be
plotted several ways. A **figure spec** selects a slice of the metrics rows and maps columns to plot
roles (`x`/`hue`/`facet`); it names no evaluation kind, so precision-vs-`k` and silhouette-vs-
`n_clusters` are the same `line` renderer against different `select` filters. `evaluate.visualize`
(t-SNE) was untouched — it projects an embedding matrix, not `metrics.parquet`.

## How it was completed

- **`pipelines/plot/renderers.py`** (new) — `renderers` registry and a pydantic `FigureSpec`
  (`figure_id`, `kind`, `select`, `x`, `hue`, `facet`, `params`). `apply_select` filters
  `metrics.parquet` by equality on the `select` dict and raises `ConfigError` on a zero-row result —
  a blank PNG is not an acceptable output, per §5.6. Two renderers registered: `line` (one line per
  `hue` value, one subplot per facet-group) and `bar` (grouped bars per `x`, colored by `hue`), the
  second existing specifically to prove the registry is a real extension point rather than a single
  hardcoded path. `_facet_groups` treats a facet column that is *entirely* null for the current
  selection (e.g. `view_id` on `cluster_quality` rows, which have no view) as absent rather than
  producing one degenerate all-null group — this is what let the same renderer handle both figures in
  `neighborhood_compare.yaml` without special-casing `cluster_quality`.
- **`pipelines/plot/pipeline.py`** (new) — three steps: `load` (validate every `FigureSpec` up front
  — duplicate `figure_id` or unknown `kind` both fail before any I/O — resolve `inputs.evaluate_run`
  via `ctx.resolve`, read and schema-validate `metrics.parquet`, apply every figure's `select` so an
  empty selection is caught here rather than mid-render) → `render` (registry lookup per
  `figures[]` entry, one PNG under `{run_dir}/figures/{figure_id}.{ext}`) → `validate_and_manifest`
  (records every rendered figure as a manifest output, `LATEST`). Registered as `"plot"` in the
  `pipelines` registry.
- **`cli.py::_run_root`** — added a `plot` branch: `artifacts/plots/{dataset_id}` (mirrors the
  `evaluate` branch immediately above it).
- **`configs/pipelines/plot.yaml`** (new) — thin defaults per §7.12 (`inputs.evaluate_run: latest`,
  empty `figures`, `format: {ext: png, dpi: 200}`). **`configs/plots/neighborhood_compare.yaml`**
  (new) — the paper's two figures: `precision_at_k` (`select: {evaluation_id: neigh, metric_name:
  normalized_precision_at_k}`, faceted by `[view_id, graph_role]`) and `silhouette` (`select:
  {evaluation_id: clusters, metric_name: silhouette}`, faceted by `[graph_role]`). This lives under a
  new `configs/plots/` directory rather than as a block inside an `configs/experiments/*.yaml`
  overlay, per §7.12: a plot spec references only `evaluation_id`/`metric_name`, so it stays reusable
  across datasets and experiments, and is selected with `--config` rather than `--experiment`.

## Files modified or created

- `src/cga/pipelines/plot/renderers.py`, `pipeline.py`, `__init__.py` (new package)
- `src/cga/pipelines/__init__.py` — registers the `plot` package
- `src/cga/cli.py` — `_run_root` resolves `plot`'s output root
- `configs/pipelines/plot.yaml` (new), `configs/plots/neighborhood_compare.yaml` (new)
- `tests/unit/pipelines/plot/__init__.py`, `test_renderers.py`, `test_pipeline.py` (new)
- `dev_docs/sprints/sprint_1.md` — Phase 1.10 checkboxes marked complete

## Challenges, ambiguity, assumptions

- **Faceting on a null column** was the one genuinely fiddly part: `cluster_quality` rows have
  `view_id = null`, and the default `facet: [view_id, graph_role]` would, applied naively, group rows
  by `(None, role)` — one group, but a visually broken subplot title. `_facet_groups` instead drops a
  facet column from consideration when it is null for *every* row in the current selection, so a
  figure selecting only `cluster_quality` rows facets cleanly on `graph_role` alone even though the
  default facet list still names `view_id`.
- **`configs/plots/` vs. a block inside `configs/experiments/`**: the architecture doc is explicit
  that these must be separate directories (§7.12's own rationale: reusability across datasets/
  experiments), so this was not a judgment call — implemented as specified.
- No new schema was needed: `plot` only reads the existing `METRICS_SCHEMA` (§6.10), written by
  Phase 1.9's `evaluate` pipeline, and writes no parquet/JSON of its own — only PNGs plus the
  standard `manifest.json`.

## CLI commands to test and run

```
uv run pytest tests/unit/pipelines/plot -q                    # 14 passed
uv run pytest tests/unit -q                                    # full suite, 339 passed

uv run cga run plot --dataset attack_cwe --config configs/plots/neighborhood_compare.yaml
uv run cga inspect artifacts/plots/attack_cwe/<run_id>
```

Verified end-to-end against the real Phase 1.9 `evaluate` run (`metrics.parquet`, 424 rows): both
`figures/precision_at_k.png` and `figures/silhouette.png` were written under
`artifacts/plots/attack_cwe/{run_id}/figures/`; `cga inspect` traces the `plot` run back through
`evaluate` to every method (`deepwalk`/`secroberta`/`gcn`/`gat`) and the `textual` reference, and
further back through `gcl_train`/`training_data`/`embed`/`ingest` to the raw source files;
re-running the same spec against the same `evaluate` run resolved the identical `metrics.parquet`
(same `latest` pointer), confirming re-plotting never re-scores.
