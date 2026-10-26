# Sprint 2 — Sweep tooling and hyperparameter selection

## Objective

Add the minimal sweep tooling needed to run and compare configurations reproducibly, then use it to
select hyperparameters for the `attack_cwe` dataset — first the `training_data` sampler
(Phase 2.1), then `gcl_train`'s loss-mixing weight `alpha` (Phase 2.2). Phases are sequential: each
holds the previous phase's winner fixed. Design decisions and rationale are in
[dev_docs/sweep_experiment_plan.md](../sweep_experiment_plan.md); this sprint implements that plan.

## Conventions for every phase

- Google-style docstrings on every module, class, and public function (per `CLAUDE.md`).
- Unit tests live in `tests/unit/`, mirroring the package layout.
- Each phase ends with an entry in `dev_docs/execution_logs/sprint2.md`.
- Run tests with `uv run pytest tests/unit -q`.

---

## Phase 2.1 — Sampler sweep infrastructure and comparison

- [x] Partition `training_data`'s `LATEST` by a `sampler_id` (mirrors `gcl_train`'s `model_id`
      pattern), so sweep points don't clobber each other's lineage.
- [x] Add `cga sweep --spec <file>`: expand a YAML spec (`sweep:` dotted-path → value-array map,
      optional `zip:` for paired axes) into per-combination `cga run` calls over the existing
      `build_config → RunContext.create → pipeline.run` path; tag each run with a `sweep_point_id`
      in its manifest; continue past a failed combination instead of aborting.
- [x] Add tests for the sweep expansion (cross vs. `zip`, deterministic `sweep_point_id`, failure
      isolation) and the `LATEST` partitioning.
- [x] Write `configs/sweeps/rank_topk_k.yaml` (`k: [5, 10, 20, 30]`) and
      `configs/sweeps/percentile_pct.yaml` (zipped `top_pct`/`bottom_pct: [0.02, 0.05, 0.08, 0.1]`). One Clarification, the model train choose either topk_k sampling  or percentile sampling. Not both. 
- [x] Run both sweeps against `attack_cwe`, fixed at GCN + SecRoBERTa + DeepWalk, 100 epochs each.
- [x] Add `configs/experiments/sampler_sweep_compare.yaml` and run `evaluate` over all 8 sweep
      points as methods.
- [x] Produce the reporting table (`mrr_textual`, `precision@10_textual`, `silhouette_mean`,
      `final_loss`, `n_examples` per sweep point) and identify the best sampler config. See
      `dev_docs/execution_logs/sprint2.md` — best config is `rank_topk`, `k=5`.

---

## Phase 2.2 — `alpha` sweep (text/graph loss weight)

`TrainStep` mixes the two loss terms as `alpha * L_text + (1 - alpha) * L_graph`
([gcl_train/pipeline.py:275](../../src/cga/pipelines/gcl_train/pipeline.py#L275)), with `alpha`
read from `gcl_train.loss.alpha` (default `0.5`). Phase 2.1 fixed `alpha` at that default while
sweeping the sampler; this phase sweeps `alpha` with the sampler held at Phase 2.1's winner
(`rank_topk`, `k=5`). The endpoints are ablations: `alpha=0.0` is graph-loss-only, `alpha=1.0` is
text-loss-only, so the sweep answers whether the dual objective beats either tower alone rather
than only locating an optimum.

`cga sweep` itself needed one small extension: a `fixed:` block on the spec (see below).

- [x] Write `configs/sweeps/alpha.yaml`: `pipeline: gcl_train`, `dataset: attack_cwe`,
      `gcl_train.loss.alpha: [0.0, 0.25, 0.5, 0.75, 1.0]`.
- [x] In the same spec, `zip:` `gcl_train.model.model_id` against `alpha` with explicit ids
      (`gcn-alpha0`, `gcn-alpha0.25`, …) so each point keeps its own
      `models/gcl/{dataset_id}/{model_id}/LATEST`. Without this, `build_model_id` returns the bare
      encoder (`gcn`) for all five points and they clobber each other's `LATEST` — the same
      collision `sampler_id` fixed for `training_data` in Phase 2.1. Decision: keep the ids in the
      spec rather than teaching `build_model_id` about `alpha`, so the existing runs' id scheme is
      untouched.
- [x] Pin the spec's non-swept `gcl_train` inputs to Phase 2.1's winner:
      `gcl_train.inputs.training_data_sampler_id: rank_topk-k5` (the pipeline config's default is
      still `rank_topk-k30`). `cga sweep` layers only `base + dataset + pipeline` — there is no
      `--experiment` layer and no `--set` flag — but the text/graph embedding choice (SecRoBERTa +
      DeepWalk) is resolved from the `training_data` run's own manifest in `LoadStep`, so pinning
      the `sampler_id` is sufficient to hold the embedding combo fixed. Added a `fixed:` block to
      `SweepSpec`/`expand_sweep` (`src/cga/core/sweep.py`) for this: putting
      `training_data_sampler_id`/`epochs` in `sweep:` instead would work (a one-element array
      crossed against everything) but bloats `sweep_point_id` with values that never vary, e.g.
      `alpha0-model_idgcn-alpha0-training_data_sampler_idrank_topk-k5-epochs100`. `fixed:` values
      merge into every point's overrides after `sweep_point_id` is computed from the varying keys
      alone, so ids stay short (`alpha0-model_idgcn-alpha0`); a `fixed:`/`sweep:` key overlap
      raises `ConfigError`. Added tests in `tests/unit/core/test_sweep.py`.
- [x] Confirm `sweep_point_id` renders the zipped pair readably before launching (dry-checked the
      expansion via `build_config` over all 5 points): `_short_value` formats floats with `%g`, so
      `0.0` → `0` and `1.0` → `1` — the spec's `model_id` values (`gcn-alpha0`, `gcn-alpha1`, not
      `gcn-alpha0.0`/`gcn-alpha1.0`) already match this, confirmed by dry-run rather than assumed.
      Phase 2.1 hit exactly this class of mismatch.
- [ ] Run the sweep: 5 `gcl_train` runs at GCN + SecRoBERTa + DeepWalk, 100 epochs each (matching
      Phase 2.1's budget so the two phases' `final_loss` values stay comparable). **Not yet run —
      real compute, next action item.**
- [x] Add `configs/experiments/alpha_sweep_compare.yaml` (5 `gcl_run` methods, one per `alpha`) and
      `configs/plots/alpha_sweep_compare.yaml`.
- [ ] Run `evaluate` then `plot` over all 5 points, produce the reporting table (`mrr_textual`,
      `precision@10_textual`, `silhouette_mean`, `final_loss` per `alpha`) and identify the best
      `alpha`. Selection metric is `mrr_textual`, same as Phase 2.1; `final_loss` is not comparable
      across `alpha` (the objective itself changes with `alpha`, so a lower loss at `alpha=0` means
      only that `L_graph` is easier to minimize, not that the model is better) — record it, but do
      not select on it. **Blocked on the sweep above.**
- [x] Write the phase up in `dev_docs/execution_logs/sprint2.md`.
