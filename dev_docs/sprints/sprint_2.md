# Sprint 2 — Sampler sweep and comparison tooling

## Objective

Determine the best `training_data` sampler configuration (`rank_topk.k` or
`percentile.{top_pct,bottom_pct}`) for the `attack_cwe` dataset, and add the minimal sweep tooling
needed to run and compare it reproducibly. Design decisions and rationale are in
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
- [ ] Run both sweeps against `attack_cwe`, fixed at GCN + SecRoBERTa + DeepWalk, 100 epochs each.
- [ ] Add `configs/experiments/sampler_sweep_compare.yaml` and run `evaluate` over all 8 sweep
      points as methods.
- [ ] Produce the reporting table (`mrr_textual`, `precision@10_textual`, `silhouette_mean`,
      `final_loss`, `n_examples` per sweep point) and identify the best sampler config.
