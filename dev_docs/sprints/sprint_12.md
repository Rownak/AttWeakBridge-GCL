# Sprint 12 — Paper experiments on `attack_cwe_ent` (mpnet-chunk + DeepWalk, A0 / A1 / A2)

## Objective

Produce the paper's comparison on Enterprise ATT&CK × CWE (`attack_cwe_ent`) with the Sprint 3 end-to-end flow
(`ingest → embed → training_data → gcl_train → evaluate → plot`) and the Sprint 3 evaluation protocol: gold-free
proxy metrics (`neighborhood` and `cluster_quality`). Every upstream artifact already exists from Sprint 11 and is
reused by pinned run id; nothing is retrained. The sprint only verifies the artifacts, evaluates, plots, and packages
one table and the figures. Phases 12.5–12.6 then add the gated residual graph adapter
(`gcl_method_update.md` §3) as new trained rows evaluated the same way.

Background: [sprint_3.md](sprint_3.md) (flow and protocol), [sprint_11.md](sprint_11.md) and its execution logs
(artifacts; 11.11 for A0, 11.12 for A1 / A2).

## Decisions (confirmed with the developer)

| Item | Value |
|---|---|
| Dataset id | `attack_cwe_ent` (`configs/datasets/attack_cwe_ent.yaml`, raw data `data/raw/enterprise_cwe/`); no new dataset config |
| Text embeddings | mpnet-chunk: `sentence_transformer-multi_qa_mpnet_dot_chunk_desc` (`description_only` text, token-weighted chunk mean) |
| Graph embeddings | `deepwalk-deepwalk` |
| Rows | mpnet-chunk (frozen), DeepWalk (frozen), A0 full dual GCN, A1 text-only GCN, A2 graph-only GCN |
| Pseudo-pairs | `rank_topk`, `k = 5` (top-5 positives, bottom-5 negatives; `negatives: bottomk`, the default) |
| Loss | `infonce` (temperature 0.1), `alpha 0.5` |
| Models | reused from Sprint 11, **not retrained**; seed 42 only (Phases 12.1–12.4); Phases 12.5–12.6 train new gated models (seeds 42–44 for G2 / G3 / G5) |
| Evaluation | Sprint 3 proxy protocol only: `neighborhood` (views `structural_r1`, `structural_r2`, `textual`; `ks [1, 5, 10, 20]`; roles `graph_1`, `graph_2`) and `cluster_quality` (`n_clusters [2, 4, 8, 16, 32]`). No gold metrics in this sprint |
| `textual` view reference | `hf_mean_pool-pt_SecRoBERTa_desc` (not a competitor; same `description_only` text) |

## Reused artifacts (pinned; no `latest` anywhere)

| Artifact | Path / id | Run | From |
|---|---|---|---|
| Canonical data (graph for `hop_radius`) | `data/processed/attack_cwe_ent/` | `20261027T194643Z-5d3f8baf` (union, `description_only`) | 11.10 |
| mpnet-chunk store | `artifacts/embeddings/attack_cwe_ent/sentence_transformer-multi_qa_mpnet_dot_chunk_desc/` | `20261027T194928Z-8cf38e55` | 11.10 |
| DeepWalk store | `artifacts/embeddings/attack_cwe_ent/deepwalk-deepwalk/` | `20261027T135210Z-3b18ba0c` | 11.6 |
| Textual reference store | `artifacts/embeddings/attack_cwe_ent/hf_mean_pool-pt_SecRoBERTa_desc/` | `20261027T200841Z-09088f9f` | 11.10 |
| Pseudo-pairs | `artifacts/training_data/attack_cwe_ent/s11c-rank_topk-k5/` | `20261027T205523Z-ac890ca3` | 11.11 |
| A0 (`branch: both`) | `models/gcl/attack_cwe_ent/ent-dual-gcn-infonce-mpnetc-s42/` | `20261027T205648Z-2aeb3259` | 11.11 |
| A1 (`branch: text`) | `models/gcl/attack_cwe_ent/ent-abl-text-gcn-nce-mpnetc-s42/` | `20261027T232407Z-adc619c8` | 11.12 |
| A2 (`branch: graph`) | `models/gcl/attack_cwe_ent/ent-abl-graph-gcn-nce-mpnetc-s42/` | `20261027T232541Z-d91ebd8e` | 11.12 |

Fallback only if an artifact is missing or fails Phase 12.1: regenerate it with the command recorded in the Sprint 11
log that produced it (11.10 for ingest / stores, 11.11 for the pseudo-pairs, 11.11 / 11.12 for the models), pin the new
run id here, and say so in the log. Nothing is regenerated otherwise.

## Conventions

- Every `--dataset` is `attack_cwe_ent`. Commit messages are prefixed `[Sprint-12.N]`.
- Google-style docstrings for any new code; unit tests in `tests/unit/`; `uv run pytest tests/unit -q` passes.
- Each phase ends by updating `dev_docs/execution_logs/sprint_12.md` (what was done, run ids, checks, commands).

---

## Phase 12.1 — Verify the reused artifacts

- [x] Each run directory in the table exists with its `manifest.json` (status succeeded) and outputs.
- [x] Stores: mpnet-chunk, DeepWalk and the SecRoBERTa reference each cover all 2,927 universe nodes (the pseudo-pair
      universe) with finite vectors; mpnet-chunk 768-d, DeepWalk 128-d.
- [x] Pseudo-pairs: manifest lists the pinned ingest, mpnet-chunk and DeepWalk runs; sampler `rank_topk`, `k = 5`,
      default `negatives` (`bottomk`); 2,927 nodes, 14,635 examples, 0 held out, seed 42.
- [x] Models: each `config.resolved.yaml` shows `encoder: gcn`, the expected `branch` (A0 has none, i.e. `both`),
      `loss.name: infonce`, temperature 0.1, `alpha 0.5`, hidden / out 128, 100 epochs, seed 42, and
      `training_data_run: 20261027T205523Z-ac890ca3`; `node_embeddings` is 2,927 × 128 and finite.
- [x] The DeepWalk store was built from the 11.6 ingest; confirm the 11.10 ingest has the same node and edge id sets
      (checked in 11.10; re-check, since `hop_radius` reads the 11.10 graph).

**Acceptance:** every check passes with no regeneration, or each regenerated artifact is pinned and explained.

## Phase 12.2 — A0 / A1 / A2 runs (reused, no training)

- [x] Record per model, from `metrics.jsonl`: 100 rows, loss at epoch 1 and 100 (lower at 100), no NaN; trainable
      parameters from the manifest (328,576 / 246,272 / 82,432).
- [x] Record the exact configuration of each run (model id, run id, branch, sampler id, training data run, loss and
      parameters) in the log; this is the paper's run table.
- [x] Reproducibility statement: A0 and A1 have bit-identical retrains (11.12 gate `20261027T232219Z-0bded1cb`, A1
      rerun `20261027T233417Z-adc619c8`); cite them, do not retrain.

**Acceptance:** all three runs are sane and fully specified; nothing is trained in this sprint.

## Phase 12.3 — Evaluate and plot (Sprint 3 protocol)

- [x] `configs/experiments/attack_cwe_ent_paper_compare.yaml` (modelled on `ncit_doid_neighborhood_compare.yaml`):
      5 methods pinned by run id — `mpnet-chunk`, `deepwalk` (`embedding_store`), `a0`, `a1`, `a2` (`gcl_run`);
      evaluation `neigh` (`neighborhood`, `ks [1, 5, 10, 20]`, roles `graph_1` / `graph_2`, views `structural_r1`
      (`hop_radius`, radius 1), `structural_r2` (radius 2), `textual` (`cosine_top_k`, `top_k 10`, reference the pinned
      SecRoBERTa store)); evaluation `clusters` (`cluster_quality`, `n_clusters [2, 4, 8, 16, 32]`);
      `visualize: {enabled: true, methods: [a0, a1, a2]}`.
- [x] `configs/plots/attack_cwe_ent_paper_compare.yaml` (modelled on `ncit_doid_neighborhood_compare.yaml`):
      `precision_at_k` line figure (`neigh`, `normalized_precision_at_k`, x `param_value`, hue `method_label`, facet
      `[view_id, graph_role]`), `silhouette` line figure (`clusters`, facet `[graph_role]`), `loss_curves` enabled,
      `plot.inputs.evaluate_run` pinned to the Phase 12.3 evaluate run.
- [x] Run, then repeat once:
      ```
      uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_paper_compare \
        --set evaluate.inputs.ingest_run=20261027T194643Z-5d3f8baf
      uv run cga run plot --dataset attack_cwe_ent --config configs/plots/attack_cwe_ent_paper_compare.yaml
      ```
      Outputs: `artifacts/evaluations/attack_cwe_ent/<evaluate run>/` (`metrics.parquet`, embedding projections),
      `artifacts/plots/attack_cwe_ent/<plot run>/` (PNGs).

**Acceptance:** the repeat evaluate run has identical `metrics.parquet` values; every method is scored on the same
query set per (view, role) (the evaluator intersects candidates across methods; record the query counts); the two
figures and the embedding projections for A0 / A1 / A2 exist. `loss_curves` renders one PNG per `model_id` under the
dataset (it cannot be filtered without a code change); only the A0 / A1 / A2 curves go into the paper package.

## Phase 12.4 — Paper table and package

- [x] One comparison table, built by a small script `scripts/paper_proxy_table.py` (reads the evaluate run's
      `metrics.parquet`; writes `paper_table.md` and `paper_table.csv` next to it). Rows: the 5 methods. Columns, as in
      Sprint 3: MRR and normalized precision@10 for `structural_r1`, `structural_r2` and `textual`, each per graph
      (ATT&CK = `graph_1`, CWE = `graph_2`), and the silhouette mean over `n_clusters` 2–32 and both graphs. Best per
      column in bold. Unit test on a hand-built `metrics.parquet` fixture.
- [x] Package for the paper: `artifacts/paper/attack_cwe_ent/sprint_12/` with `paper_table.md` / `.csv`, the two
      figures, the A0 / A1 / A2 loss curves and projections, and `runs.yaml` (every pinned run id from this file, the
      evaluate and plot run ids, and the exact commands).
- [x] Short reading in the log (no tuning): which rows lead on structural views, on the textual view and on
      silhouette; where A0 / A1 / A2 sit between the two frozen baselines.
- [x] Caveats in the log, carried into the paper text: proxy relevance sets, not gold labels; neighborhoods are scored
      within each graph, so these metrics do not measure ATT&CK → CWE alignment; the `textual` reference
      (SecRoBERTa) is itself a text encoder, so text-based rows have an advantage there; single seed.

**Acceptance:** the table regenerates byte-identically from the evaluate run with one command; the package folder holds
every figure and run id referenced in the log.

## Phase 12.5 — Gated residual graph adapter over frozen text (implement and train)

Source idea: [gcl_method_update.md](../design_docs/gcl_method_update.md) (§3, gated residual graph adapter; background in
[sapbert_residual_graph_adapter.md](../design_docs/sapbert_residual_graph_adapter.md)). The model starts at frozen
mpnet-chunk text and adds graph information only where it lowers the loss. Model code only: pseudo-pair data, loss and
evaluators are unchanged; nothing from Phases 12.1–12.4 is retrained.

**Model (`encoder: residual_gated`, `gcl_train`).** `x` = frozen mpnet-chunk vector (768-d, detached), `Â` = the GCN
normalised adjacency with self-loops over the same intra-graph `edge_index` as `gcn`, `s` = DeepWalk vector (128-d):

```
text variant:   z_i = LN( x_i + g_i ⊙ f(Â X)_i )          f = Linear(768, r) – ReLU – Linear(r, 768),  r = 128
                g_i = σ( W_g [ x_i ‖ (Â X)_i ] + b_g )     W_g: Linear(1536, 768), b_g initialised to −4 (g ≈ 0.018)
dual variant:   z_i = LN( x_i + g^t_i ⊙ f_t(Â X)_i + g^s_i ⊙ f_s(Â S)_i )
                g^s_i = σ( W_s [ x_i ‖ (Â S)_i ] + b_s ),  f_s = Linear(128, r) – ReLU – Linear(r, 768)
```

`LN` = `LayerNorm(768)` without affine parameters. The last layer of every `f` starts at std 1e-3 / zero bias, so the
untrained model is `LN(x)`. The gate is per node and per dimension; the per-node trust score used in diagnostics is the
mean of `g_i` over the 768 dimensions. Options: `gate: learned | none | fixed_zero` (`none` = `z = LN(x + f(ÂX))`;
`fixed_zero` = `z = LN(x)`, no trainable path), `graph_branch: false | true` (the dual variant above),
`gate_bias_init: -4.0`, `gate_weight_decay: 1e-4` (applied to `W_g` / `W_s` only, via a separate optimiser parameter
group; the other parameters keep weight decay 0), `adapter_dim: 128`. Output dimension is 768 (not 128): the export,
`evaluate` and the loss accept any width. The gate scores of the final model are exported as
`node_embeddings/gate.npy` (`(n,)` mean gate; `(n, 2)` text / graph for the dual variant) and listed in the manifest.

**Variants (each trained on the pinned bottom-5 pseudo-pairs `s11c-rank_topk-k5`, run `20261027T205523Z-ac890ca3`,
`infonce` temperature 0.1, `alpha 0.5`, Adam 1e-3, 100 epochs, CPU; epochs and loss equal to A0 / A1 / A2):**

| Id | Model | Config | Idea ablation item |
|---|---|---|---|
| A0, A1 | current GCN (concat dual / text-only) | reused, 12.2 | 1 |
| G2 | residual, no gate: `LN(x + f(ÂX))` | `gate: none` | 2 |
| G3 | gated residual, text | `gate: learned` | 3 |
| G4 | gated residual, gate fixed to 0 (sanity, equals frozen up to LN) | `gate: fixed_zero` | 4 |
| G5 | dual, gate arbitration of text and DeepWalk | `gate: learned`, `graph_branch: true` | 5 (against A0, concatenation) |

Model ids: `ent-g{2,3,4,5}-gated-mpnetc-s{seed}`. Seeds: 42 for the table; G2, G3 and G5 also at 43 and 44 (the idea
asks for 3–5 seeds; three is the cheapest that gives a spread). The hard-negative sampler run
`20261028T023558Z-decc373b` (`s11c-rank_topk-k5-xhard`, from 11.14) is used for one extra pair, G3-hard and G5-hard,
seed 42, to answer the idea's "bottom-5 and hard" request; these two are reported separately.

- [x] **Model code:** `GatedResidualEncoder` and `build_gated_residual_encoder` in `gcl_train/models.py`
      (Google-style docstrings with the equations above); a branch in `gcl_train/pipeline.py` for
      `encoder: residual_gated` (reading `gate`, `graph_branch`, `adapter_dim`, `gate_bias_init`,
      `gate_weight_decay`), the separate optimiser parameter group, and the `gate.npy` export with its manifest
      output. Defaults leave `gcn`, `gat` and `residual_adapter` untouched (existing tests and the A0 / A1 / A2
      embeddings are unaffected; no existing run is regenerated).
- [x] **Unit tests** (`tests/unit/pipelines/gcl_train/`): `gate: fixed_zero` output equals `LN(x)` exactly; the
      untrained `learned` model has mean gate close to σ(−4) and output within a small tolerance of `LN(x)`; `none`
      has no gate parameters; the dual model with both gates forced to 0 equals `LN(x)`; an isolated node gives
      `(ÂX)_i = x_i` (self-loop only) and no NaN; the gate-bias init and the `W_g`-only weight-decay group are
      applied; output shape `(n, 768)`, finite, and gradients reach `W_g` for `learned` only; same seed, same output;
      the old encoders are unchanged. `uv run pytest tests/unit -q` passes.
- [x] **Train** G2, G3, G4, G5 (seed 42), G2 / G3 / G5 (seeds 43, 44), and G3-hard / G5-hard (seed 42): 14 runs. Per
      run: 100 `metrics.jsonl` rows, loss lower at epoch 100 (G4 excepted: nothing trainable, report constant), no NaN,
      2,927 × 768 finite embeddings, resolved config shows `encoder: residual_gated`, `gate`, the training data run
      and the seed; record trainable parameters and loss at epoch 1 and 100. Rerun G3 seed 42 once and confirm
      identical embeddings.
- [x] **Gate diagnostics** (`scripts/gate_diagnostics.py`, read-only; unit test on a hand-built fixture): for G3 and
      G5, per graph role, the histogram of the mean gate, its mean / median / 5th / 95th percentile, the Spearman
      correlation with node degree (degree from the pinned ingest `edges.parquet`), and the share of nodes with mean
      gate below 0.05 and above 0.95. Report the collapse cases the idea lists: a gate stuck near 0 (no benefit; model
      equals frozen) or near 1 (plain GCN-like) is reported as such, not tuned away.
- [x] **Sanity (G4 vs frozen):** report the mean top-10 cosine-neighbour overlap between `LN(x)` and frozen `x`, and
      the maximum absolute difference of the Phase 12.6 table columns between G4 and frozen mpnet-chunk. If any column
      differs by more than 0.01, also train `norm: none` (G4 without LN) once to isolate the LayerNorm effect; state the
      outcome either way.
- [x] **Slow-training risk:** if the G3 mean gate has not moved from its initial value (mean gate < 0.05 at epoch 100),
      train G3 once more at 300 epochs and report it as a separate row labelled with its epochs; it is not mixed into
      the main table (A0 / A1 / A2 are 100 epochs). No other tuning in this sprint.
- [x] **Execution log** (`dev_docs/execution_logs/sprint_12.md`): model definition as implemented, tests, run ids and
      configuration of every run, losses and parameters, diagnostics, the sanity result, commands.

**Acceptance:** the gated encoder is selectable by config only; every run in the table above has a pinned run id, passes
the sanity checks and is fully specified in the log; no existing run or default behaviour changed.

## Phase 12.6 — Evaluate, plot, table and package the gated models (as 12.3 and 12.4)

Same Sprint 3 proxy protocol, same views, ks, roles and textual reference as Phase 12.3, so the new rows are directly
comparable with the 12.3 table.

- [x] `configs/experiments/attack_cwe_ent_paper_gated.yaml` (copy of `attack_cwe_ent_paper_compare.yaml`): 9 methods
      pinned by run id: the five of 12.3 (`mpnet-chunk`, `deepwalk`, `a0`, `a1`, `a2`) plus `g2`, `g3`, `g4`, `g5`
      (seed 42, `gcl_run`); same `neigh` and `clusters` evaluations; `visualize: {enabled: true, methods: [g3, g5]}`.
      `configs/experiments/attack_cwe_ent_paper_gated_seeds.yaml`: `g2`, `g3`, `g5` at seeds 42 / 43 / 44, plus
      frozen `mpnet-chunk`, same evaluations, no visualisation. `configs/experiments/attack_cwe_ent_paper_gated_hard.yaml`:
      `mpnet-chunk`, `g3`, `g5` (bottom-5) and `g3-hard`, `g5-hard`.
- [x] `configs/plots/attack_cwe_ent_paper_gated.yaml` (as 12.3): `precision_at_k` and `silhouette` figures for the
      9-method run, `loss_curves` enabled, `plot.inputs.evaluate_run` pinned.
- [x] Run, then repeat the main evaluate once (same command pattern as 12.3, `--experiment attack_cwe_ent_paper_gated`,
      `--set evaluate.inputs.ingest_run=20261027T194643Z-5d3f8baf`), then `cga run plot`; run the seeds and hard
      evaluations once each.
- [x] **Table:** `scripts/paper_proxy_table.py` on the 9-method evaluate run (no code change expected; it already
      handles any method list; best per column in bold). Add `scripts/paper_proxy_seed_table.py` (unit test on a
      fixture): mean ± sample std over seeds 42–44 of the same 13 columns for `g2`, `g3`, `g5`, with frozen
      mpnet-chunk as a reference row.
- [x] **Package:** `artifacts/paper/attack_cwe_ent/sprint_12_gated/` with `paper_table.md` / `.csv`, the seed table, the
      hard-negative table, the two figures, the G2–G5 loss curves, the G3 / G5 projections, the gate histogram and
      gate-vs-degree figures from 12.5, and `runs.yaml` (all pinned run ids from 12.1, 12.2 and 12.5, the evaluate /
      plot run ids, the commands).
- [x] **Checks:** the repeat evaluate run is identical (apart from `run_id`); the five 12.3 rows reproduce their 12.3
      values exactly (the neighbourhood query set is the same universe; state any difference); `n_queries` is equal
      across the 9 methods per (view, role) in `neigh`; the tables regenerate byte-identically with one command.
- [x] **Reading and caveats in the log (no tuning):** the idea's success criterion: whether G5 is at least A1 within
      seed noise, and whether the gated models stay at or above frozen mpnet-chunk on the textual view (the view that
      measures text; the structural views reward structure and are reported as the benefit or cost of the gate); G3
      against G2 (value of the gate) and G5 against A0 (gate arbitration against concatenation); the gate behaviour
      (collapsed, intermediate or saturated) and its relation to degree; bottom-5 against hard negatives for G3 and
      G5. Carry the 12.4 caveats unchanged (proxy relevance sets, within-graph metrics that do not measure alignment,
      SecRoBERTa reference favours text rows, silhouette node sets differ across stores) and add: three seeds only for
      G2 / G3 / G5, one seed for every other row; the gold-based comparison of the gated models is not part of this
      sprint (a Sprint 11-style gold evaluation would be the next step to test alignment).

**Acceptance:** the 9-row table and the seed table regenerate from their evaluate runs with one command each; the package
folder holds every figure and run id referenced in the log; the gated models are fully specified and reproducible.

---

## Out of scope

- New model variants, architecture changes, sampler or loss sweeps, other seeds, retraining of A0 / A1 / A2 (Phases 12.5–12.6 are the exception: the gated variants and their seeds are new).
- Gold-based metrics (`gold_alignment`, `gold_alignment_prf`, `gold_set_retrieval`) for any row, including the gated models; Sprint 11 covers the older rows.
- New embeddings or training data unless Phase 12.1 finds a missing artifact.
- Any evaluator change. New code: the table script (12.4), the gated encoder and gate diagnostics (12.5), the seed table script (12.6).
