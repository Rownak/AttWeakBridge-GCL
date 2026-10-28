# Sprint 12 — Paper experiments on `attack_cwe_ent` (mpnet-chunk + DeepWalk, A0 / A1 / A2)

## Objective

Produce the paper's comparison on Enterprise ATT&CK × CWE (`attack_cwe_ent`) with the Sprint 3 end-to-end flow
(`ingest → embed → training_data → gcl_train → evaluate → plot`) and the Sprint 3 evaluation protocol: gold-free
proxy metrics (`neighborhood` and `cluster_quality`). Every upstream artifact already exists from Sprint 11 and is
reused by pinned run id; nothing is retrained. The sprint only verifies the artifacts, evaluates, plots, and packages
one table and the figures.

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
| Models | reused from Sprint 11, **not retrained**; seed 42 only |
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

- [ ] `configs/experiments/attack_cwe_ent_paper_compare.yaml` (modelled on `ncit_doid_neighborhood_compare.yaml`):
      5 methods pinned by run id — `mpnet-chunk`, `deepwalk` (`embedding_store`), `a0`, `a1`, `a2` (`gcl_run`);
      evaluation `neigh` (`neighborhood`, `ks [1, 5, 10, 20]`, roles `graph_1` / `graph_2`, views `structural_r1`
      (`hop_radius`, radius 1), `structural_r2` (radius 2), `textual` (`cosine_top_k`, `top_k 10`, reference the pinned
      SecRoBERTa store)); evaluation `clusters` (`cluster_quality`, `n_clusters [2, 4, 8, 16, 32]`);
      `visualize: {enabled: true, methods: [a0, a1, a2]}`.
- [ ] `configs/plots/attack_cwe_ent_paper_compare.yaml` (modelled on `ncit_doid_neighborhood_compare.yaml`):
      `precision_at_k` line figure (`neigh`, `normalized_precision_at_k`, x `param_value`, hue `method_label`, facet
      `[view_id, graph_role]`), `silhouette` line figure (`clusters`, facet `[graph_role]`), `loss_curves` enabled,
      `plot.inputs.evaluate_run` pinned to the Phase 12.3 evaluate run.
- [ ] Run, then repeat once:
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

- [ ] One comparison table, built by a small script `scripts/paper_proxy_table.py` (reads the evaluate run's
      `metrics.parquet`; writes `paper_table.md` and `paper_table.csv` next to it). Rows: the 5 methods. Columns, as in
      Sprint 3: MRR and normalized precision@10 for `structural_r1`, `structural_r2` and `textual`, each per graph
      (ATT&CK = `graph_1`, CWE = `graph_2`), and the silhouette mean over `n_clusters` 2–32 and both graphs. Best per
      column in bold. Unit test on a hand-built `metrics.parquet` fixture.
- [ ] Package for the paper: `artifacts/paper/attack_cwe_ent/sprint_12/` with `paper_table.md` / `.csv`, the two
      figures, the A0 / A1 / A2 loss curves and projections, and `runs.yaml` (every pinned run id from this file, the
      evaluate and plot run ids, and the exact commands).
- [ ] Short reading in the log (no tuning): which rows lead on structural views, on the textual view and on
      silhouette; where A0 / A1 / A2 sit between the two frozen baselines.
- [ ] Caveats in the log, carried into the paper text: proxy relevance sets, not gold labels; neighborhoods are scored
      within each graph, so these metrics do not measure ATT&CK → CWE alignment; the `textual` reference
      (SecRoBERTa) is itself a text encoder, so text-based rows have an advantage there; single seed.

**Acceptance:** the table regenerates byte-identically from the evaluate run with one command; the package folder holds
every figure and run id referenced in the log.

---

## Out of scope

- New model variants, architecture changes, sampler or loss sweeps, other seeds, retraining.
- Gold-based metrics (`gold_alignment`, `gold_alignment_prf`, `gold_set_retrieval`); Sprint 11 covers them.
- New embeddings or training data unless Phase 12.1 finds a missing artifact.
- Any evaluator change; the only new code is the table script in Phase 12.4.
