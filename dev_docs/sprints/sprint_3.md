# Sprint 3 — Run AttWeakBridge-GCL on `ncit_doid` with the `attack_cwe` setup

## Objective

Run the full pipeline (`ingest → embed → training_data → gcl_train → evaluate → plot`) on the OAEI
Bio-ML `ncit_doid` dataset and repeat the Sprint 2 experiment set (sampler sweep, `alpha` sweep,
encoder × graph-embedder grid), treating `ncit_doid` exactly like `attack_cwe`: **two graphs, no
ground truth**.

**Scope decision.** This sprint deliberately ignores the dataset's gold alignments
(`refs_equiv/*`) and their train/test split. No gold-based evaluator, no `supervised` sampler.
Evaluation uses the same proxy metrics as Sprint 2 (`neighborhood`, `cluster_quality`). Scoring
against the gold mappings is a separate, later sprint.

Background: [oaei_bioml_ncit_doid_dataset_structure.md](../oaei_bioml_ncit_doid_dataset_structure.md),
[sprint_2.md](sprint_2.md), [execution_logs/sprint2.md](../execution_logs/sprint2.md).

## Fixed choices

| Choice | Value | Reason |
|---|---|---|
| Text embedding (trained towers + baseline) | **SapBERT** `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, label `sapbert` | Biomedical model trained for synonym/entity matching; the default SecRoBERTa is a cybersecurity model. PubMedBERT is the drop-in alternative if SapBERT is unavailable. |
| Generic text reference | **bert-base-uncased**, label `bert_base` | Used only as the `textual` view's reference store in `evaluate`, kept distinct from the method under comparison (same rule as `full_grid_compare.yaml`). It is not trained on. |
| Text variant | `label_synonyms_definition` | Richest text; `embed.inputs.text_variant` defaults to `description`, which `ncit_doid` does not have, so it **must** be overridden on every embed run. |
| Structural embeddings | DeepWalk, node2vec | Same as Sprint 2. |
| Starting sampler / alpha / epochs | `rank_topk`, `k=5` / `alpha=0.5` / 100 | Sprint 2 winners, used as the starting point and then re-swept. |

## Starting state

- The `ncit_doid` adapter and [configs/datasets/ncit_doid.yaml](../../configs/datasets/ncit_doid.yaml)
  exist but have **never been run end-to-end** (`data/processed/` only holds `attack_cwe`).
- Sweep specs and experiment files under `configs/` name `attack_cwe` embedder/model ids, so
  `ncit_doid_*` copies are needed. Do not reuse them in place.
- `pruning.mode: alignment_reachable` is declared but not implemented. It depends on alignments, so
  it is out of scope; only `mode: none` is used.

## Conventions for every phase

- Google-style docstrings on every module, class, and public function (per `CLAUDE.md`).
- Unit tests in `tests/unit/`; run them with `uv run pytest tests/unit -q`.
- Each phase ends with an entry in `dev_docs/execution_logs/sprint3.md`.
- Every `--dataset` is `ncit_doid`. Commit messages are prefixed `[Sprint-3.N]`.

---

## Phase 3.0.0 — NCIT parser: add edges for named intersection members

Why: the parser only keeps direct `rdfs:subClassOf rdf:resource` pointers, so 7,841 of 15,762 NCIT
classes (49.7%) have no edge (no DeepWalk walk, isolated GCN node). Their parents are written only as
`owl:equivalentClass → owl:intersectionOf` logical definitions. See
[ncit_edge_extraction_analysis.md](../ncit_edge_extraction_analysis.md) (option B). Restriction
fillers (options C/D) stay out of scope.

- [x] Inspect the open point from the analysis: `C27691` has a second `owl:equivalentClass` that is a
      bare `owl:Restriction`. Confirm the parser ignores it and still reads the intersection one.
- [x] `owl_parser.py`: `OwlClassRecord` gains `intersection_parent_iris`; `parse_owl_ontology` descends
      into `owl:equivalentClass → owl:Class → owl:intersectionOf` and collects members that are named
      classes (`rdf:Description`/`owl:Class` with `rdf:about`), ignoring `owl:Restriction` members.
      `raw_sha256` includes the new field.
- [x] `iter_owl_edges` takes `edge_sources` (`subClassOf`, `equivalent_intersection`) and emits the
      intersection parents as `relation="subClassOf"` edges, deduplicated against direct pointers and
      dropping targets that are not known classes.
- [x] `OntologyConfig.edge_sources` (default `[subClassOf]`, so DOID is unchanged), set to both sources
      for NCIT in `configs/datasets/ncit_doid.yaml`; the adapter passes it through.
- [x] Unit tests (`tests/unit/adapters/`): fixture class modeled on `C100051` (one named + one
      restriction member) yields exactly one extra edge; a class with 3 named members yields 3 edges;
      a member also listed as a direct parent is not duplicated; default `edge_sources` is unchanged.
- [x] Update the `owl_parser.py` module docstring and `oaei_bioml_ncit_doid_dataset_structure.md`
      (logical axioms are no longer "not interpreted").
- [x] Re-ingest (Phase 3.0.1) and confirm NCIT edges ≈ 23,486 and isolated NCIT nodes ≈ 6.
- [x] The Phase 3.1 `deepwalk-deepwalk` store was built from the old graph and is stale: re-run DeepWalk
      after re-ingest (and re-check that the text stores, which are node/text-only, remain valid).


- [x] **Disable the gold alignments for this sprint.** Remove (or comment out) the `alignments:`
      block in `configs/datasets/ncit_doid.yaml`. This is needed even though nothing here scores
      against the gold set: `training_data` marks every endpoint of a `test`-split alignment pair as
      `held_out`, which would silently reintroduce the dataset's own split into training. With no
      alignments, `ncit_doid` behaves exactly like `attack_cwe`: the only held-out set is ingest's
      random 80/10/10 node split. The adapter already supports `alignments: None`. Keep the removed
      block recorded (git history or a comment) so the later gold-set sprint can restore it.
- [x] `uv run cga run ingest --dataset ncit_doid --dry-run`, then a real run.
- [x] Check ingest stats against the dataset doc, and record actual numbers: nodes (~15,762 NCIT /
      ~8,465 DOID, fewer after dropping unusable classes), `rdfs:subClassOf` edges (~16,030 /
      ~9,388, minus edges to `owl:Thing` or missing classes), and `n_alignments == 0`.
- [x] Text coverage per variant and side: % of nodes with a definition (DOID's `IAO_0000115` is
      partial) and % with at least one synonym.
- [x] Spot-check one known class (e.g. NCIT `C3224` Melanoma) in `texts.parquet` for label,
      synonyms, and definition.
- [ ] **Scale check (done during Phase 3.2).** `training_data` builds dense similarity matrices
      (text ≈ 15.8k × 8.5k, graph ≈ 15.8k² ≈ 1 GB fp32). Record peak RAM. Only if it does not fit,
      raise a follow-up for a size-reduction approach; do not build one pre-emptively.

## Phase 3.1 — Embeddings

- [x] SapBERT (the text tower):
      `uv run cga run embed --dataset ncit_doid --set embed.inputs.text_variant=label_synonyms_definition --set embed.embedder.model_ref=cambridgeltl/SapBERT-from-PubMedBERT-fulltext --set embed.embedder.label=sapbert`
- [x] bert-base-uncased (reference only): same command with
      `model_ref=bert-base-uncased` and `label=bert_base`.
- [x] DeepWalk: `uv run cga run embed --dataset ncit_doid --config configs/pipelines/embed_deepwalk.yaml`
- [ ] (skipped, optional; needed for Phase 3.5) node2vec: same file with `--set embed.embedder.kind=node2vec --set embed.embedder.label=node2vec`. (optional)
- [x] Record each store's `status.parquet` failure count (classes with empty text) and its runtime.

## Phase 3.2 — First end-to-end training run

SapBERT + DeepWalk, `rank_topk` with `k=5`, GCN then GAT.

- [ ] `training_data` with `training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert`,
      `training_data.sampler.name=rank_topk`, `training_data.sampler.k=5`. Record `n_examples`,
      universe size, `held_out` count, and peak RAM (feeds the Phase 3.0 scale check).
- [ ] `gcl_train` GCN: `gcl_train.inputs.training_data_sampler_id=rank_topk-k5`,
      `gcl_train.model.model_id=gcn-sapbert-deepwalk`, `gcl_train.train.epochs=100`. Then GAT with
      `gcl_train.model.encoder=gat` and `model_id=gat-sapbert-deepwalk`.
- [ ] Check training sanity: loss decreases in `metrics.jsonl` and no NaNs.
- [ ] `configs/experiments/ncit_doid_neighborhood_compare.yaml` (modeled on
      `neighborhood_compare.yaml`): methods = `deepwalk`, `sapbert`, the two trained models;
      evaluations = `neighborhood` (`structural_r1`, `structural_r2`, and `textual` with the
      `hf_mean_pool-bert_base` reference) + `cluster_quality`. Same `ks` and `n_clusters` as
      `full_grid_compare.yaml` so numbers stay comparable with Sprint 2.
- [ ] `configs/plots/ncit_doid_neighborhood_compare.yaml`, then run `evaluate` and `plot`.

## Phase 3.3 — Sampler sweep (repeat of Phase 2.1)

- [ ] `configs/sweeps/ncit_doid_rank_topk_k.yaml` (`k: [5, 10, 20, 30]`) and
      `configs/sweeps/ncit_doid_percentile_pct.yaml` (zipped `top_pct`/`bottom_pct:
      [0.02, 0.05, 0.08, 0.1]`), both with `dataset: ncit_doid`. Use `fixed:` to pin
      `training_data.inputs.text_embeddings.embedder_id: hf_mean_pool-sapbert`, because the
      pipeline default is SecRoBERTa.
- [ ] Run both sweeps, then per point `cga run gcl_train --dataset ncit_doid --set
      gcl_train.inputs.training_data_sampler_id=<id> --set gcl_train.model.model_id=gcn-<id> --set
      gcl_train.train.epochs=100` (same procedure as Phase 2.1).
- [ ] `configs/experiments/ncit_doid_sampler_sweep_compare.yaml` +
      `configs/plots/ncit_doid_sampler_sweep_compare.yaml` (8 `gcl_run` methods).
- [ ] Reporting table per sweep point: `mrr_textual`, `precision@10_textual`, `silhouette_mean`,
      `final_loss`, `n_examples`. Select on `mrr_textual`, as in Sprint 2. Note whether the winner
      matches `attack_cwe`'s (`rank_topk`, `k=5`).

## Phase 3.4 — `alpha` sweep (repeat of Phase 2.2)

- [ ] `configs/sweeps/ncit_doid_alpha.yaml`: `pipeline: gcl_train`, `dataset: ncit_doid`, `alpha`
      in `[0.0, 0.25, 0.5, 0.75, 1.0]` zipped with `model_id` (`gcn-alpha0` … `gcn-alpha1`, using the
      `%g` rendering rule from Phase 2.2). `fixed:` sets the Phase 3.3 winner's
      `training_data_sampler_id` and `epochs: 100`.
- [ ] Dry-check the expansion with `build_config` before launching (Phase 2.2 lesson: the
      `model_id` and `sweep_point_id` strings must match).
- [ ] `configs/experiments/ncit_doid_alpha_sweep_compare.yaml` + plot config. Select on
      `mrr_textual`; record `final_loss` but do not select on it (the objective changes with `alpha`).
- [ ] Ablation readout: does an intermediate `alpha` beat both `alpha=0` (graph-only) and `alpha=1`
      (text-only)?

## Phase 3.5 — Comparison grid and write-up

- [ ] With the Phase 3.3/3.4 winners fixed, train SapBERT × {DeepWalk, node2vec} × {GCN, GAT} = 4
      models (`model_id` `{enc}-sapbert-{graph}`, same scheme as `full_grid_compare.yaml`). Each needs
      its own `training_data` run with the matching graph embedder.
- [ ] `configs/experiments/ncit_doid_full_grid_compare.yaml`: 4 trained models + baselines
      (`deepwalk`, `node2vec`, `sapbert`), evaluations as in Phase 3.2.
- [ ] Final table in the execution log (`mrr_textual`, `precision@10_textual`, `silhouette_mean` per
      method) and a short comparison against the `attack_cwe` results.
- [ ] Update `README.md` with the `ncit_doid` run commands and new experiment/sweep files.

---

## Out of scope (deferred to the gold-set sprint)

- Scoring against `refs_equiv/*` (Hits@K / MRR on gold pairs, `test.cands` ranking).
- The `supervised` sampler and any use of the dataset's train/test alignment split.
- `refs_subs/` subsumption mappings and NCIT logical axioms as edges.
- `pruning.mode: alignment_reachable`, extending `cga sweep`, and encoder fine-tuning.

**Caveat to carry into the write-up.** As in Sprint 2, evaluation scores all nodes of both graphs
(including train nodes) against proxy relevance sets, not held-out data or gold labels. Treat the
results as a relative comparison between configurations, not as alignment accuracy.
