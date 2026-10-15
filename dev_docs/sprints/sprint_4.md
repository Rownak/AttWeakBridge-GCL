# Sprint 4: Supervised `ncit_doid` run on the gold alignments

## Objective

Train AttWeakBridge-GCL on `ncit_doid` using the dataset's **gold equivalence pairs** as supervised
positive cross-graph examples. Score the trained model by retrieval over the **held-out gold test
pairs** (Hits@K, MRR). This is the "gold-set sprint" that Sprint 3 deferred.

The pipeline chain is unchanged (`ingest → training_data → gcl_train → evaluate`). Only two pieces
change: the existing `supervised` sampler is fixed, and a new `gold_alignment` evaluator is added.
The supervised experiment runs under its own dataset config, `ncit_doid_supervised`, so Sprint 3's
`ncit_doid` artifacts and `LATEST` pointers are never touched. Embedding stores from Sprint 3 are
reused, with no re-embedding.

Background: [sprint_3.md](sprint_3.md), [execution_logs/sprint3.md](../execution_logs/sprint3.md),
[oaei_bioml_ncit_doid_dataset_structure.md](../oaei_bioml_ncit_doid_dataset_structure.md).

## What already exists (reuse, do not rebuild)

| Piece | Where | State |
|---|---|---|
| Gold TSV parsing + candidate pools | `adapters/ncit_doid/refs_parser.py` | Done. Writes `alignments.parquet` with `split` (train/test) and `candidates` on test rows. |
| `alignments:` block | `configs/datasets/ncit_doid.yaml` | Commented out in Sprint 3. `ncit_doid.yaml` is left **unchanged**; the block is enabled only in the new supervised config. |
| Dataset config → artifact paths | `cli.py` | `--dataset X` loads `configs/datasets/X.yaml`, and every artifact path uses the file's `dataset.id`. A new file with a new id therefore gets its own lineage. |
| Test-pair isolation | `training_data.BuildUniverseStep` | Every endpoint of a `test` alignment is `held_out`, so it is absent from all similarity matrices. `ValidateAndManifestStep` fails the run if any example references a held-out node. |
| `supervised` sampler | `training_data/samplers.py` | Registered, but needs fixing (see Phase 4.1). |
| Retrieval metrics | `evaluate/metrics.py` | `hit_at_k`, `top_k_by_similarity`, `mrr_via_argsort` with a deterministic `(-sim, id)` tie-break. Reuse them as-is. |
| Evaluator registry | `evaluate/evaluators/` | Each new evaluation goes in a new file; the pipeline needs no change apart from exposing alignments (Phase 4.3). |

### Gold-set facts (measured from `data/raw/ncit_doid/refs_equiv/`)
- 1,406 train pairs and 3,280 test pairs, with no pair in both splits.
- 23 train pairs share an endpoint with a test pair. These are dropped from training, which is
  expected because that endpoint is held out.
- Not strictly 1:1: 52 NCIT sources and 18 DOID targets appear in more than one gold pair (across
  both splits).
- `test.cands.tsv`: 101 candidates per test pair, and the true target is included.

## Fixed choices

| Choice | Value | Reason |
|---|---|---|
| Dataset configs | keep `ncit_doid.yaml` as is (`id: ncit_doid`, no alignments); add `ncit_doid_supervised.yaml` (`id: ncit_doid_supervised`, alignments on) | Completed and remaining Sprint 3 experiments, sweep specs, and the README keep working unchanged. The supervised id gets its own `data/processed/`, `artifacts/`, and `models/` lineage, so a gold-aware ingest can never become `LATEST` for `ncit_doid` runs, and no Sprint 3 command needs a pinned `ingest_run`. |
| Node split | ingest `split.fractions: [1.0, 0.0, 0.0]` | The random 80/10/10 node split would hold out ~36% of train pairs (an endpoint lands in val/test) for no benefit. With all nodes `train`, `held_out` comes **only** from gold test endpoints, which is the split this experiment evaluates on. This needs no code change (`validate_fractions` accepts zeros). It is passed with `--set` because the pipeline layer (`configs/pipelines/ingest.yaml`) merges after the dataset layer and would override a value set in the dataset YAML. |
| Positives | gold `train` pairs (NCIT anchor → DOID `text_pos`) | As the task asks. NCIT-anchored only, because the gold files are NCIT→DOID. |
| Text negatives | the anchor's `negatives_per_positive` (default 5) **least text-similar** DOID nodes, via `_topk_bottomk_indices` on its `text_g1_g2` row, with the anchor's gold targets (any split) excluded before ranking | Same rule as the graph negatives, and the same rank-based idea as `rank_topk`. Held-out DOID nodes are never candidates because they are absent from `text_g1_g2`. One example row per negative, which keeps `examples.parquet` unchanged. |
| Graph pos/neg | the anchor's j-th most/least structurally similar NCIT node (same rule as `rank_topk`) | The current sampler picks them **uniformly at random**, so the graph-view loss pulls random nodes together. Using the existing structural ranking keeps `alpha` meaningful without inventing a new signal. |
| Embeddings | `hf_mean_pool-sapbert` (text), `deepwalk-deepwalk` (graph), `hf_mean_pool-bert_base` (baseline only) | The Sprint 3 stores are copied into the `ncit_doid_supervised` lineage (Phase 4.0). Node ids and texts are unchanged, and `training_data` joins stores by `node_id`, so the copies are valid (the same argument as Sprint 3.0.0). Re-embedding would take about 3 hours for identical vectors. |
| Model | GCN, `alpha=0.5`, 100 epochs, triplet loss | Sprint 3 defaults. No sweep in this sprint. |

## Conventions for every phase

- Google-style docstrings on every module, class, and public function (per `CLAUDE.md`).
- Unit tests go in `tests/unit/`. Run them with `uv run pytest tests/unit -q`.
- Each phase ends with an entry in `dev_docs/execution_logs/sprint_4.md`.
- Commit messages are prefixed `[Sprint-4.N]`.
- Every Sprint 4 command uses `--dataset ncit_doid_supervised`. Unsupervised work (including any
  remaining Sprint 3 phases) keeps using `--dataset ncit_doid`.

---

## Phase 4.0: Add the supervised dataset config and ingest it

- [x] **Create** `configs/datasets/ncit_doid_supervised.yaml`: a copy of `ncit_doid.yaml` with
      `dataset.id: ncit_doid_supervised` and the `alignments.equivalence` block enabled (`train`,
      `test`, `candidates`). Keep `adapter: ncit_doid`, the same `root`, and the same
      `adapter_config` (including NCIT `edge_sources`), so the graph matches Sprint 3 exactly.
      Replace the Sprint 3 "restore this block" comment with a header comment pointing to this
      sprint and noting the required `--set` for the node split. The adapter block is duplicated
      because the config loader has no `include`/inheritance mechanism, and adding one is out of
      scope.
- [x] Do **not** edit `configs/datasets/ncit_doid.yaml`, the `ncit_doid_*` sweep, experiment, or
      plot configs, or past execution logs.
- [x] Dry-run check: `uv run cga run ingest --dataset ncit_doid_supervised --dry-run` resolves to
      `data/processed/ncit_doid_supervised/` and finds the three `refs_equiv` files.
- [x] Ingest the supervised variant with an all-train node split:
      `uv run cga run ingest --dataset ncit_doid_supervised --set "ingest.split.fractions=[1.0,0.0,0.0]"`
- [x] Check `stats.json`: `n_alignments == 4686`, `n_alignments_by_split == {train: 1406, test: 3280}`,
      every node `train`, and node/edge counts identical to `ncit_doid` run `20261011T134334Z-6d45b8af`
      (24,227 nodes, 32,874 edges).
- [x] **Reuse the embedding stores.** Copy the `LATEST` run directory of each of the three stores
      from `artifacts/embeddings/ncit_doid/<embedder_id>/` to
      `artifacts/embeddings/ncit_doid_supervised/<embedder_id>/` and write each `LATEST`. The stores
      are `hf_mean_pool-sapbert` (`20261009T182919Z-95011340`), `hf_mean_pool-bert_base`
      (`20261009T191700Z-1278c23a`), and `deepwalk-deepwalk` (`20261011T134446Z-d5aa89ed`). Before
      relying on the copies, confirm that the new ingest's `node_id` order and `text_sha256` match
      the old run. Record the copy (source run ids) in the execution log, because the copied
      manifests still name the `ncit_doid` ingest run.
- [x] Record how many gold pairs fall outside the `intersection` universe. The 6 isolated NCIT
      nodes have no DeepWalk vector, so any pair touching them cannot be trained or evaluated.

## Phase 4.1: Fix the `supervised` sampler

All changes are in `training_data/samplers.py` and `training_data/pipeline.py`. The sampler
signature and the `examples.parquet` schema stay unchanged.

- [x] **Pass the full gold set to the sampler.** `SampleStep` currently pre-filters to trainable
      pairs, so the sampler cannot see test pairs it must avoid as negatives. Pass **all** alignment
      rows (with `split`) instead. The sampler then:
  - uses as positives only rows with `split == "train"` whose endpoints are both in the
    non-held-out similarity matrices (`text_g1_g2.row_ids`/`col_ids`), which is the existing
    eligibility check;
  - builds `gold_pairs = {(n1, n2)}` over **all** rows for negative exclusion.

  Leakage stays structurally impossible: held-out nodes are absent from the matrices, and the
  `ValidateAndManifestStep` assertion is unchanged.
- [x] **Generalize `_topk_bottomk_indices`.** Add an `exclude_indices` parameter (a collection)
      next to the existing `exclude_index`, and drop all of those columns before ranking. Existing
      callers keep their behavior, and the `rank_topk`/`percentile` tests must still pass unchanged.
- [x] **Text negatives.** For each positive `(a, p)`, take the anchor's `text_g1_g2` row and call
      `_topk_bottomk_indices(row, negatives_per_positive, exclude_indices=<columns of a's gold targets>)`.
      Use the **bottom-k** (least text-similar DOID nodes) as `text_neg_id`, one example row per
      negative. The gold targets are all `n2` with `(a, n2)` in `gold_pairs`, across both splits; a
      target that is held out is already absent from the row. The result is deterministic, and the
      seed only drives the final shuffle.
- [x] **Graph pos/neg.** Replace the uniform-random pick with `_topk_bottomk_indices` over the
      anchor's `graph_g1` row (self excluded). Pair negative j with the j-th top and bottom
      structural neighbors.
- [x] Shuffle the output with a seeded RNG (as `rank_topk` does). Build a column-index lookup for
      `text_g1_g2.col_ids` once, and hoist the `set(text_g1_g2.row_ids)` that is currently rebuilt
      inside the loop.
- [x] `build_sampler_id`: add `"supervised": ("negatives_per_positive",)` so the id becomes
      `supervised-negatives_per_positive5`.
- [x] `stats.json`: add `n_gold_train_pairs`, `n_gold_train_pairs_used`, and
      `n_gold_train_pairs_skipped` (endpoint held out or outside the universe).
- [x] Update the module and function docstrings and the `SampleStep` comment.
- [x] Tests (`tests/unit/pipelines/training_data/`):
  - **Split isolation:** no example id column contains an endpoint of a test pair; a train pair
    sharing an endpoint with a test pair is skipped.
  - **Negative validity:** every `(anchor_id, text_neg_id)` is cross-graph (`graph_2`), not held
    out, and not in the gold set of either split. Two fixtures cover this: an anchor with two gold
    targets, and a gold target placed as the *least* text-similar column, which must be skipped in
    favor of the next one.
  - Text negatives are the `negatives_per_positive` least similar eligible columns, in rank order.
  - `_topk_bottomk_indices` with `exclude_indices` never returns an excluded column.
  - Exactly `negatives_per_positive` rows per used positive, and identical output for the same seed.
  - Graph pos/neg are within-graph (`graph_1`), not the anchor, and taken from the structural
    ranking.
  - Update the existing `test_supervised_*` tests in `test_pipeline.py`. Keep the no-op test for
    empty alignments.

## Phase 4.2: Supervised training run

- [x] `training_data`:
      `uv run cga run training_data --dataset ncit_doid_supervised --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5`
- [x] Record `n_examples` (about 1,380 × 5), `n_held_out_nodes`, and the used/skipped pair counts.
- [x] `gcl_train`:
      `uv run cga run gcl_train --dataset ncit_doid_supervised --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.model_id=gcn-supervised --set gcl_train.train.epochs=100`
- [x] Sanity check: `loss` and `loss_text` decrease in `metrics.jsonl` and contain no NaN.
- [x] **Unsupervised reference on the same split.** The Sprint 3 `gcn-rank_topk-k5` model lives
      under `models/gcl/ncit_doid/`, so `evaluate --dataset ncit_doid_supervised` cannot load it.
      Retrain it here (about 5 minutes) with
      `training_data.sampler.name=rank_topk`, `k=5`, SapBERT, then `gcl_train` with
      `model_id=gcn-rank_topk-k5`. This version also holds out the gold test endpoints, so it is a
      strictly held-out comparison. It is a single reference run, not a sweep.

## Phase 4.3: `gold_alignment` evaluator

A new file, `evaluate/evaluators/gold_alignment.py`, registered as `gold_alignment` and imported in
`evaluators/__init__.py`.

- [x] Expose alignments to evaluators. `evaluate.LoadStep` reads `alignments.parquet` from the
      ingest run it already resolves and stores it on `_EvaluateState` as `alignments`. Add the
      field to `EvalResources`. Other evaluators are untouched.
- [x] Params: `ks` (e.g. `[1, 5, 10]`), `split` (default `test`), and `modes` (`[global, candidates]`).
      The evaluator raises `ConfigError` if the selected split has no pairs.
- [x] Queries: one per `split` gold pair `(n1, n2)` whose endpoints are present in **every** method
      (the same intersection rule as `neighborhood`, so all methods share a query set). The relevant
      set is `{n2}`. Similarity is cosine between the method's `n1` vector and the candidate vectors.
  - `global`: the candidates are all `graph_2` ids in the method intersection, **minus other gold
    targets of `n1`** (any split). Without this filter, a correct sibling target ranked above `n2`
    would count as an error for the 1:n sources.
  - `candidates`: the candidates are the row's `test.cands` pool ∩ method intersection, with `n2`
    always included. Skip rows without a pool.
- [x] Metrics: reuse `hit_at_k` (via `top_k_by_similarity`) and `mrr_via_argsort`. Emit `MetricRow`s
      with `metric_name` `hits_at_k` (`param_name="k"`) and `mrr`, `view_id` set to the mode,
      `graph_role="graph_1"`, and `n_queries`. Attach `query_set_sha256` via
      `digest.add_scope_provenance`.
- [x] Tests (`tests/unit/pipelines/evaluate/test_gold_alignment.py`), on a hand-built 2 + 4 node
      fixture with known vectors:
  - Hits@1, Hits@K, and MRR equal hand-computed values. A target at rank 3 gives MRR = 1/3,
    Hits@1 = 0, and Hits@5 = 1.
  - Ties break by node id, deterministically.
  - The `global` filter drops a sibling gold target, while `candidates` mode ranks only within the
    pool.
  - Only `test` pairs are queried, and a pair missing from one method is dropped for every method.
  - `ConfigError` is raised when there are no alignments.

## Phase 4.4: Evaluate and record results

- [x] `configs/experiments/ncit_doid_gold_compare.yaml`:
  - methods: `sapbert`, `bert_base`, `deepwalk` (embedding stores), `gcn_rank_topk_k5` (the
    Phase 4.2 unsupervised reference), `gcn_supervised`;
  - one evaluation: `{evaluation_id: gold, kind: gold_alignment, params: {ks: [1, 5, 10], split: test, modes: [global, candidates]}}`.
- [x] `uv run cga run evaluate --dataset ncit_doid_supervised --experiment ncit_doid_gold_compare`.
      Outputs go to `artifacts/evaluations/ncit_doid_supervised/<run_id>/metrics.{parquet,json}`
      as usual.
- [x] Execution log table, one row per method: Hits@1/5/10 and MRR for `global` and `candidates`,
      plus `n_queries`. State the expected readings: `deepwalk` should score near chance (the two
      graphs share no walks), and `sapbert` is the text-only bar the supervised model must beat.
- [x] Optional, only if an existing figure kind fits without code changes:
      `configs/plots/ncit_doid_gold_compare.yaml` with a bar chart of `mrr` and `hits_at_k` per method.
- [x] `README.md`: add the `ncit_doid_supervised` dataset config, the Sprint 4 commands, and the
      new evaluator to the evaluator list. Leave the existing `ncit_doid` sections unchanged.


## Phase 4.5: Supervised run with `alpha=0.75`

One extra training point, no code changes. Phase 4.4 trained at `alpha=0.5`; this checks whether weighting the text
loss more heavily (0.75) moves held-out retrieval. It is a single point, not a sweep.

- [x] `gcl_train` on the existing supervised examples (`LATEST` of `supervised-negatives_per_positive5`):
      `uv run cga run gcl_train --dataset ncit_doid_supervised --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.loss.alpha=0.75 --set gcl_train.model.model_id=gcn-supervised-alpha0.75 --set gcl_train.train.epochs=100`
- [x] Sanity check: `loss` and `loss_text` decrease in `metrics.jsonl` and contain no NaN.
- [x] Add a sixth method to `configs/experiments/ncit_doid_gold_compare.yaml`:
      `{method_id: gcn_supervised_alpha0.75, label: "GCN supervised a=0.75", kind: gcl_run, model_id: gcn-supervised-alpha0.75, ref: latest}`.
      Re-run `evaluate` and `plot` (commands as in Phase 4.4).
- [x] Execution log: add the new row to the results tables and compare with `gcn_supervised` (`alpha=0.5`) and
      `sapbert`. State plainly whether it closes the gap to SapBERT.

## Phase 4.6: Supervised run with random text negatives

This phase replaces the bottom-k text negatives (`_topk_bottomk_indices`) with **uniformly random** eligible DOID
nodes. Random negatives are more diverse than the 520 recurring bottom-k nodes and somewhat harder on average, but
they are still not mined hard negatives. Graph pos/neg keep the structural ranking. Everything else matches Phase 4.2
(`alpha=0.5`, GCN, 100 epochs).

- [x] **Sampler option.** Add `text_negatives: bottomk | random` to the `supervised` sampler. The default is `bottomk`,
      so Phase 4.2 is unchanged. In `random` mode, draw `negatives_per_positive` distinct columns uniformly, without
      replacement, from the anchor's `text_g1_g2` columns minus its gold targets (any split), using the seeded
      `random.Random(seed)`. If fewer columns are eligible, use all of them.
- [x] `_SAMPLER_ID_PARAMS["supervised"]` += `"text_negatives"`, so the id becomes
      `supervised-negatives_per_positive5-text_negativesrandom`. Do not add the key to
      `configs/pipelines/training_data.yaml`, so the existing `supervised-negatives_per_positive5` id is unchanged.
- [x] Tests (`test_samplers.py`):
  - random negatives are cross-graph, non-gold (either split), and not held out;
  - there are `negatives_per_positive` distinct negatives per positive;
  - the same seed gives identical output and a different seed gives different negatives;
  - `bottomk` output is unchanged.
- [x] `training_data`:
      `uv run cga run training_data --dataset ncit_doid_supervised --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5 --set training_data.sampler.text_negatives=random`
      Record `n_examples`, the used/skipped counts, the number of distinct text negatives, and the mean SapBERT
      cos(anchor, neg) against the bottom-k run.
- [x] `gcl_train`:
      `uv run cga run gcl_train --dataset ncit_doid_supervised --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5-text_negativesrandom --set gcl_train.model.model_id=gcn-supervised-randneg --set gcl_train.train.epochs=100`
      Sanity check: `loss`/`loss_text` decrease, there is no NaN, and record the final loss.
- [x] Evaluate: add `{method_id: gcn_supervised_randneg, label: "GCN supervised rand-neg", kind: gcl_run, model_id: gcn-supervised-randneg, ref: latest}`
      to `ncit_doid_gold_compare.yaml`, then re-run `evaluate` and `plot` (commands as in Phase 4.4).
- [x] Execution log: add the new row to the results tables and compare it with `gcn_supervised` (bottom-k) and
      `sapbert`. State whether negative choice moves held-out retrieval and whether it closes the gap to SapBERT.

---

## Out of scope

- Hyperparameter sweeps (`negatives_per_positive`, `alpha`, encoder, epochs), and GAT/node2vec variants.
- Hard-negative mining, and DOID-anchored (reverse-direction) examples.
- `refs_subs/` subsumption mappings, `pruning.mode: alignment_reachable`, and encoder fine-tuning.
- Mixing supervised and unsupervised (`rank_topk`) examples in one run.

**Caveats for the write-up.**
- Training is transductive. Test-pair nodes stay in the GCN graph and carry their text and structural
  features through message passing, but they never appear in any example, as anchor, positive, or
  negative.
- Text negatives are the *least* similar DOID nodes, so they are easy negatives. If the supervised
  model barely moves away from `sapbert`, this is the first knob for a later sprint (hard
  negatives are out of scope here).
