# Sprint 5: SapBERT-Residual Graph Adapter (SRGA) for supervised `ncit_doid`

## Objective

Beat SapBERT on held-out NCIT–DOID gold-pair retrieval (`gold_alignment`: Hits@1/5/10 and MRR, `global` and
`candidates` modes) by adding graph structure **beside** frozen SapBERT, as proposed in
[design_docs/sapbert_residual_graph_adapter.md](../design_docs/sapbert_residual_graph_adapter.md) (the "design" below).
SRGA is built one testable piece at a time, and every phase ends in an evaluation and a go/stop decision. The sprint
stops adding architecture as soon as a component fails to beat the simpler variant before it.

The bars to beat come from the Sprint 4 test set (3,280 pairs) and the design's §2 probes:

| Bar | global MRR | global Hits@10 | candidates MRR |
|---|---|---|---|
| A0 SapBERT | 0.448 | 0.654 | 0.519 |
| A1 training-free SapBERT + 0.25·neighbour mean (design §2, β picked on train pairs) | 0.469 | 0.701 | 0.529 |

Background: [sprint_4.md](sprint_4.md), [execution_logs/sprint_4.md](../execution_logs/sprint_4.md),
[data_analysis/hard_negative_selection.md](../data_analysis/hard_negative_selection.md).

## What already exists (reuse, do not rebuild)

| Piece | Where | Use in Sprint 5 |
|---|---|---|
| Supervised lineage, stores, gold alignments | `ncit_doid_supervised` (Sprint 4.0) | The template for the new dataset config and the store copy |
| `supervised` sampler (positives = usable gold train pairs) | `training_data/samplers.py` | It supplies `(anchor, text_pos)` rows. SRGA's loss ignores the sampled negatives. |
| Test-pair isolation (`held_out`, leakage assertion) | `training_data/pipeline.py` | Extended to `val` pairs |
| `gold_alignment` evaluator (`split`, `modes`, `ks`) | `evaluate/evaluators/gold_alignment.py` | Unchanged; run with `split: val` and `split: test` |
| `infonce` loss, full-batch trainer, export as `node_embeddings` | `gcl_train/` | The trainer gets one new loss path. The export is unchanged. |
| Experiment/plot configs, `--set`, `LATEST` lineages, `metrics.parquet` | `configs/`, CLI | Unchanged conventions |

## Fixed choices

| Choice | Value | Reason |
|---|---|---|
| Lineage (approved) | new `configs/datasets/ncit_doid_srga.yaml` (`id: ncit_doid_srga`), a copy of `ncit_doid_supervised.yaml` plus a val split | Carving `val` out of train pairs needs a new ingest. A new id keeps Sprint 4's `ncit_doid_supervised` artifacts and `LATEST` pointers untouched, the same reasoning as Phase 4.0. |
| Validation | 15% of gold **train** NCIT sources (all of a sampled source's pairs) relabeled `val`, seed 42; endpoints held out like test | Design §6: nothing may be tuned on test. Sampling by source keeps a 1:n source wholly in one split. |
| Model selection (approved) | on `val`, `global` mode only (val pairs have no `test.cands` pool) | `candidates` mode needs pools, which only exist for test pairs. |
| Text input | the existing SapBERT store (`label_synonyms_definition`, mean pooled), frozen | Holds the bar fixed. A stronger SapBERT input is a separate question (out of scope). |
| Residual weight | one **global, fixed** λ per run (not learned, not per node) | This is the design's safest option, and per-graph λ is equivalent for NCIT→DOID cosine up to norms. Per-graph or per-node weights only come back if a later sprint needs them. |
| DeepWalk | **not** used by SRGA (adapter input off) | It is not cross-graph comparable (design §1). Its value as an adapter input (A7) is out of scope unless Phase 5.3 succeeds and time remains. |
| Seeds | 3 (`seed ∈ {42, 43, 44}`) for every trained variant; report mean ± std and paired win/loss counts against SapBERT | Phase 4.5/4.6 differences were single-seed. |
| Hits@K | `ks: [1, 5, 10]` | Same as Sprint 4 |

## Conventions for every phase

- Google-style docstrings, tests in `tests/unit/`, `uv run pytest tests/unit -q` green.
- Each phase ends with an entry in `dev_docs/execution_logs/sprint_5.md` containing:
  - results tables (val, then test where allowed);
  - run ids;
  - the go/stop decision.
- Commit prefix `[Sprint-5.N]`.
- All commands use `--dataset ncit_doid_srga`. Every new option is **default-off**, so `ncit_doid`,
  `ncit_doid_supervised` and `attack_cwe` runs are unchanged.
- **Test discipline:**
  - choose everything (β, λ, τ, epochs, variant) on `val`;
  - evaluate test **once** per finished configuration;
  - never pick a test winner retroactively.

## Decision gates (summary)

| After | Go if (on `val`, mean over 3 seeds where trained) | Else |
|---|---|---|
| 5.1 | A1 (best β) > A0 | Record. Run 5.2 only to check supervision alone, skip 5.3, and write up. |
| 5.2 | A3 ≥ A0 (the floor holds) | Debug the loss/λ (λ=0 must equal A0). Do not add the graph adapter on a broken core. |
| 5.3 | A4 > max(A1, A3) by more than the seed std, **and** A5 (rewired) < A4 | Stop adding architecture. Report the best of A1/A3 as the Sprint 5 result in 5.4. |
| 5.4 | final test MRR > A0 in both modes | Document the negative result and its stratified breakdown. Stop. |
| 5.5 | A11 (SRGA-GCN) ≥ A3 − 1 seed std on val, **and** A13 (rewired) < A11 | Record that the paper's GCN tower does not transfer to the supervised task as an SRGA adapter; A3 stays the result. |
| 5.6 | A14 (SRGA-GAT) ≥ A11 − 1 seed std on val, **and** A16 (rewired) < A14 | Record that attention adds nothing over the paper's GCN in the supervised setting; A11 stays the paper-compatible variant. |

---

## Phase 5.0: Validation split and the SRGA lineage

**Capability:** a held-out `val` split of gold train pairs, and the `ncit_doid_srga` lineage with A0 baselines on
val and test. No model changes.

- [x] **Schema (approved by the developer on 2026-10-15).** Add `"val"` to `AlignmentSplit` in
      `data/schemas.py`. Only the `Literal` changes; the parquet schema (`split: string`) does not.
- [x] **Generic val carve in ingest** (no adapter change):
  - `IngestAlignmentsStep` reads an optional `dataset.alignments.val_split: {fraction, seed}` (it sits beside the
    `equivalence` block, which is still the only part passed to the adapter);
  - it samples that fraction of distinct `node_1_id`s among `train` rows with `random.Random(seed)` and relabels
    all their train rows `val`;
  - when the key is absent, nothing changes;
  - add `n_alignments_by_split` with `val` to `stats.json`.
- [x] `training_data`: `_HELD_OUT_ALIGNMENT_SPLITS = {"test", "val"}`. Val endpoints become `held_out`, so the
      existing leakage assertion covers them. `supervised_positives` already takes only `split == "train"`, and gold
      targets of every split (including val) are already excluded from negatives.
- [x] `configs/datasets/ncit_doid_srga.yaml`: copy of `ncit_doid_supervised.yaml`, with `id: ncit_doid_srga` and
      `alignments.val_split: {fraction: 0.15, seed: 42}`.
- [x] Ingest with `--set "ingest.split.fractions=[1.0,0.0,0.0]"` (as in 4.0). Copy the three embedding stores into
      `artifacts/embeddings/ncit_doid_srga/` and verify `node_id` order and `text_sha256` (Phase 4.0 procedure).
- [x] Tests:
  - the val carve is deterministic for a seed;
  - it is disjoint from train and test;
  - it moves whole sources;
  - no val endpoint appears in `examples.parquet`;
  - the missing-key path is unchanged;
  - `gold_alignment` with `split: val` queries only val pairs.
- [x] Evaluate **A0** (SapBERT, BERT-base) on `val` (global) and `test` (both modes) with new experiment
      `configs/experiments/ncit_doid_srga_compare.yaml`. The evaluation blocks are `gold_val` (`split: val`,
      `modes: [global]`) and `gold_test` (`split: test`, `modes: [global, candidates]`).

**Training objective / negatives:** none (no training).
**Ablation:** A0 bar.
**Acceptance:**
- `val` holds about 15% of train sources (≈ 200 pairs);
- train ∩ val ∩ test = ∅ on pairs and on endpoints used in examples;
- A0 **test** metrics equal Sprint 4 exactly (MRR 0.448 global / 0.519 candidates), because test pairs and stores are
  unchanged;
- A0 val Hits@1/5/10/MRR recorded.

---

## Phase 5.1: SRGA output vector with a training-free adapter (A1)

**Capability:** the SRGA export `z = [x̂ ; √λ·normalize(h)]` (design §5.1), with `h` from a **parameter-free**
neighbour adapter. This builds the residual floor and the edge-direction plumbing that Phase 5.3 reuses, and puts the
§2 probe through the real pipeline.

- [x] `gcl_train/models.py`: `ResidualAdapterEncoder`, selected by `gcl_train.model.encoder: residual_adapter`.
  - Params: `adapter: neighbor_mean | mlp | relational` (5.1 uses `neighbor_mean`), `lambda`, `adapter_dim`,
    `directional`.
  - It returns `z`, with the SapBERT input used frozen (`requires_grad=False`).
  - `neighbor_mean` builds `h = mean(x̂ over 1-hop neighbours)`, or with `directional: true`,
    `h = [mean parents ; mean children]`. A node without neighbours gets `h = 0`.
- [x] `gcl_train/pipeline.py` (load):
  - build `edge_type` (0 = stored direction src→dst = child→parent, 1 = reversed) beside the symmetrized
    `edge_index`;
  - the existing `gcn`/`gat` path ignores it.
- [x] `trainer`/pipeline: when the model has no trainable parameters (`neighbor_mean`), skip the optimizer loop
      (`epochs` ignored) and go straight to export, recording `n_trainable_params: 0` in the manifest.
- [x] Runs:
  - β grid `lambda ∈ {0, 0.1, 0.25, 0.5, 1}` × `directional ∈ {false, true}`;
  - `model_id = srga-nb-l<λ>[-dir]`, using `--set gcl_train.model.encoder=residual_adapter --set
    gcl_train.model.adapter=neighbor_mean --set gcl_train.model.lambda=<λ>`;
  - training_data: supervised positives (needed only for the run lineage);
  - add them to `ncit_doid_srga_compare.yaml` and evaluate on **val**.
- [x] Pick the best β (and direction) on val, then evaluate that one configuration on test.
- [x] Tests:
  - `lambda=0` export ranks identically to SapBERT (metrics equal A0);
  - `z` has 768 + h columns;
  - the directional `h` uses the correct parent/child sides on a 3-node fixture;
  - the `gcn` path is unchanged (existing tests pass).

**Training objective / negatives:** none (parameter-free).
**Ablation:** **A1** vs A0, plus directional vs undirected (A6, training-free form).
**Acceptance:**
- `srga-nb-l0` equals A0 on val and test (floor check);
- the best-β A1 is recorded on val, and once on test (expected about 0.469 global MRR from design §2).

**Decision:** gate 5.1 (above).

---

## Phase 5.2: Trainable SRGA core without edges (A3)

**Capability:** the trainable residual adapter and the design §6 objective, with **no graph input**: `h = MLP(x̂)`.
This builds the loss and trainer path once, and measures what supervision adds *without* structure (the A3 control
for 5.3).

- [x] `ResidualAdapterEncoder` with `adapter: mlp`:
  - `h = MLP_2(x̂)` (768 → 256 → 256, ReLU, LayerNorm, dropout 0.2);
  - SapBERT frozen; λ fixed;
  - final layer initialized small (std 1e-3), so `z` starts close to SapBERT plus noise.
- [x] `gcl_train/losses.py` + `trainer.py`: register `infonce_full` (design §6):
  - **Positives:** distinct `(anchor_id, text_pos_id)` from `examples.parquet` (train split). Sampled negatives
    are ignored.
  - **Candidates:** all non-held-out universe nodes with the positive's `graph_role` (≈ 5k DOID). The reverse term
    uses all non-held-out NCIT nodes, with weight `gcl_train.loss.reverse_weight` (default 0.5).
  - **Mask:** other gold targets of the anchor (any split), from the ingest `alignments.parquet`, which is resolved
    through the training_data manifest. Held-out nodes are already absent.
  - Score `s = cos(z_a, z_d)`, temperature `gcl_train.loss.temperature` (start 0.05).
  - When `loss.name == infonce_full`, the graph-view loss is skipped (`alpha` ignored). This removes the
    within-graph triplet loss from the retrieval objective.
- [x] Runs:
  - `lambda` = the β chosen in 5.1 (fixed), τ ∈ {0.05, 0.1}, epochs ∈ {50, 100, 200}, Adam lr 1e-3, weight decay
    1e-4;
  - pick τ and epochs on val with seed 42, then run 3 seeds of the chosen setting;
  - `model_id = srga-mlp-l<λ>-s<seed>`.
- [x] Tests (`test_losses.py`, `test_trainer.py`):
  - `infonce_full` equals a hand-computed value on a 2-anchor × 4-candidate fixture;
  - masked gold targets contribute nothing;
  - held-out nodes never appear among candidates;
  - the reverse term weight is respected;
  - the `triplet`/`infonce` paths are unchanged;
  - with `lambda=0` the loss gradient on the adapter is zero.

**Training objective / negatives:**
- full-softmax InfoNCE over all trainable DOID (and NCIT for the reverse term);
- no mining and no sampled negatives;
- false-negative guard = gold mask (any split). The label-match guard is skipped: design data found 1 case.

**Ablation:** **A3** (supervision without structure) vs A0 and A1.
**Acceptance:**
- loss decreases with no NaN;
- training-set MRR rises above SapBERT's (the under-fitting problem of Sprint 4 is gone);
- val MRR (mean of 3 seeds) ≥ A0;
- A3 vs A1 recorded.

**Decision:** gate 5.2.

---

## Phase 5.3: Directional graph adapter, the full SRGA (A4), and its structural controls (A5, A6)

**Capability:** the design §5.2 adapter. The parameter-free neighbour means from 5.1 are combined with the trainable
residual core from 5.2. Each component is introduced only after 5.1 and 5.2 pass.

- [x] `adapter: relational`:
  - `a_n = [x̂_n ; mean_parents x̂ ; mean_children x̂ ; x̂_n ⊙ mean_parents x̂ ; log(1+in_deg), log(1+out_deg)]`;
  - `h = MLP_2(LayerNorm(a_n + RGCN_1(a)))`, one relational layer with separate parent and child weights
    (built from `edge_type`);
  - dropout 0.2, weight decay 1e-4;
  - `directional: false` collapses to one undirected relation (the A6 control).
- [x] **Rewiring control (A5):**
  - `gcl_train.model.edge_rewire: {enabled, seed}` applies a degree-preserving double-edge swap *within each
    graph role* before training;
  - it never creates cross-graph edges and leaves universe and examples untouched.
- [x] Runs, with the same objective, λ, τ and epochs as chosen in 5.2, and 3 seeds each:
  - **A4** `srga-rel-l<λ>-s<seed>`;
  - **A5** `srga-rel-rewired-l<λ>-s<seed>`;
  - **A6** `srga-rel-undir-l<λ>-s<seed>`.
- [x] Tests:
  - the relational layer uses separate weights per `edge_type` (parent-only and child-only fixtures give different
    `h`);
  - rewiring preserves every node's degree and graph role and is seed-deterministic;
  - the `relational` export keeps the `[x̂ ; √λ ĥ]` layout.

**Training objective / negatives:** same as 5.2 (full-softmax InfoNCE with gold mask, no structural triplet).
**Ablation:**
- **A4** vs **A3** (does structure add to supervision);
- **A4** vs **A1** (does training add to free structure);
- **A5** (graph vs capacity);
- **A6** (directionality).

**Acceptance:**
- val table of A0, A1, A3, A4, A5, A6 (mean ± std over 3 seeds);
- paired win/loss counts against A0 on val.

**Decision:** gate 5.3. A4 must beat both A1 and A3 by more than the seed std, and A5 must be clearly below A4.
Otherwise SRGA's graph adapter is not justified on this data, and the sprint reports the best simpler variant.

---

## Phase 5.4: Final test evaluation, λ curve, and stratified analysis

**Capability:** a single, honest test readout of the variant selected on val, plus the evidence that the gain comes
from structure.

- [x] **Selected variant:** the best of {A1, A3, A4} on val (per gates 5.1–5.3). Evaluate it, A0 and the
      runner-up on **test** (`global` + `candidates`, 3 seeds where trained). This is the one test evaluation of the
      sprint.
- [x] **A9 λ curve:** for the selected trained variant, retrain at `lambda ∈ {0.1, 0.25, 0.5, 1}` (seed 42) and
      evaluate on val. Confirm the curve peaks above λ = 0 and that λ = 0 equals A0.
- [x] **Stratified win/loss report** on test, for the selected variant vs A0, split by:
  - whether SapBERT ranks the gold target first;
  - the hop distance (≤ 2 vs > 2) between SapBERT's top confuser and the gold target in DOID.

  Report counts fixed, broken and unchanged. This is an analysis script whose output goes in the execution log; it
  needs no pipeline change.
- [x] Plot: extend `configs/plots/` with `ncit_doid_srga_compare.yaml` (existing `bar` renderer: `hits_at_k` and
      `mrr` faceted by mode).
- [x] README: add the `ncit_doid_srga` config and the SRGA commands. Leave existing sections unchanged.

**Training objective / negatives:** unchanged from the selected phase.
**Ablation:**
- **A9** (residual floor and optimum);
- stratified A4 (or the selected variant) vs A0, from design §7.

**Acceptance:**
- test tables (Hits@1/5/10, MRR; global and candidates; mean ± std);
- the selected variant's test MRR > A0 (0.448 global / 0.519 candidates) in both modes, with fixes concentrated in
  the "SapBERT wrong" stratum and few breaks in the "SapBERT right" stratum;
- if this fails, the execution log states the negative result plainly, with the stratified breakdown.

**Expected outputs:**
- `artifacts/evaluations/ncit_doid_srga/<run_id>/metrics.{parquet,json}` with `gold_val`/`gold_test` scopes;
- plot run figures;
- the Sprint 5 summary in `execution_logs/sprint_5.md`.

---

## Phase 5.5: The paper's GCN text branch as the SRGA adapter (A11), with no-floor (A12) and rewired (A13) controls

**Why.** The paper's unsupervised `attack_cwe` model is a `DualEncoder`: a 2-layer `GCN` over the text embeddings, plus
a linear text projection, plus a DeepWalk branch, fused by `fc3`. This phase asks whether the **same GCN text branch**,
with DeepWalk removed, also works for the supervised NCIT–DOID task when it is:
- placed beside frozen SapBERT as the SRGA residual (so λ = 0 is still SapBERT); and
- trained with the Phase 5.2 full-softmax InfoNCE.

If it does, the paper can present one architecture family for both settings: unsupervised GCL and supervised alignment.

It also answers an open Sprint 4 question. The Sprint 4 GCN lost to SapBERT, but that run changed two things at once:
- it had no SapBERT floor (a random projection to 128-d);
- it used a weak loss (triplet with easy negatives).

The no-floor control A12 keeps the full-softmax loss and removes the floor, which separates the two causes.

**Capability:** `ResidualAdapterEncoder` gains `adapter: gcn`, which reuses the existing `GCN` class unchanged, and
`gcl_train.model.residual_floor` (default `true`).

- [x] `adapter: gcn` (`h` built like the paper's text branch, DeepWalk removed):
  - `t_n = Linear(768 → 256)(x̂_n)` (the paper's `fc1`);
  - `g = GCN(x̂, edge_index)`: the existing 2-layer `GCN` tower (`GCNConv 768 → 256`, ReLU, `GCNConv 256 → 256`),
    undirected, symmetric normalisation with self-loops, exactly as the `gcn` encoder uses it;
  - `h_n = Linear(512 → 256)(Dropout(0.2)([t_n ; g_n]))` (the paper's `fc3`); last layer std 1e-3, zero bias;
  - export: `z_n = [x̂_n ; √λ · h_n / ||h_n||]`, λ = 0.25. About 591k trainable parameters.
- [x] `gcl_train.model.residual_floor: false` (A12 only):
  - export `z_n = h_n` with no SapBERT block;
  - default `true`, so A1–A6 and every existing run are unchanged.
- [x] `edge_rewire` (Phase 5.3) is reused as-is for A13.
- [x] Runs, with the same objective and settings as A3 so the only difference is the adapter: `infonce_full`,
      τ 0.05, reverse weight 0.5, Adam lr 1e-3, weight decay 1e-4, λ 0.25, 3 seeds (42/43/44).
  - **A11** SRGA-GCN: 50 epochs. `model_id = srga-gcn-l0.25-s<seed>`.
  - **A12** GCN without the floor: it learns the whole embedding, so first pick epochs ∈ {50, 100, 200} on val with
    seed 42, then run 3 seeds. `model_id = gcn-full-e<epochs>-s<seed>`.
  - **A13** A11 on a rewired graph (`edge_rewire.enabled=true`, `seed=<seed>`). `model_id = srga-gcn-rewired-l0.25-s<seed>`.
  - New experiment `configs/experiments/ncit_doid_srga_gcn.yaml`: A0, A1, A3, A4, A11, A12, A13; `gold_val` +
    `gold_train`.
- [x] Val analysis:
  - mean ± std table;
  - paired wins/losses against A0, and A11 vs A3 / A4 (`scripts/srga_paired_ranks.py`).
- [x] Test, once per finished configuration, after the val table is recorded:
  - add A11 and A12 to `ncit_doid_srga_compare.yaml` and evaluate;
  - compare against the Phase 5.4 numbers of A0, A3 and A4 (same test pairs, same models);
  - run `scripts/srga_stratified.py` for A11 vs A0, beside A3 and A4.
- [x] Tests (`test_models.py`, `test_pipeline.py`):
  - `adapter: gcn` output has 768 + 256 columns, and its SapBERT block is the unit text vector;
  - unlike `mlp`, its `h` changes when edges change;
  - `lambda = 0` gives zero gradient on the adapter;
  - `residual_floor: false` exports `h` only (256 columns);
  - a pipeline run with `infonce_full` + `gcn` adapter completes;
  - the `gcn`/`gat` `DualEncoder` tests pass unchanged.

**Training objective / negatives:** same as Phases 5.2–5.3 (full-softmax InfoNCE over every non-held-out candidate,
gold-target mask of any split, no sampled negatives, no graph-view triplet).

**Ablation:**
- **A11 vs A3:** does the paper's GCN branch add to supervision without a graph?
- **A11 vs A4:** paper GCN (undirected, 2 hops, symmetric norm) vs the directional relational adapter.
- **A12 vs A11:** what the SapBERT floor is worth with a strong loss. This answers whether Sprint 4 failed because of
  the loss or the missing floor.
- **A13 vs A11:** is it the graph or the capacity?

**Acceptance:**
- loss decreases with no NaN for all variants;
- val table (mean ± std, 3 seeds) and paired counts for A0, A1, A3, A4, A11, A12, A13;
- test table for A11 and A12 in both modes, read once, next to the Phase 5.4 A0/A3/A4 rows;
- A12's result interpreted against Sprint 4 (GCN supervised, triplet: test MRR 0.223 global).

**Decision:** gate 5.5.
- A11 is reported as the paper-compatible supervised variant if it is within one seed std of A3 (or better) on val,
  and A13 is clearly below it. Otherwise A3 stays the result, and the log says the GCN tower does not transfer as an
  adapter.
- Either way, the selection decision for Sprint 5 (A3, Phase 5.4) is **not** reopened on test. A11 is an added
  finding, not a re-selection.

**Caveats:**
- **Val is small.** Phase 5.4 showed 211 val pairs cannot resolve differences of about 0.01–0.02 MRR (A4 tied A3 on val
  but led on test). Prefer the paired counts and the test readout for interpretation, and state the val noise.
- **Over-smoothing.** A 2-layer undirected GCN mixes parents and children. The design's training-free probe showed GCN
  smoothing lowers Hits@1, so watch Hits@1 against A3.
- **Scope.** The paper's `attack_cwe` setting has no gold pairs, so this phase does not rerun anything there; it only
  shows the GCN text branch is usable in the supervised setting.

---

## Phase 5.6: The paper's GAT text branch as the SRGA adapter (A14), with no-floor (A15) and rewired (A16) controls

**Why.** Phase 5.5 showed that the paper's GCN text branch works in the supervised setting, both beside frozen SapBERT
(A11) and without it (A12). The paper's `DualEncoder` also has a GAT variant (`gcl_train.model.encoder: gat`). This phase
repeats 5.5 with the existing `GAT` tower in place of the `GCN` one, so the paper can report both message-passing
families in the supervised setting.

The question this adds: does **learned attention over neighbours** beat the GCN's fixed symmetric average? Attention
could down-weight unhelpful neighbours, for example a parent that is also a SapBERT confuser. That may reduce breaks
where A4 broke many.

**Capability:** `ResidualAdapterEncoder` gains `adapter: gat`, which reuses the existing `GAT` class unchanged.
`residual_floor` and `edge_rewire` are reused from Phases 5.5 and 5.3.

- [x] `adapter: gat` (built like the 5.5 `gcn` adapter, with the tower swapped):
  - `t_n = Linear(768 → 256)(x̂_n)` (the paper's `fc1`);
  - `g = GAT(x̂, edge_index)`: the existing 2-layer `GAT` tower:
    - layer 1 is `GATConv(768 → 256 / H, heads=H)`, concatenated to 256;
    - then ReLU;
    - layer 2 is `GATConv(256 → 256, heads=1, concat=False)`;
    - `H = gcl_train.model.gat_heads[0]` (default 8, so 32 per head);
  - `h_n = Linear(512 → 256)(Dropout(0.2)([t_n ; g_n]))` (the paper's `fc3`); last layer std 1e-3, zero bias;
  - export: `z_n = [x̂_n ; √λ · h_n / ||h_n||]` with the floor (λ = 0.25), or `z_n = h_n` with `residual_floor: false`;
  - the per-head width `256 / H` keeps the tower about the same size as A11's GCN, at about 590k parameters in total.
    So A14 vs A11 compares attention against averaging, not model size. `adapter_dim` must be divisible by `H`;
    otherwise raise a `ValueError`.
- [x] Runs, with the same objective and settings as 5.5 (`infonce_full`, τ 0.05, reverse weight 0.5, Adam lr 1e-3,
      weight decay 1e-4, λ 0.25, seeds 42/43/44):
  - **A14** SRGA-GAT: 50 epochs (same as A11). `model_id = srga-gat-l0.25-s<seed>`.
  - **A15** GAT without the floor:
    - pick epochs ∈ {50, 100, 200} on val with seed 42, the same grid as A12, so A15 vs A12 is a like-for-like
      comparison;
    - then run 3 seeds;
    - `model_id = gat-full-e<epochs>-s<seed>`.
  - **A16** A14 on a rewired graph (`edge_rewire.enabled=true`, `seed=<seed>`). `model_id = srga-gat-rewired-l0.25-s<seed>`.
  - New experiment `configs/experiments/ncit_doid_srga_gat.yaml` (A0, A3, A11, A12, A14, A15, A16; `gold_val` +
    `gold_train`), and `ncit_doid_srga_gat_sel.yaml` for the A15 epoch grid.
- [x] Val analysis:
  - mean ± std table;
  - paired wins/losses against A0, and A14 vs A11, A15 vs A12, A14 vs A16 (`scripts/srga_paired_ranks.py`).
- [x] Test, once per finished configuration, after the val table is recorded:
  - add A14 and A15 to `ncit_doid_srga_compare.yaml` and evaluate;
  - compare against the Phase 5.4/5.5 rows (same test pairs, same models);
  - run `scripts/srga_stratified.py` for A14 and A15 next to A11 and A12;
  - run `scripts/srga_pool_diagnostic.py` (held-out-only / trainable-only pools) for A14 and A15, as in 5.5 §4.
- [x] Attention readout (analysis only, no pipeline change): a script that rebuilds one A14 model per seed from its
      final checkpoint and reads the first-layer attention weights (`GATConv(..., return_attention_weights=True)`).
  - Report the mean attention mass each node gives to its self-loop, its parents (`edge_type` 1) and its children
    (`edge_type` 0), split by graph role.
  - Also report it for the val/test query sources whose SapBERT top confuser is a 1-hop neighbour, compared with all
    other sources.
  - This shows whether attention learned to treat parents and children differently, which the undirected GCN cannot do.
- [x] Tests (`test_models.py`, `test_pipeline.py`):
  - the `gat` output has 768 + 256 columns and its SapBERT block is the unit text vector;
  - its `h` changes when edges change;
  - the parameter count matches the GCN adapter within 1%;
  - `lambda = 0` gives zero gradient on the adapter;
  - `residual_floor: false` exports `h` only;
  - `adapter_dim` not divisible by `gat_heads[0]` raises;
  - a pipeline run with `infonce_full` + `gat` completes, with and without the floor;
  - the `gcn`/`gat` `DualEncoder` and the 5.5 `gcn` adapter tests pass unchanged.

**Training objective / negatives:** same as Phases 5.2–5.5 (full-softmax InfoNCE over every non-held-out candidate,
gold-target mask of any split, no sampled negatives, no graph-view triplet).

**Ablation:**
- **A14 vs A11:** learned attention vs the GCN's symmetric average, at matched size.
- **A15 vs A12:** the same comparison without the SapBERT floor.
- **A14 vs A15:** what the floor is worth for GAT (MRR vs breaks of SapBERT-correct queries).
- **A16 vs A14:** graph or capacity.

**Acceptance:**
- loss decreases with no NaN for all variants;
- val table (mean ± std, 3 seeds) and paired counts for A0, A3, A11, A12, A14, A15, A16;
- test table for A14 and A15 in both modes, read once, next to the A0/A3/A4/A11/A12 rows;
- stratified fixed/broken and pool-diagnostic tables for A14 and A15;
- the attention readout table.

**Decision:** gate 5.6.
- GAT is reported as an equivalent or better paper-compatible variant if A14 is within one seed std of A11 (or better)
  on val, and A16 is clearly below A14. Otherwise the log records that attention adds nothing over the GCN here, and
  A11 stays the paper-compatible variant.
- As in 5.5, no earlier selection is reopened on test. A14 and A15 are added findings.

**Caveats:**
- **GAT is not deterministic.** The seeding module notes that GAT kernels may be non-deterministic, so expect more seed
  spread than A11. Report the per-seed values.
- **Val is small** (211 pairs). Lean on the paired counts and the single test readout, as in 5.4 and 5.5.
- **Attention is a weak explanation.** The attention readout describes what the model attends to, not why it ranks
  correctly. Treat it as descriptive.
- **Duplicate edges.** The GAT tower uses the symmetrized edge list with duplicates, as the paper's `gat` encoder does.
  A pair stored under two `edge_sources` therefore gets two attention slots.

---

## Out of scope (pursue only if Phase 5.4 succeeds, in a later sprint)

- **Design §5.3 per-node confidence gate (A8).** Per-graph or learned λ.
- **DeepWalk as adapter input (A7).**
- **Variant B** additive residual (A10).
- **Anchor-based re-ranker** (design §8).
- **A2** score fusion of the Sprint 4 GCN (it needs a two-store "fusion" method kind).
- **Hard-negative band sampling and label-match masking.** Full softmax already covers all negatives.
- **Re-embedding SapBERT with `[CLS]`/label-only text** (it changes the bar, so it belongs in its own experiment).
- **Changes to the ingest adapters, embedding pipeline, or the `gold_alignment` evaluator.**

**Caveats for the write-up.**
- Training is transductive: val/test nodes take part in message passing as features only, never in any example.
- About 200 val pairs make selection noisy. Prefer the coarse grids above, and report seed std and paired win/loss
  counts, not just means.
