# Design: SapBERT-residual graph adapter for supervised NCIT–DOID alignment

**Status:** proposal (no code yet). **Goal:** beat SapBERT on held-out NCIT–DOID gold-pair retrieval (Hits@K, MRR) by
adding graph structure *on top of* SapBERT instead of mixing it in.

**Builds on:**
- [execution_logs/sprint_4.md](../execution_logs/sprint_4.md) (Phases 4.4–4.6);
- [data_analysis/hard_negative_selection.md](../data_analysis/hard_negative_selection.md).

**Data used:** all numbers are from the `ncit_doid_supervised` lineage:
- ingest `20261014T185110Z-c44babf2`;
- SapBERT `20261009T182919Z-95011340`;
- `gcn-supervised-randneg` `20261015T132141Z-c198456f`;
- the `gold_alignment` evaluator, on 3,280 test pairs, or the 1,406 train pairs where stated.

The probes in Section 2 were computed in memory and are not saved artifacts.

---

## 0. Recommendation in one paragraph

Do not fuse four representations into one learned vector. Keep **frozen SapBERT as an untouched residual term** and
learn only a **graph correction** next to it:

```
z_n = [ x̂_n ; √λ_g(n) · ĥ_n ]          score(n, d) = cos(z_n, z_d) ≈ cos_SapBERT + λ · cos_graph
```

- `x̂` is the unit SapBERT vector.
- `ĥ` comes from a small **directional graph adapter** (separate parent and child aggregation of SapBERT features,
  with optional DeepWalk input).
- `λ_g` is a **per-graph** scalar, initialized at 0 and chosen on a validation split, so the model starts at
  SapBERT and λ = 0 is always one of the options considered.

Train `ĥ` with:
- **full-softmax InfoNCE** over all trainable DOID nodes;
- other gold targets masked out;
- the within-graph structural triplet loss removed from the retrieval objective.

A per-node 4-way gate is not recommended as the first design. At most, add a **bounded per-node confidence scalar**
inside the residual term later. Two results back this direction:
- a *training-free* version (λ = 0.25 on 1-hop neighbour means, tuned on train pairs) already gives 0.469 MRR against
  SapBERT's 0.448;
- adding the current GCN's score to SapBERT's reaches about 0.49 (Section 2).

---

## 1. Review of the current setup

| Part | Current state | Issue for cross-graph retrieval |
|---|---|---|
| Model | `DualEncoder`: `fc3([fc1(x_sap), GCN(x_sap), fc2(x_dw), GCN(x_dw)])`, 768+128 → **128**, random init | SapBERT passes through a random projection, so its ranking must be relearned from 1,383 pairs. |
| Message passing | 2-layer GCN, **undirected** (`gcl_train` adds both directions); edges are child→parent in `edges.parquet` | Parents and children are averaged together, so a node is blurred toward its own granularity confusers. |
| DeepWalk | one input channel into the shared output | NCIT and DOID are disconnected, so their DeepWalk spaces are not aligned. Cross-graph DeepWalk cosine is at chance (MRR 0.0007). |
| Objective | `α·triplet(text) + (1−α)·triplet(graph)`, margin 1.0, on **one shared** vector | The graph term pulls each NCIT anchor toward its structural neighbours: the parent/child concepts that are SapBERT's main confusers. |
| Negatives | bottom-k (Phase 4.2) or random (Phase 4.6), 5 per positive | Mostly easy: the loss collapses (bottom-k) or stays weak (random). |
| Evaluation | `gold_alignment`: cosine over the exported vector, `global` and `candidates` modes, test split only | Sound. There is **no validation split**, so nothing can be tuned without touching test. |

---

## 2. Evidence (training-free probes)

These check whether structure helps when it is **added beside** SapBERT rather than blended in. For a concatenation
`[x̂ ; √β·ĥ]`, cosine ranking equals ranking by `cos_x + β·cos_h`, so each probe was scored with the unchanged
evaluator. `nb1` is the unit mean of a node's 1-hop neighbours' SapBERT vectors, and `nb2` is the same at 2 hops.

| Method | train MRR (global) | test Hits@1 | test Hits@10 | test MRR (global) | test MRR (candidates) |
|---|---|---|---|---|---|
| SapBERT | 0.435 | 0.342 | 0.654 | 0.448 | 0.519 |
| GCN-style smoothing `D^-½(A+I)D^-½ x` (no training) | 0.408 | **0.260** | **0.709** | 0.412 | 0.470 |
| neighbour mean only (`nb1`) | 0.178 | 0.108 | 0.360 | 0.184 | 0.230 |
| **SapBERT + 0.25·`nb1`** (β picked on train) | **0.464** | 0.354 | 0.701 | **0.469** | **0.529** |
| SapBERT + 0.1·`nb1` + 0.05·`nb2` | 0.452 | 0.355 | 0.682 | 0.465 | 0.528 |
| SapBERT + λ·GCN-randneg, λ=0.25 | (biased) | 0.370 | 0.730 | **0.491** | **0.552** |
| SapBERT + λ·GCN-randneg, λ=0.5 | (biased) | 0.364 | 0.736 | 0.489 | 0.554 |
| GCN-randneg alone (Phase 4.6) | — | 0.169 | 0.522 | 0.283 | 0.371 |

What the probes show:
1. **Smoothing trades precision for recall.** It lowers Hits@1 (0.342 → 0.260) but raises Hits@10 (0.654 → 0.709).
   Structure carries real information, but replacing a node's text with its neighbourhood blurs the fine distinctions
   that decide rank 1.
2. **Structure helps as a residual.** One scalar β, chosen on train pairs and with zero trained parameters, beats
   SapBERT on every test metric.
3. **The current trained GCN has complementary signal that its architecture throws away.** On its own it reaches
   0.283 MRR, but beside SapBERT it reaches about 0.49. Caveat: λ here was read off the **test** set, because train-pair
   selection is biased once the GCN has trained on those pairs. This is a diagnostic only, not a result, and needs a
   validation split (Section 6).
4. **Cross-graph anchors are available.** For 85.9% of test pairs, the NCIT source has a gold train anchor within 2
   hops. For 63.7%, that anchor's DOID partner is also within 2 hops of the gold target (24.3% at 1 hop). How well
   this separates the gold target from its near confusers was **not** measured.

---

## 3. Why direct concatenation underperforms SapBERT

1. **Relearning instead of keeping.** The fused vector passes through a random linear map to 128 dimensions. Even on
   its *own* training pairs the Phase 4.2 GCN scores train MRR 0.267 against SapBERT's 0.435, so it under-fits
   SapBERT's geometry with only 1,383 pairs.
2. **The smoothing destroys exactly the needed distinctions.** Undirected GCN averaging makes "nephritis" look like
   "chronic pyelonephritis" (Section 2, probe 1). The hard-negative analysis showed parent/child confusers are a main
   SapBERT failure mode.
3. **An unaligned channel.** DeepWalk coordinates mean nothing *across* the two disconnected graphs, so mixing them
   into the cross-graph vector adds noise that the model must learn to cancel.
4. **A conflicting objective.** The graph triplet loss actively pulls an anchor toward its structural neighbours,
   working against discrimination from those same neighbours.
5. **Weak training signal.** Easy negatives (Phase 4.2 loss → 0.004) leave the fused vector unconstrained where it
   matters, which is around the close competitors.
6. **No floor.** Nothing in the architecture can recover pure SapBERT, so any graph error directly degrades ranking.

---

## 4. Is per-node gating over the four sources sound?

**Partly.** Adaptive weighting is the right instinct, because some nodes have informative neighbourhoods and some do
not. A per-node softmax gate over `{SapBERT, GCN-text, GCN-graph, DeepWalk}` is still a poor *first* design:

- **No guarantee.** A softmax can move SapBERT's weight toward 0 for any node, so the SapBERT floor is lost.
- **Two of the four sources are redundant or invalid.** GCN-text is SapBERT smoothed (Section 2 shows it is better
  used as a residual), and DeepWalk is not cross-graph comparable. The gate would mainly learn to suppress them.
- **Too little data.** With 1,383 positives, a per-node gate network can fit spurious gate patterns. Test nodes'
  gates are never supervised.
- **Asymmetry problems.** A query and a candidate with different mixtures are compared through different
  sub-spaces, and cosine normalization then depends on each candidate's mixture.

**What to keep from the idea:** a *bounded scalar* confidence `c_n ∈ [0, 1]` per node that scales only the **graph
residual**, never SapBERT (Section 5.3). It gets the adaptivity without the risk.

**Weight granularity:**
- *Global λ* works (probe 2).
- *Per-graph λ* (NCIT vs DOID, 2 scalars) is cheap and justified, because the hierarchies differ in depth and density.
- *Per-node* weights only as the bounded residual confidence above, regularized toward its mean.

---

## 5. Proposed architecture: SapBERT-residual graph adapter (SRGA)

### 5.1 Output vector

```
x̂_n  = normalize(SapBERT_n)                         (frozen, 768-d)
h_n  = Adapter(n)                                    (trainable, d_h = 256)
z_n  = [ x̂_n ; √λ_g(n) · c_n · normalize(h_n) ]     (exported; the evaluator uses cos(z_n, z_d))
```

- **The SapBERT block is never modified.** At λ = 0, `z` ranks exactly like SapBERT.
- **λ settings:**
  - `λ_g ∈ {λ_ncit, λ_doid}` is either learned (softplus, initialized ≈ 0.05) or fixed from a validation grid;
  - start with **fixed λ from a grid** (0, 0.1, 0.25, 0.5, 1) chosen on validation, because that is the simplest
    and safest choice.
- `c_n` (Section 5.3) is 1 in the base model.
- The exported vector works with the unchanged `gold_alignment` evaluator, because the cosine of the concatenation
  is the fused score up to per-node norms. Training uses the same cosine, so train and evaluation agree.

### 5.2 Directional graph adapter

The input features are frozen, with no smoothing of the SapBERT block itself:

```
p_n = mean_{parents(n)} x̂        c_n' = mean_{children(n)} x̂        (zeros + indicator if none)
a_n = [ x̂_n ; p_n ; c_n' ; x̂_n ⊙ p_n ; deg features ; (optional) DW_n ]
h_n = MLP_2(LayerNorm(a_n + RGCN-1(a)))    # one relational layer: separate weights for parent/child edges
```

- **Direction matters.** Parents and children carry different meaning ("is-a X" vs "has subtypes Y"), and averaging
  them is what blurs granularity. `edges.parquet` already stores src = child, dst = parent.
- **Shallow (1 relational layer).** 2-hop information is already in `p`/`c'`. Deeper GNNs over-smooth (probe 1).
- **DeepWalk** is only an adapter *input* (structural role: depth, branching), never a block of `z`. Ablation A7
  decides whether to keep it.
- **Self features `x̂_n` are included**, so `h` models "this concept *in its context*", not just its context.
- **Regularization:** dropout 0.2, weight decay 1e-4, and early stopping on validation MRR.
- **Size:** about 1M parameters, versus the current about 0.5M.

### 5.3 Optional confidence gate (later step)

`c_n = σ(w · f_n + b)`, with `f_n` = [log degree, has parents, has children, cos(x̂_n, p_n), cos(x̂_n, c_n')] and `b`
initialized so `c ≈ 0.5`. Add an L2 penalty pulling `c` toward its mean. Use one scalar per node that scales only the
residual. Add it only if ablation A8 shows a gain over per-graph λ.

### 5.4 Variant B (follow-up only)

An additive adapter inside the SapBERT space, `z = normalize(x̂ + γ_g · W_o h)` with `W_o` zero-initialized and an
L2 penalty on `γ W_o h`. It is more expressive (it can rotate SapBERT locally) but has no exact floor. Try it only
after SRGA is established.

---

## 6. Training objective and negatives

**Main loss: full-softmax InfoNCE**, NCIT→DOID, over all non-held-out DOID nodes:

```
L = − mean_{(a,p) ∈ gold-train}  log  exp(s(a,p)/τ) / Σ_{d ∈ DOID_train ∖ (gold(a) ∖ {p})} exp(s(a,d)/τ)
s(a,d) = cos(z_a, z_d),   τ ≈ 0.05 (tuned on validation)
```

- **All 5,195 trainable DOID nodes are negatives.** The hard competitors are included automatically, which matches
  the `global` evaluation exactly. Full-batch costs about 1,383 × 5,195 logits, which is trivial.
- **Mask the other gold targets** (any split), and exclude held-out nodes (already absent).
- **Symmetric term** (weight 0.5): DOID→NCIT over all non-held-out NCIT nodes.
- **Optional hard-negative emphasis:** up-weight the SapBERT top-50 band, or add a margin term over it, if A4 shows the
  top ranks still lag.
- **False-negative guard:** mask candidates with an identical normalized label or shared synonym, and optionally
  candidates with `cos_sap(a,d) > cos_sap(a,p) + 0.1`.

**Remove the within-graph structural triplet loss** from the retrieval objective. If a structural self-supervised
signal is wanted, put it on an **auxiliary head** of `h` only (for example, predicting parent vs child), so it cannot
pull `z` toward confusers.

**If staying on `examples.parquet` for a first step:** use the existing `infonce` loss (in-batch negatives) with
band-plus-random negatives (ranks 3–50 plus random, from the `supervised` sampler). This needs no trainer change, but
it is weaker than full softmax.

**Validation split (required).** Hold out 15% of gold train pairs and treat their endpoints as held out, like test.
Use them for λ, τ, epochs and every ablation decision. Touch test only once per finished configuration.

**Seeds.** Use 3 seeds per configuration and report mean ± std. The Phase 4.5/4.6 differences were single seeds.

---

## 7. Ablations that prove structure adds value

Score every ablation on **validation**, then on test for the final set. All are SRGA unless noted.

| ID | Variant | What it tests | Expected if structure is real |
|---|---|---|---|
| A0 | SapBERT | bar | — |
| A1 | training-free `x̂ ⊕ √β·nb1` (β on validation) | free structural baseline (0.469 test now) | SRGA must beat it |
| A2 | score fusion with the current GCN-randneg (λ on validation) | cheap trained baseline | SRGA should beat it |
| A3 | SRGA, adapter **without edges** (MLP on `x̂` only) | supervision without structure | A4 > A3 |
| **A4** | **SRGA, full** | proposed | best |
| A5 | A4 on a **degree-preserving rewired** graph | is it the graph or the capacity? | A5 ≈ A3 < A4 |
| A6 | parent-only / child-only / undirected | value of directionality | directional ≥ undirected |
| A7 | ± DeepWalk input | is DeepWalk useful at all | likely small or none |
| A8 | + confidence gate `c_n` | does per-node adaptivity pay off | keep only if > A4 |
| A9 | λ curve 0 → 1 | is the floor respected, and where is the optimum | the curve peaks above λ=0 |
| A10 | Variant B | additive vs concatenated residual | — |

**Stratified reporting.** Split test pairs by whether SapBERT gets them right, and by how far the top SapBERT
confuser is from the gold target in DOID (≤ 2 hops vs farther). Real structural value should show up as fixes in the
"SapBERT wrong" group without losses in the "SapBERT right" group. Report **wins and losses** against SapBERT, not
only the means.

**Success criterion.** Validation and test MRR above A0 **and** A1 in both modes, over 3 seeds, with A5 clearly below
A4.

---

## 8. Second stage (optional): anchor-based re-ranking

Structure is only comparable *across* disconnected graphs through known alignments. A pairwise re-ranker on SRGA's
top-K (for example 20) can use **anchor-consistency features** that no node embedding can express:
- the number of neighbours of `n` whose gold-train partner is a neighbour of `d`;
- parent-of-match agreement;
- the SapBERT and SRGA scores.

Coverage is promising (63.7% of test pairs have such an anchor within 2 hops on both sides), but its discriminative
power is untested. This is a separate pipeline step (a new evaluator or method kind). Keep it out of the first
iteration.

---

## 9. Minimal integration into the dataset-agnostic pipeline

Everything stays default-off, so existing runs and configs are unchanged.

| Component | Change | Scope |
|---|---|---|
| `gcl_train/models.py` | New `ResidualAdapterEncoder`, selected by `gcl_train.model.encoder: residual_adapter`, with params `adapter_dim`, `lambda_init` (or fixed `lambda` per role), `use_struct_input`, `directional`. `forward` returns `z`; SapBERT input is used frozen (no gradient). | new class |
| `gcl_train/pipeline.py` (load) | Keep the edge direction: build `edge_type` (0 = src→dst as stored, 1 = reversed) next to `edge_index` instead of only symmetrizing. Pass `graph_role` per row for per-graph λ. Generic: direction comes from `edges.parquet`, with no dataset knowledge. | small |
| `gcl_train/losses.py` + `trainer.py` | Register `infonce_full`: anchors = positive rows' `anchor_id`; candidates = all non-held-out nodes of the positive's graph role (from `universe.parquet`); gold mask from the ingest `alignments.parquet` (already resolvable via the training_data manifest). When the loss is `infonce_full`, the trainer skips the graph-view loss (or `gcl_train.loss.graph_view: off`). | moderate |
| Validation split | Dataset or ingest option `alignments.val_fraction` (seeded), writing `split = val` for a slice of train pairs. `training_data` adds `val` to `_HELD_OUT_ALIGNMENT_SPLITS`, and the evaluator runs with `split: val`. | small |
| Export / evaluate | Unchanged: `z` is exported as `node_embeddings` (768 + d_h), and `gold_alignment` scores it by cosine. | none |
| A1 baseline | The same encoder with `trainable: false` and `adapter: neighbor_mean` (or `epochs: 0`), so the training-free baseline runs through the same pipeline and lineage. | small |
| Configs | `configs/experiments/ncit_doid_srga_ablation.yaml` (methods A0–A9) and one `gcl_train` overlay per variant. | configs only |

Phase 1 can skip `infonce_full` by using the existing `infonce` loss and sampler negatives. That needs only the
model, the edge-direction changes, the validation split and the configs.

---

## 10. Plan and risks

**Order.** First the validation split. Then:
1. A1 and A2 (no new model; this gives the bar to beat on validation).
2. SRGA (A4), first with `infonce` plus sampler negatives.
3. Switch to `infonce_full`.
4. Ablations A3, A5–A9 with 3 seeds.
5. Optionally the gate (A8), Variant B and the anchor re-ranker.

**Risks:**
- **Small gains on top of SapBERT** (probe gains were +0.02 to +0.04 MRR). Mitigate with 3 seeds and stratified
  win/loss reporting so a real gain is distinguishable from noise.
- **Over-fitting λ or τ** to 200-odd validation pairs. Use a coarse grid and prefer per-graph over per-node
  parameters.
- **Transductive training:** test nodes take part in message passing (features only, never in examples). This is
  unchanged from Sprint 4, but note it in any write-up.
- **A weaker SapBERT input than intended:** the store uses mean-pooled `label + synonyms + definition`. A
  `[CLS]`/label-based SapBERT may raise the bar (see the hard-negative analysis, P2). Re-check A0 if the text input
  changes, and use the strongest SapBERT variant as both baseline and SRGA input.
