# Sprint 5, Phase 5.3: Directional graph adapter, the full SRGA (A4), and its structural controls (A5, A6)

Plan: [sprint_5.md](../sprints/sprint_5.md). Previous phases: [sprint_5_1.md](sprint_5_1.md), [sprint_5_2.md](sprint_5_2.md).
Lineage `ncit_doid_srga`. No test split is read in this phase (the single test readout is Phase 5.4).

**Outcome in one line:** gate 5.3 **fails**. The full SRGA (A4) does not beat the no-edge supervised adapter (A3):
val MRR 0.4791 +- 0.0183 vs 0.4784 +- 0.0042. A rewired graph (A5) is far worse, so the real graph is not noise, but it adds
nothing on top of supervision. Per the plan, stop adding architecture and carry A3 (the best of A1/A3) into Phase 5.4.

## 1. Model architecture

`ResidualAdapterEncoder(adapter="relational")` in [models.py](../../src/cga/pipelines/gcl_train/models.py). Same export as
5.1/5.2: `z = [x̂ ; sqrt(lambda) * normalize(h)]`, SapBERT block frozen, lambda = 0.25 fixed.

```
x̂ = SapBERT / ||SapBERT||  (frozen, d = 768)
parents(n), children(n): 1-hop neighbours by edge type  (edges are stored child -> parent)

a_n = [ x̂_n ; p_n ; c_n ; x̂_n ⊙ p_n ; log(1+n_children) ; log(1+n_parents) ]        4d + 2 = 3074
        p_n = mean_{parents(n)} x̂        c_n = mean_{children(n)} x̂       (0 if none)

u_n = W_in a_n + b_in                                                           256
r_n = W_par · mean_{parents(n)} u   +   W_chi · mean_{children(n)} u            256   (one relational layer)
h_n = MLP_2( LayerNorm(u_n + r_n) )        MLP_2 = Linear -> ReLU -> Dropout(0.2) -> Linear   (last layer std 1e-3)

z_n = [ x̂_n ; sqrt(lambda) · h_n / ||h_n|| ]                                   768 + 256 = 1024
```

Score between a source concept `a` and a target concept `d`: `z_a · z_d = x̂_a · x̂_d + lambda * ĥ_a · ĥ_d` (cosine of `z`
ranks the same up to per-node norms), as in the earlier phases.

- Trainable parameters: **1,050,368** (A4, directional) and **787,968** (A6, undirected). The A3 MLP had 263,168.
- **Deviation from the plan's formula, and why.** The plan writes `h = MLP_2(LayerNorm(a_n + RGCN_1(a)))`. A residual
  `a_n + RGCN(a)` forces the relational layer to map 3,074 -> 3,074, about 9M extra weights per relation, against the design's
  "about 1M parameters". So `a_n` is first projected to 256 (`W_in`) and the relational layer and residual run at width
  256. The separate parent/child weights, the one relational layer, the residual and the LayerNorm + MLP_2 are as specified.
- **A6 (`directional: false`)** collapses to one relation: `a_n = [x̂_n ; m_n ; x̂_n ⊙ m_n ; log(1+deg_n)]` (`3d + 1`),
  `m_n` = mean over all neighbours, one weight `W_nbr`: `r_n = W_nbr · mean_{nbrs(n)} u`.
- Direction convention is unchanged from 5.1: an edge delivers the source's vector to the target. A type-0 edge
  (stored child -> parent) feeds the target's *children* mean, a type-1 edge (the reversed copy) its *parents* mean.
  `n_children` counts type-0 neighbours (in-degree in the stored graph) and `n_parents` type-1 (out-degree).
- Duplicate `(src, dst)` edges are collapsed before every mean and count.

### Loss and training (same as Phase 5.2)

```
L = L_fwd + 0.5 * L_rev        full-softmax InfoNCE over all non-held-out DOID (fwd) / NCIT (rev) nodes,
                               tau = 0.05, gold alternatives (any split) masked, 1,174 train pairs, no sampled negatives
```

Adam, lr 1e-3, weight decay 1e-4, 50 epochs, full batch, dropout 0.2, seeds 42/43/44, lambda 0.25: the settings chosen in
Phase 5.2, unchanged, so A3 / A4 / A5 / A6 differ only in the adapter and the graph.

### A5: degree-preserving rewiring

`gcl_train.model.edge_rewire: {enabled, seed}` ([rewire.py](../../src/cga/pipelines/gcl_train/rewire.py)).
- Takes the de-duplicated stored (child -> parent) edges of each graph role and repeats `10 * n_edges` double-edge swaps:
  `(a->b), (c->d)  =>  (a->d), (c->b)`.
- A swap is skipped if it makes a self-loop or a duplicate edge. So every node keeps its **in- and out-degree**, the
  graph stays simple, and edges stay inside their own graph (no cross-graph edge; an edge joining two roles raises).
- Applied in the `load` step to the message-passing graph only. The universe, examples, gold mask and embeddings
  inputs are untouched. The rewiring seed equals the training seed of that run.
- Off by default; the `gcn`, `gat` and earlier SRGA paths build the same edges as before (tests pass).

## 2. What was implemented

- [models.py](../../src/cga/pipelines/gcl_train/models.py): `relational` adapter (directional and undirected), shared helper
  `_mean_and_count_over_incoming`; `neighbor_mean` and `mlp` are unchanged.
- [rewire.py](../../src/cga/pipelines/gcl_train/rewire.py) (new) and the `load` hook in
  [pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py).
- [scripts/srga_paired_ranks.py](../../scripts/srga_paired_ranks.py) (new): per-query ranks and paired wins/losses.
  It calls the evaluator's own `_build_queries` and `mrr_via_argsort`, and its MRRs match `metrics.parquet`
  (A0 0.4125, A3 seed 42 0.4756, A4 seed 42 0.4728). It reads artifacts and writes nothing.
- Configs: [ncit_doid_srga_rel.yaml](../../configs/experiments/ncit_doid_srga_rel.yaml) (A0, A1, A3, A4, A5, A6; val + train).
- Tests (439 pass; 9 new):
  - Parent-only and child-only sides use separate weights: with `LayerNorm`/`MLP_2` removed, the leaf node (parent only)
    changes only with `W_par`, the root (child only) only with `W_chi`, and the two contributions add.
  - The export keeps the `[x̂ ; sqrt(lambda) ĥ]` layout and norms.
  - A6 has one relation (`W_nbr`, no `W_par`/`W_chi`), and it does not need `edge_type`.
  - A directional adapter without `edge_type` raises; `lambda = 0` gives zero gradient.
  - Rewiring: degrees, roles, simplicity, no cross-graph edge, seed determinism, duplicate collapse, error on a cross-graph edge.
  - A pipeline run calls the rewiring once, with the configured seed, only for the rewired run.

## 3. Results

Val (211 pairs, global); mean +- sample std over 3 seeds. Evaluate run `20261015T184250Z-fcfb39bc`.

| variant | what it tests | val MRR | Hits@1 | Hits@5 | Hits@10 | train MRR |
|---|---|---|---|---|---|---|
| A0 SapBERT | bar | 0.4125 | 0.2938 | 0.5498 | 0.6398 | 0.4395 |
| A1 nb, l=0.25 dir | free structure | 0.4555 | 0.3365 | 0.6019 | 0.6919 | 0.4911 |
| A3 MLP, no edges | supervision only | 0.4784 +- 0.0042 | 0.3491 +- 0.0055 | 0.6430 +- 0.0179 | 0.7330 +- 0.0072 | 0.657 +- 0.028 |
| **A4 relational (full)** | proposed | 0.4791 +- 0.0183 | 0.3428 +- 0.0166 | 0.6509 +- 0.0261 | 0.7457 +- 0.0278 | 0.718 +- 0.012 |
| A5 A4 on rewired graph | graph vs capacity | 0.3714 +- 0.0242 | 0.2670 +- 0.0323 | 0.5087 +- 0.0119 | 0.5845 +- 0.0072 | 0.715 +- 0.042 |
| A6 relational, undirected | directionality | 0.4747 +- 0.0190 | 0.3507 +- 0.0264 | 0.6382 +- 0.0239 | 0.7267 +- 0.0234 | 0.711 +- 0.005 |

Per-seed val MRR (42 / 43 / 44): A3 0.4756 / 0.4764 / 0.4832; A4 0.4728 / 0.4648 / 0.4998; A5 0.3670 / 0.3497 / 0.3976;
A6 0.4568 / 0.4727 / 0.4947.

Paired wins / losses against A0 on val (ranks of the gold target; ties are equal ranks). Per-seed rows:
[scripts/srga_paired_ranks.py](../../scripts/srga_paired_ranks.py).

| variant | wins vs A0 (mean +- std) | losses vs A0 | ties (of 211) |
|---|---|---|---|
| A1 | 96 | 33 | 82 |
| A3 | 102.3 +- 3.5 | 35.7 +- 6.4 | 73 |
| A4 | 110.0 +- 5.2 | 45.0 +- 6.6 | 56 |
| A5 | 86.3 +- 6.8 | 81.7 +- 13.1 | 43 |
| A6 | 111.7 +- 4.6 | 42.3 +- 9.0 | 57 |

Paired against each other, same seed (wins / losses / ties of the first variant, seeds 42 / 43 / 44):
A3 vs A4 57/89/65, 66/88/57, 64/85/62; A4 vs A5 114/48/49, 114/51/46, 106/45/60; A4 vs A6 75/60/76, 78/65/68, 79/62/70.

Model runs, `models/gcl/ncit_doid_srga/`, all dated 20261015 (seeds 42 / 43 / 44):
- A4 `srga-rel-l0.25-s<seed>`: `T183938Z-5a3dbf85` / `T184028Z-e2003369` / `T184124Z-27930482`
- A5 `srga-rel-rewired-l0.25-s<seed>`: `T183955Z-ccb6a098` / `T184045Z-7b105549` / `T184142Z-efb43eb7`
- A6 `srga-rel-undir-l0.25-s<seed>`: `T184012Z-18f5ba51` / `T184107Z-bef902cf` / `T184200Z-866c7e85`

Training loss (epoch 1 -> 50), all 9 runs: about 6.2-6.4 -> 3.0-3.2 (A4, A6), 3.4-4.0 (A5). No NaN.

## 4. Gate 5.3: STOP

The gate asks for A4 > max(A1, A3) by more than the seed std **and** A5 < A4.

| condition | result |
|---|---|
| A4 > A1 | 0.4791 vs 0.4555: yes (+0.024) |
| A4 > A3 by more than the seed std | 0.4791 vs 0.4784: **no** (+0.0007; the std is 0.004 for A3 and 0.018 for A4) |
| A5 clearly below A4 | 0.3714 vs 0.4791 (-0.108, about 6 std): yes |

The first condition fails, so the graph adapter is not justified on this data. What the controls show:

- **The real graph carries information a rewired graph does not.** A5 is far below A4 and below A3, with more losses than
  wins against A0 (81.7 vs 86.3). With wrong neighbours, the adapter uses them and gets worse than having no graph at all.
- **But it adds nothing over supervision.** A4 equals A3 in MRR, and A3 beats A4 head to head on all three seeds (more
  per-query wins than losses). A4's train MRR is higher (0.718 vs 0.657) with no val gain, so the extra capacity fits the
  train pairs without helping unseen ones.
- **Directionality is not resolved.** A4 vs A6 differ by +0.004 MRR with std 0.018-0.019 (A4 wins 75-79 queries, loses
  60-65, on each seed); within noise. The training-free result of Phase 5.1 (directional clearly better) is not
  reproduced for the trained adapter.
- A4 is noisier than A3 (std 0.018 vs 0.004), the typical cost of a larger model on 1,174 pairs.

**Decision:** stop adding architecture (per the plan). Phase 5.4 evaluates the best of {A1, A3, A4} on val: **A3**
(0.4784 +- 0.0042; A4's mean is 0.0007 higher but within noise, with 4x the parameters, 4x the variance and a loss in the
head-to-head counts). The runner-up for the test readout is A1 or A4; the plan lets the log say. Recommended: A1 (the
training-free, parameter-free runner-up, mean 0.4555) and A4 if a structural reference on test is wanted. The choice is the
developer's to confirm before 5.4.

## 5. Caveats

- 211 val pairs and 3 seeds: the A3/A4/A6 differences (<= 0.005 MRR) are well inside seed noise; only the A5 drop is clear.
  The conclusion is "no demonstrated gain from edges on top of supervision", not "edges cannot help".
- The A5 control rewires with the training seed, so each A5 seed has a different rewired graph (the std includes that).
- A5's degree features (`log(1+n_children)`, `log(1+n_parents)`) are identical to A4's by construction.
- `x̂ ⊙ p_n` has entries about 1e-3 (a product of two unit-vector coordinates), so that block contributes little at default
  initialisation; kept as specified.
- The adapter width/projection deviation (section 1) was made to keep the model near 1M parameters; an un-projected
  residual was not tried.
- Training is transductive: val/test nodes are message-passing inputs, never in a loss term.
- Seeds are not bit-reproducible on this machine (see 5.2). `ruff` is not installed, so lint was not run.

## 6. Files

Modified: `src/cga/pipelines/gcl_train/{models,pipeline}.py`, `tests/unit/pipelines/gcl_train/{test_models,test_pipeline}.py`,
`dev_docs/sprints/sprint_5.md` (Phase 5.3 boxes ticked).
Created: `src/cga/pipelines/gcl_train/rewire.py`, `tests/unit/pipelines/gcl_train/test_rewire.py`,
`scripts/srga_paired_ranks.py`, `configs/experiments/ncit_doid_srga_rel.yaml`, this log.
Untracked artifacts: 9 model lineages under `models/gcl/ncit_doid_srga/` and one evaluation run.

## 7. Commands

```
uv run pytest tests/unit -q

# A4 (seed 43 shown); A6: --set gcl_train.model.directional=false ; A5: add the two edge_rewire --set lines
uv run cga run gcl_train --dataset ncit_doid_srga --set seed=43 --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=relational --set gcl_train.model.directional=true --set gcl_train.model.lambda=0.25 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.model.model_id=srga-rel-l0.25-s43
#   A5 extra: --set gcl_train.model.edge_rewire.enabled=true --set gcl_train.model.edge_rewire.seed=43 --set gcl_train.model.model_id=srga-rel-rewired-l0.25-s43

uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_rel
uv run python scripts/srga_paired_ranks.py --dataset ncit_doid_srga --split val --group A1=srga-nb-l0.25-dir --group A3="srga-mlp-l0.25-s{seed}" --group A4="srga-rel-l0.25-s{seed}" --group A5="srga-rel-rewired-l0.25-s{seed}" --group A6="srga-rel-undir-l0.25-s{seed}"
```
