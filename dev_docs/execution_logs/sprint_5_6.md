# Sprint 5, Phase 5.6: The paper's GAT text branch as the SRGA adapter (A14), with no-floor (A15) and rewired (A16) controls

Plan: [sprint_5.md](../sprints/sprint_5.md). Previous phases: [5.4](sprint_5_4.md), [5.5](sprint_5_5.md).
Lineage `ncit_doid_srga`; val = 211 pairs, test = 3,280.

## 0. Outcome in brief

- **GAT works as well as GCN, and for a clear reason: its attention stays almost uniform.**
  - A14 (GAT beside frozen SapBERT) is level with A11 (GCN) on val (0.4906 ± 0.0102 vs 0.4924 ± 0.0031) and on test
    (0.5279 ± 0.0049 vs 0.5234 ± 0.0030 global MRR; 0.5906 ± 0.0051 vs 0.5852 ± 0.0025 candidates).
  - The attention readout shows why: every head gives each neighbour within a few percent of `1/(degree + 1)`, so the GAT
    behaves like the GCN's average. It did not learn to treat parents, children, or SapBERT's confusers differently.
- **Gate 5.6 passes.** A14 is within one A11 seed std of A11 on val, and the rewired control A16 (0.4763) is below A14. GAT
  is reported as an equivalent paper-compatible variant, not a better one.
- **A15 (GAT, no SapBERT floor) repeats the A12 pattern slightly weaker.** Test global MRR is 0.5289 ± 0.0044 (A12 0.5361
  ± 0.0073) and candidates MRR 0.6065 ± 0.0034 (A12 0.6209 ± 0.0088). It breaks 406 of the 1,123 SapBERT-correct queries.
- **One unplanned test read, disclosed.** The pool diagnostic was run with A16 on the test split, so A16's test MRR was seen
  (0.4901 ± 0.0060 global). It is not in the test experiment and no decision depends on it (section 6). The same slip is
  corrected in the Phase 5.5 log for A13.
- No earlier selection is reopened. A3 remains the Sprint 5 selection (Phase 5.4).

## 1. Model architecture and equations

`ResidualAdapterEncoder(adapter="gat")` in [models.py](../../src/cga/pipelines/gcl_train/models.py). It is the 5.5 `gcn`
adapter with the graph tower swapped for the paper's existing `GAT` class, unchanged.

```
x̂_n = SapBERT_n / ||SapBERT_n||                     frozen, 768-d
t_n = fc1(x̂_n)                                       Linear(768 → 256)
g   = GAT(x̂, edge_index)                             GATConv(768 → 32, heads=8) concatenated to 256 → ReLU
                                                      → GATConv(256 → 256, heads=1), whole graph at once
h_n = fc3( Dropout_0.2( [ t_n ; g_n ] ) )            Linear(512 → 256), weights N(0, 1e-3²), bias 0

attention, layer 1, head k:  e_ij = LeakyReLU( a_kᵀ [W_k x̂_i ‖ W_k x̂_j] ),  α_ij = softmax_{j ∈ N(i) ∪ {i}} e_ij
A14 (floor):     z_n = [ x̂_n ; √λ · h_n / ||h_n|| ]      1024-d, λ = 0.25
A15 (no floor):  z_n = h_n                                 256-d
```

- Trainable parameters: **591,872**, against 590,848 for the GCN adapter (+0.17%). The per-head width `256 / 8 = 32` is what keeps
  the two towers the same size, so A14 vs A11 compares learned attention with the GCN's fixed average, not model size.
  `adapter_dim` not divisible by `gat_heads[0]` raises a `ValueError`.
- Same loss and training as 5.2–5.5 (the only change from A11/A12 is the tower):
  `L = L_fwd + 0.5 · L_rev`, full-softmax InfoNCE over every non-held-out candidate with the gold-target mask (any split), τ 0.05,
  Adam lr 1e-3, weight decay 1e-4, λ 0.25, seeds 42/43/44.
- Epochs: A14 / A16 = 50 (same as A11 and A13); A15 = **100**, chosen on val (section 3.1).
- A16 is A14 on a rewired message-passing graph (Phase 5.3 `edge_rewire`, seed = training seed).
- The GAT tower uses the symmetrized edge list with duplicates, as the paper's `gat` encoder does. A pair stored under two
  `edge_sources` gets two attention slots.

## 2. What was implemented

- [models.py](../../src/cga/pipelines/gcl_train/models.py): `adapter: gat` and the `gat_heads` constructor argument;
  [pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py) forwards `gcl_train.model.gat_heads`. `residual_floor` and
  `edge_rewire` are reused unchanged. The `gcn` adapter keeps its `self.gcn` attribute.
- Tests (454 pass; 8 new):
  - `gat` output has 768 + 256 columns, and its SapBERT block is the unit text vector;
  - its `h` changes with the edges (with the last layer enlarged, since the 1e-3 init hides the difference);
  - its parameter count is within 1% of the GCN adapter's;
  - `lambda = 0` gives zero gradient on the adapter;
  - `residual_floor: false` exports `h` only;
  - `adapter_dim` not divisible by the head count raises;
  - pipeline runs complete with `infonce_full`, with and without the floor, and with `edge_rewire`;
  - the 5.5 `gcn` adapter tests and the `gcn`/`gat` `DualEncoder` tests pass unchanged.
- [scripts/srga_gat_attention.py](../../scripts/srga_gat_attention.py) (new, analysis only, no pipeline change): rebuilds each seed's
  A14 model from its final checkpoint and reads first-layer attention (`return_attention_weights=True`), split by self, parents
  and children, and for queries whose SapBERT top confuser is a 1-hop neighbour of the gold target.
- Configs: [ncit_doid_srga_gat_sel.yaml](../../configs/experiments/ncit_doid_srga_gat_sel.yaml) (A15 epochs),
  [ncit_doid_srga_gat.yaml](../../configs/experiments/ncit_doid_srga_gat.yaml) (val table: A0, A3, A11, A12, A14, A15, A16),
  and A14/A15 added to [ncit_doid_srga_compare.yaml](../../configs/experiments/ncit_doid_srga_compare.yaml) for the test readout.
- README: the SRGA section lists the `gat` adapter and the attention script.

## 3. Results

### 3.1 A15 epoch selection (val, seed 42), evaluate run `20261015T233415Z-a36c094d`

| epochs | val MRR | val Hits@1 | val Hits@10 | train MRR |
|---|---|---|---|---|
| A12 (GCN, 200 epochs, reference) | 0.5394 | 0.4313 | 0.7583 | 0.9894 |
| 50 | 0.4771 | 0.3602 | 0.7204 | 0.8807 |
| **100** | **0.5214** | 0.4028 | 0.7867 | 0.9786 |
| 200 | 0.4477 | 0.3365 | 0.7062 | 0.9878 |

Chosen: **100 epochs**, in the middle of the grid (unlike A12, whose choice was at the edge). 200 epochs over-fits for the GAT
(val MRR drops 0.074 while train MRR rises 0.009). The seed-42 spread across epochs (0.077) is larger than the A14 vs A11 gap,
so this selection is noisy.

### 3.2 Val table (211 pairs, global; mean ± sample std over 3 seeds), evaluate run `20261015T233540Z-65c9ef6c`

| variant | what it is | val MRR | Hits@1 | Hits@5 | Hits@10 | train MRR |
|---|---|---|---|---|---|---|
| A0 | SapBERT | 0.4125 | 0.2938 | 0.5498 | 0.6398 | 0.4395 |
| A3 | MLP, no edges | 0.4784 ± 0.0042 | 0.3491 ± 0.0055 | 0.6430 ± 0.0179 | 0.7330 ± 0.0072 | 0.657 ± 0.028 |
| A11 | GCN + floor | 0.4924 ± 0.0031 | 0.3633 ± 0.0099 | 0.6509 ± 0.0099 | 0.7472 ± 0.0137 | 0.600 ± 0.005 |
| A12 | GCN, no floor | 0.5169 ± 0.0209 | 0.3949 ± 0.0356 | 0.6651 ± 0.0192 | 0.7520 ± 0.0072 | 0.990 ± 0.001 |
| **A14** | **GAT + floor** | **0.4906 ± 0.0102** | 0.3555 ± 0.0142 | 0.6619 ± 0.0137 | 0.7441 ± 0.0082 | 0.609 ± 0.011 |
| A15 | GAT, no floor | 0.5022 ± 0.0166 | 0.3807 ± 0.0192 | 0.6524 ± 0.0166 | 0.7551 ± 0.0274 | 0.976 ± 0.002 |
| A16 | A14 on a rewired graph | 0.4763 ± 0.0188 | 0.3491 ± 0.0301 | 0.6414 ± 0.0145 | 0.7251 ± 0.0142 | 0.646 ± 0.022 |

Per-seed val MRR (42 / 43 / 44): A14 0.4803 / 0.5008 / 0.4907; A15 0.5214 / 0.4924 / 0.4928; A16 0.4658 / 0.4651 / 0.4981.

Paired wins / losses against A0 on val (mean ± std over seeds):

| variant | wins | losses | ties (of 211) |
|---|---|---|---|
| A11 | 108.3 ± 3.2 | 24.3 ± 0.6 | 78 |
| A12 | 100.7 ± 3.2 | 60.7 ± 1.5 | 50 |
| **A14** | 109.3 ± 4.2 | 25.3 ± 3.5 | 76 |
| A15 | 106.3 ± 6.7 | 60.0 ± 5.2 | 45 |
| A16 | 109.7 ± 3.1 | 43.0 ± 3.6 | 59 |

Paired against each other, same seed (wins / losses / ties of the first variant, seeds 42 / 43 / 44):
- A11 vs A14: 57/61/93, 52/61/98, 55/62/94 (mostly ties; A14 wins slightly more often).
- A12 vs A15: 64/76/71, 70/72/69, 67/76/68.
- A14 vs A16: 78/56/77, 91/47/73, 74/59/78 (A14 ahead on every seed).
- A14 vs A15: 67/84/60, 77/74/60, 78/74/59.

Training loss (epoch 1 → last), all 12 runs, no NaN: A14 6.1–6.2 → 3.49–3.65 (50 epochs); A16 6.3 → 3.54–3.70 (50 epochs);
A15 9.5–9.9 → 0.41 (100 epochs).

### 3.3 Test readout (3,280 pairs, once), evaluate run `20261015T233935Z-038f363f`

The test numbers of A0, A1, A3, A4, A11 and A12 equal Phases 5.4 and 5.5 exactly (same models). A16 is not in this experiment.

**global**

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| A3 MLP | 0.3784 ± 0.0035 | 0.6359 ± 0.0020 | 0.7295 ± 0.0069 | 0.4957 ± 0.0024 |
| A11 GCN + floor | 0.3982 ± 0.0054 | 0.6732 ± 0.0027 | 0.7708 ± 0.0030 | 0.5234 ± 0.0030 |
| A12 GCN, no floor | 0.4079 ± 0.0102 | 0.6952 ± 0.0044 | 0.7725 ± 0.0040 | 0.5361 ± 0.0073 |
| **A14 GAT + floor** | 0.4028 ± 0.0043 | 0.6777 ± 0.0055 | 0.7774 ± 0.0037 | **0.5279 ± 0.0049** |
| A15 GAT, no floor | 0.3925 ± 0.0043 | 0.6952 ± 0.0014 | 0.7849 ± 0.0028 | 0.5289 ± 0.0044 |

**candidates** (101 per pair)

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| A3 MLP | 0.4325 ± 0.0034 | 0.7355 ± 0.0034 | 0.8450 ± 0.0015 | 0.5687 ± 0.0022 |
| A11 GCN + floor | 0.4440 ± 0.0060 | 0.7613 ± 0.0029 | 0.8710 ± 0.0025 | 0.5852 ± 0.0025 |
| A12 GCN, no floor | 0.4742 ± 0.0134 | 0.8103 ± 0.0023 | 0.9034 ± 0.0045 | 0.6209 ± 0.0088 |
| **A14 GAT + floor** | 0.4510 ± 0.0062 | 0.7627 ± 0.0036 | 0.8732 ± 0.0024 | **0.5906 ± 0.0051** |
| A15 GAT, no floor | 0.4532 ± 0.0019 | 0.8014 ± 0.0038 | 0.9047 ± 0.0018 | 0.6065 ± 0.0034 |

- A14 is +0.0045 (global) and +0.0054 (candidates) over A11, about 1 std. This is not a reliable difference.
- A15 is below A12 in MRR (−0.007 global, −0.014 candidates), with a lower Hits@1 and slightly higher Hits@10 in global mode.

### 3.4 Stratified fixed/broken on test (global, vs A0; mean over 3 seeds, rank-1 definition)

Strata sizes: SapBERT right, near 442 / far 681; SapBERT wrong, near 901 / far 1,256 (confuser within 2 hops of the target in DOID).

| variant | fixed | broken | fixed, "wrong, near" | fixed, "wrong, far" | broken, "right, near" | broken, "right, far" |
|---|---|---|---|---|---|---|
| A11 (5.5) | 281.0 | 98.0 | 79.7 | 201.3 | 34.0 | 64.0 |
| **A14** | 298.0 | 99.7 | 76.7 | 221.3 | 39.3 | 60.3 |
| A12 (5.5) | 626.7 | 411.7 | 278.7 | 348.0 | 149.3 | 262.3 |
| A15 | 570.0 | 405.7 | 222.3 | 347.7 | 145.7 | 260.0 |

Net rank-1 gain (fixed − broken): A11 +183, A14 +198, A12 +215, A15 +164. A14 improves the rank of 1,676 queries and worsens 333 (5.0 : 1),
against A11's 5.2 : 1; A15 improves 1,557 and worsens 883 (1.8 : 1), against A12's 1.6 : 1. As with the GCN, the floor roughly
quarters the breaks (100 vs 406 for GAT), and the no-floor variants fix far more "wrong, near" queries (222 vs 77).

## 4. Attention readout (A14, descriptive)

[scripts/srga_gat_attention.py](../../scripts/srga_gat_attention.py), first `GATConv` layer, averaged over the 8 heads, 3 seeds
(spread across seeds below 0.001 in every cell, so one value is shown).

Attention mass of a node, mean over nodes with at least one neighbour (self + parents + children = 1):

| graph | self | self if uniform | parents | parents if uniform | children | children if uniform |
|---|---|---|---|---|---|---|
| NCIT | 0.346 | 0.345 | 0.466 | 0.466 | 0.495 | 0.494 |
| DOID | 0.418 | 0.417 | 0.446 | 0.447 | 0.553 | 0.553 |

(The parent and children columns are the mass over the nodes that have at least one parent / child, so they do not sum to 1 within a row.)

Test queries whose SapBERT top confuser is a 1-hop neighbour of the gold target in DOID (594 of 3,280):

| confuser relation | n queries | attention to the confuser | mean attention to the target's other neighbours | uniform 1/(deg+1) |
|---|---|---|---|---|
| parent | 242 | 0.393 | 0.251 | 0.391 |
| child | 352 | 0.182 | 0.181 | 0.182 |

Per head, a separate check (not in the script) gave attention within a few percent of uniform: `α_ij · (deg_i + 1)` has std 0.015–0.032
per head, and the largest deviation of any edge in any head is 0.22–0.27 (seeds 42/43/44). The head average is therefore not hiding
heads that disagree.

Reading:
- **Attention is essentially uniform.** Self, parents and children each get what a plain average would give them, and the
  confuser gets the same weight as an average neighbour (in the parent row, 0.393 vs 0.251 for the others reflects that a node
  with a few parents gives each of them a large uniform share, not a learned preference: it matches the uniform 0.391).
- So the GAT's learned attention did not learn to separate parents from children or to down-weight SapBERT's confusers.
  The GAT here is a mean aggregator with extra parameters, which is why A14 ≈ A11.
- Likely cause, not tested: SapBERT features are unit-norm (coordinates about 0.04), so the attention logits start near zero, and
  50 epochs at lr 1e-3 with a loss dominated by the text pathway never moves them far. Larger attention learning rates, an
  attention-specific temperature or direction-aware features could change this. That was out of scope.
- This describes what the model does; it is not an explanation of why its ranking is right.

## 5. Gate 5.6 and what the controls say

| condition | result |
|---|---|
| A14 ≥ A11 − 1 seed std on val | 0.4906 vs 0.4924 − 0.0031 = 0.4893: **yes** (by 0.0013; the gap is far inside A14's own std of 0.0102) |
| A16 (rewired) < A14 | 0.4763 vs 0.4906: **yes** (A14 ahead on all 3 seeds, 78/56, 91/47, 74/59 queries) |

**Gate 5.6: GO.** GAT is an equivalent paper-compatible variant. Nothing in the data shows it is better than the GCN.

- **A14 vs A11 (attention vs averaging at matched size).** Equal. Val −0.002, test +0.005 global and +0.005 candidates, all within noise, and the
  attention readout explains why: the attention stayed uniform.
- **A15 vs A12 (same, without the floor).** A15 is slightly lower and the epoch choice more fragile (200 epochs fails). The GAT does not
  rescue the no-floor variant's tendency to break SapBERT-correct queries.
- **A14 vs A15 (what the floor is worth for GAT).** Same trade-off as the GCN: A15 reaches a higher candidates MRR (0.6065 vs 0.5906) and a
  similar global MRR (0.5289 vs 0.5279), but breaks 406 SapBERT-correct queries against A14's 100.
- **A16 vs A14 (graph or capacity).** The graph, but the effect is small: −0.014 val MRR, A14 wins the pair on every seed. This is a smaller drop than the A13 vs A11 gap (−0.019) and the A5 vs A4 gap (−0.108) in earlier phases.

## 6. Test discipline note

The pool diagnostic ([scripts/srga_pool_diagnostic.py](../../scripts/srga_pool_diagnostic.py)) was run with A16 on the test split, so A16's
test MRR was printed: 0.4901 ± 0.0060 (all pool), 0.5639 ± 0.0059 held-out-only, 0.6051 ± 0.0068 trainable-only. The plan reads only A14 and A15
on test, and A16 is not in `ncit_doid_srga_compare.yaml`. The number is not used for any decision here, and it points the same way as the val
result (A16 below A14). It is listed so the record is complete. The cause was passing every group to the diagnostic on both splits; the same
slip with A13 in Phase 5.5 is now corrected in that log.

## 7. Pool diagnostic, held-out-exclusion shortcut (A14, A15)

As in 5.5 §4, MRR with the candidate pool cut (the gold target is always kept); `held-out only` has every candidate equally unseen by the loss.

| model | val: all | val: held-out only | val: trainable only | test: all | test: held-out only | test: trainable only |
|---|---|---|---|---|---|---|
| A11 | 0.4924 ± 0.0031 | 0.5433 ± 0.0062 | 0.6382 ± 0.0129 | 0.5234 ± 0.0030 | 0.5945 ± 0.0025 | 0.6462 ± 0.0009 |
| A12 | 0.5169 ± 0.0209 | 0.5931 ± 0.0100 | 0.6129 ± 0.0156 | 0.5361 ± 0.0073 | 0.6199 ± 0.0041 | 0.6320 ± 0.0077 |
| A14 | 0.4906 ± 0.0102 | 0.5516 ± 0.0077 | 0.6314 ± 0.0024 | 0.5279 ± 0.0049 | 0.5993 ± 0.0044 | 0.6495 ± 0.0060 |
| A15 | 0.5022 ± 0.0166 | 0.5805 ± 0.0175 | 0.6084 ± 0.0141 | 0.5289 ± 0.0044 | 0.6167 ± 0.0046 | 0.6347 ± 0.0029 |

A14 and A15 behave like A11 and A12: the no-floor variant's lead is largest on the held-out-only pool (A15 − A14: +0.029 val, +0.017 test) and absent on the trainable-only pool
(A15 below A14), so the no-floor advantage is not the held-out-exclusion shortcut.

## 8. Limitations

- **Val is small** (211 pairs): the A14 vs A11 difference and the A15 epoch choice are inside noise. The test readout (3,280 pairs) is the more reliable comparison, and it too shows no reliable difference.
- **A15's epoch selection is fragile**: seed 42 alone chose it, and its best epoch count (100) sits between a weak 50 and a weak 200.
- **Attention was not tuned.** The finding is that attention stayed near uniform under the Phase 5.2 recipe, not that GAT cannot learn useful attention.
- GAT kernels can be non-deterministic and runs are not bit-reproducible here (Phase 5.2); the seed spread of A14 (std 0.010 val) is larger than A11's (0.003).
- The first-layer readout does not cover the second `GATConv`, which has one head.
- Training is transductive. `ruff` is not installed, so lint was not run.

## 9. Files

- Modified: `src/cga/pipelines/gcl_train/{models,pipeline}.py`, `tests/unit/pipelines/gcl_train/{test_models,test_pipeline}.py`,
  `configs/experiments/ncit_doid_srga_compare.yaml`, `README.md`, `dev_docs/sprints/sprint_5.md` (Phase 5.6 boxes ticked),
  `dev_docs/execution_logs/sprint_5_5.md` (A13 test-read correction).
- Created: `scripts/srga_gat_attention.py`, `configs/experiments/ncit_doid_srga_gat_sel.yaml`,
  `configs/experiments/ncit_doid_srga_gat.yaml`, this log.
- Untracked artifacts (all dated 20261015; seeds 42 / 43 / 44):
  - A14 `srga-gat-l0.25-s<seed>`: `T233132Z-40538b23` / `T233207Z-09135889` / `T233242Z-958865b2`;
  - A16 `srga-gat-rewired-l0.25-s<seed>`: `T233149Z-6431c092` / `T233224Z-a6aa038f` / `T233258Z-7b033d94`;
  - A15 `gat-full-e100-s<seed>`: `T233329Z-ded662e0` / `T233507Z-0bf067ed` / `T233523Z-c583bf2f`;
  - selection runs `gat-full-e50-s42` `T233315Z-103a895c`, `gat-full-e200-s42` `T233346Z-4cee19ab`;
  - evaluate runs `20261015T233415Z-a36c094d` (A15 epochs), `20261015T233540Z-65c9ef6c` (val table), `20261015T233935Z-038f363f` (test readout).

## 10. Commands

```
uv run pytest tests/unit -q

# A14 (seed 43 shown). A16: add the two edge_rewire --set lines. A15: add residual_floor=false and use epochs 100.
uv run cga run gcl_train --dataset ncit_doid_srga --set seed=43 --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=gat --set gcl_train.model.lambda=0.25 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.model.model_id=srga-gat-l0.25-s43
#   A16 extra: --set gcl_train.model.edge_rewire.enabled=true --set gcl_train.model.edge_rewire.seed=43 --set gcl_train.model.model_id=srga-gat-rewired-l0.25-s43
#   A15 extra: --set gcl_train.model.residual_floor=false --set gcl_train.train.epochs=100 --set gcl_train.model.model_id=gat-full-e100-s43

uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_gat_sel
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_gat
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_compare

uv run python scripts/srga_paired_ranks.py --dataset ncit_doid_srga --split val --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}" --group A14="srga-gat-l0.25-s{seed}" --group A15="gat-full-e100-s{seed}" --group A16="srga-gat-rewired-l0.25-s{seed}"
uv run python scripts/srga_stratified.py --dataset ncit_doid_srga --split test --group A11="srga-gcn-l0.25-s{seed}" --group A14="srga-gat-l0.25-s{seed}" --group A15="gat-full-e100-s{seed}"
uv run python scripts/srga_pool_diagnostic.py --dataset ncit_doid_srga --split test --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}" --group A14="srga-gat-l0.25-s{seed}" --group A15="gat-full-e100-s{seed}"
uv run python scripts/srga_gat_attention.py --dataset ncit_doid_srga --split test --model-id "srga-gat-l0.25-s{seed}"
```

Pass only the groups the plan allows on `--split test` (A16 is not one of them).
