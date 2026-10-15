# Sprint 5, Phase 5.5: The paper's GCN text branch as the SRGA adapter (A11), with no-floor (A12) and rewired (A13) controls

Plan: [sprint_5.md](../sprints/sprint_5.md). Previous phases: [5.1](sprint_5_1.md), [5.2](sprint_5_2.md),
[5.3](sprint_5_3.md), [5.4](sprint_5_4.md). Lineage `ncit_doid_srga`; val = 211 pairs, test = 3,280.

## 0. Outcome in brief

- **The paper's GCN text branch works for the supervised task.** Beside frozen SapBERT (A11) it beats A3 and A4 and
  gate 5.5 passes:
  - Test MRR is 0.5234 ± 0.0030 (global) and 0.5852 ± 0.0025 (candidates), against A3's 0.4957 and 0.5687, A4's 0.5084 and
    0.5949, and SapBERT's 0.4475 and 0.5193.
  - It has the fewest breaks of any trained variant, 98 SapBERT-correct test queries lost against A3's 137 and A4's 259.
  - A rewired graph (A13) is clearly worse.
- **The same branch without the SapBERT floor (A12) is the strongest on MRR, but loses the safety net.** Test MRR is
  0.5361 ± 0.0073 (global) and 0.6209 ± 0.0088 (candidates), but it breaks 412 of the 1,123 queries SapBERT got right.
- **Sprint 4's GCN failure was mostly the loss, not the missing floor.** With the full-softmax loss and no floor, the
  same kind of GCN reaches 0.536 global test MRR against 0.223 for the Sprint 4 triplet-trained one (section 5).
- **A selection check was run and found no shortcut.** A12's train MRR is 0.99, so the candidate pool was split into
  held-out and trainable nodes to check for a memorisation shortcut. A12's lead grows on the pool where every
  candidate is equally unseen (section 4).
- **Test selection from Phase 5.4 is not reopened.** A3 remains the pre-declared Sprint 5 selection. A11 and A12 are
  additional findings, read once on test after their val table was recorded.

## 1. Model architecture and equations

`ResidualAdapterEncoder(adapter="gcn")` in [models.py](../../src/cga/pipelines/gcl_train/models.py). It is the text half of
the paper's `DualEncoder` (`fc1`, the `GCN` tower, `fc3`) with the DeepWalk branch removed. The `GCN` class is reused
unchanged: `GCNConv(768 → 256)`, ReLU, `GCNConv(256 → 256)`, undirected, symmetric normalisation with self-loops.

```
x̂_n = SapBERT_n / ||SapBERT_n||            frozen, 768-d
t_n = fc1(x̂_n)                              Linear(768 → 256)
g   = GCN(x̂, edge_index)                    GCNConv(768→256) → ReLU → GCNConv(256→256), whole graph at once
h_n = fc3( Dropout_0.2( [ t_n ; g_n ] ) )   Linear(512 → 256), weights N(0, 1e-3²), bias 0

A11 (floor):       z_n = [ x̂_n ; sqrt(λ) · h_n / ||h_n|| ]     768 + 256 = 1024-d,  λ = 0.25
A12 (no floor):    z_n = h_n                                    256-d, no SapBERT block
```

- Trainable parameters: **590,848** (`fc1` 196,864 + `GCNConv` 196,864 + 65,792 + `fc3` 131,328). A3 had 263,168 and A4 1,050,368.
- A11 scores by `z_a · z_d = x̂_a · x̂_d + λ · ĥ_a · ĥ_d`, as in A3/A4. `λ = 0` gives exactly SapBERT (zero gradient on the adapter, tested).
- **Difference from A3:** A3 computes `h` from the node's own text only. Here the GCN branch also mixes in the neighbours' text
  (undirected, 2 hops, no parent/child distinction), through the same message passing the paper uses.
- **Difference from A4:** A4 separates parents and children and adds degree features, while A11 uses the paper's undirected GCN.
- **A13** is A11 on a rewired message-passing graph (Phase 5.3 `edge_rewire`: degree-preserving double-edge swaps within
  each graph role, no cross-graph edge), with the rewiring seed equal to the training seed.

Loss and training for all three: the Phase 5.2 full-softmax InfoNCE over the 1,174 train pairs,

```
L = L_fwd + 0.5 · L_rev     softmax over every non-held-out DOID (fwd) / NCIT (rev) node, other gold targets of any split masked
τ = 0.05, Adam lr 1e-3, weight decay 1e-4, full batch, seeds 42/43/44
epochs: A11 / A13 = 50 (same as A3);  A12 = 200 (chosen on val, below)
```

## 1b. Model Description

A11 and A12 share the same learned part, built from the paper's GCN text branch. They differ only in whether frozen SapBERT is kept in the output.

**The shared learned part (`h`)**

For every concept:
1. **Text projection.** Take its unit-length SapBERT vector (768 numbers) and pass it through a linear layer down to 256 numbers. This is the paper's `fc1`.
2. **GCN branch.** Run the paper's 2-layer GCN over each ontology's whole graph. Each layer replaces a concept's representation with a normalised average of itself and its neighbours, then applies a learned transform: 768 → 256, then 256 → 256.
   - Parents and children are treated the same (undirected).
   - After two layers, each concept has absorbed information from neighbours up to 2 hops away.
3. **Fuse.** Join the two 256-number results, apply dropout (0.2), and mix them with one more linear layer down to 256 numbers. This is the paper's `fc3`. The result is `h`.

This is the paper's `DualEncoder` with the DeepWalk branch removed: about 591k trainable parameters. SapBERT itself is never changed.

**A11: GCN with the SapBERT floor**

- Output: `[SapBERT vector ; √0.25 × h scaled to unit length]` (1,024 numbers).
- Similarity is roughly SapBERT similarity plus 0.25 × the similarity of the learned parts.
- The learned part can only add a correction on top of SapBERT. At weight 0 it ranks exactly like SapBERT, so SapBERT is the floor.
- Trained for 50 epochs.

**A12: GCN without the floor**

- Output: just `h` (256 numbers). SapBERT is used only as the input features, not kept in the output.
- The model has to learn the whole matching space itself, so nothing anchors it to SapBERT's ranking.
- Trained for 200 epochs (chosen on val).

**Training for both.** Same as A3/A4: for each known NCIT–DOID training pair, make the NCIT concept closer to its DOID partner than to every other DOID concept the loss is allowed to see, and the reverse. Other valid partners are excluded. Temperature 0.05, Adam optimiser, 3 seeds.

**What the difference showed on test:**

| | A11 (floor) | A12 (no floor) |
|---|---|---|
| global MRR | 0.523 | 0.536 |
| candidates MRR | 0.585 | 0.621 |
| SapBERT-correct queries broken | 98 | 412 (about 37%) |
| train MRR | 0.60 | 0.99 |

A12 ranks better on average but breaks many queries SapBERT already got right, and it memorises the training pairs. A11 is the safer of the two: slightly lower average, but it rarely undoes a correct SapBERT answer.


## 2. What was implemented

- [models.py](../../src/cga/pipelines/gcl_train/models.py): `adapter: gcn` and `residual_floor` (default `true`, so every earlier
  model is unchanged); [pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py) forwards `gcl_train.model.residual_floor`.
- Tests (446 pass; 7 new):
  - `gcn` output has 768 + 256 columns and the SapBERT block is the unit text vector;
  - its `h` changes with the edges while `mlp`'s does not (compared with the last layer enlarged, since the 1e-3 init hides it);
  - the parameter count is 590,848;
  - `lambda = 0` gives zero gradient on the adapter;
  - `residual_floor: false` exports only `h`;
  - pipeline runs complete with `infonce_full` for the floor and no-floor variants and with `edge_rewire`;
  - the `gcn`/`gat` `DualEncoder` tests pass unchanged.
- Analysis scripts: [scripts/srga_pool_diagnostic.py](../../scripts/srga_pool_diagnostic.py) (new, section 4);
  [scripts/srga_paired_ranks.py](../../scripts/srga_paired_ranks.py) gained an optional `keep_candidates` argument (the default
  reproduces the evaluator's pool exactly).
- Configs: [ncit_doid_srga_gcn_sel.yaml](../../configs/experiments/ncit_doid_srga_gcn_sel.yaml) (A12 epochs),
  [ncit_doid_srga_gcn.yaml](../../configs/experiments/ncit_doid_srga_gcn.yaml) (val table: A0, A1, A3, A4, A11, A12, A13),
  and A11/A12 added to [ncit_doid_srga_compare.yaml](../../configs/experiments/ncit_doid_srga_compare.yaml) for the test readout.
- README: the SRGA section lists the `gcn` adapter, `residual_floor` and the pool diagnostic.

## 3. Results

### 3.1 A12 epoch selection (val, seed 42), evaluate run `20261015T201530Z-55822502`

| epochs | val MRR | val Hits@1 | val Hits@10 | train MRR |
|---|---|---|---|---|
| A3 (reference, seed 42) | 0.4756 | 0.3460 | 0.7393 | 0.6370 |
| 50 | 0.5248 | 0.3934 | 0.8057 | 0.8959 |
| 100 | 0.5278 | 0.3981 | 0.7915 | 0.9814 |
| **200** | **0.5394** | 0.4313 | 0.7583 | 0.9894 |

Chosen: **200 epochs** (the highest val MRR; the largest value in the plan's grid, so the optimum may be higher). The
three values are within about 0.015 of each other on 211 pairs, so the choice is weak.

### 3.2 Val table (211 pairs, global; mean ± sample std over 3 seeds), evaluate run `20261015T201905Z-c395a133`

| variant | what it is | val MRR | Hits@1 | Hits@5 | Hits@10 | train MRR |
|---|---|---|---|---|---|---|
| A0 | SapBERT | 0.4125 | 0.2938 | 0.5498 | 0.6398 | 0.4395 |
| A1 | nb, λ=0.25 dir | 0.4555 | 0.3365 | 0.6019 | 0.6919 | 0.4911 |
| A3 | MLP, no edges | 0.4784 ± 0.0042 | 0.3491 ± 0.0055 | 0.6430 ± 0.0179 | 0.7330 ± 0.0072 | 0.657 ± 0.028 |
| A4 | relational | 0.4791 ± 0.0183 | 0.3428 ± 0.0166 | 0.6509 ± 0.0261 | 0.7457 ± 0.0278 | 0.718 ± 0.012 |
| **A11** | **paper GCN + floor** | **0.4924 ± 0.0031** | 0.3633 ± 0.0099 | 0.6509 ± 0.0099 | 0.7472 ± 0.0137 | 0.600 ± 0.005 |
| A12 | paper GCN, no floor | 0.5169 ± 0.0209 | 0.3949 ± 0.0356 | 0.6651 ± 0.0192 | 0.7520 ± 0.0072 | 0.990 ± 0.001 |
| A13 | A11 on a rewired graph | 0.4733 ± 0.0116 | 0.3491 ± 0.0099 | 0.6193 ± 0.0224 | 0.7393 ± 0.0171 | 0.639 ± 0.006 |

Per-seed val MRR (42 / 43 / 44): A11 0.4941 / 0.4888 / 0.4942; A12 0.5394 / 0.5133 / 0.4981; A13 0.4639 / 0.4699 / 0.4862.

Paired wins / losses against A0 on val (mean ± std over seeds):

| variant | wins | losses | ties (of 211) |
|---|---|---|---|
| A3 | 102.3 ± 3.5 | 35.7 ± 6.4 | 73 |
| A4 | 110.0 ± 5.2 | 45.0 ± 6.6 | 56 |
| **A11** | 108.3 ± 3.2 | **24.3 ± 0.6** | 78 |
| A12 | 100.7 ± 3.2 | 60.7 ± 1.5 | 50 |
| A13 | 106.3 ± 2.1 | 37.3 ± 4.0 | 67 |

Paired against each other, same seed (wins / losses / ties of the first variant, seeds 42 / 43 / 44):
- A11 vs A3: 80/54/77, 81/49/81, 72/55/84.
- A11 vs A13: 91/48/72, 77/58/76, 75/56/80.
- A11 vs A4: 65/75/71, 74/74/63, 58/86/67.
- A12 vs A3: 80/72/59, 85/68/58, 78/80/53.

Training loss (epoch 1 → last): A11 6.1–6.2 → 3.55–3.64; A13 6.3 → 3.67–3.77 (50 epochs); A12 9.5–9.8 → 0.10–0.11 (200 epochs).
No NaN in any run.

### 3.3 Test readout (3,280 pairs, once), evaluate run `20261015T203126Z-cfe2330e`

The test numbers of A0, A1, A3 and A4 equal Phase 5.4 exactly (same models). A13 was not read on test, as planned.

**global**

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| A3 MLP | 0.3784 ± 0.0035 | 0.6359 ± 0.0020 | 0.7295 ± 0.0069 | 0.4957 ± 0.0024 |
| A4 relational | 0.3904 ± 0.0129 | 0.6511 ± 0.0097 | 0.7386 ± 0.0109 | 0.5084 ± 0.0104 |
| **A11 GCN + floor** | 0.3982 ± 0.0054 | 0.6732 ± 0.0027 | 0.7708 ± 0.0030 | **0.5234 ± 0.0030** |
| A12 GCN, no floor | 0.4079 ± 0.0102 | 0.6952 ± 0.0044 | 0.7725 ± 0.0040 | **0.5361 ± 0.0073** |

**candidates** (101 per pair)

| method | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| A0 SapBERT | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| A3 MLP | 0.4325 ± 0.0034 | 0.7355 ± 0.0034 | 0.8450 ± 0.0015 | 0.5687 ± 0.0022 |
| A4 relational | 0.4589 ± 0.0096 | 0.7617 ± 0.0044 | 0.8714 ± 0.0026 | 0.5949 ± 0.0070 |
| **A11 GCN + floor** | 0.4440 ± 0.0060 | 0.7613 ± 0.0029 | 0.8710 ± 0.0025 | 0.5852 ± 0.0025 |
| A12 GCN, no floor | 0.4742 ± 0.0134 | 0.8103 ± 0.0023 | 0.9034 ± 0.0045 | **0.6209 ± 0.0088** |

A11 beats A3 and A4 in global MRR and on every global metric. In candidates mode it beats A3 but is level with A4 (0.5852 vs
0.5949 MRR, the Hits@5 and Hits@10 are equal).

### 3.4 Stratified fixed/broken on test (global, vs A0; mean over 3 seeds, rank-1 definition)

Strata sizes: SapBERT right, near 442 / far 681; SapBERT wrong, near 901 / far 1,256 (confuser within 2 hops of the target in DOID).

| variant | fixed | broken | fixed in "wrong, near" | fixed in "wrong, far" | broken, "right, near" | broken, "right, far" |
|---|---|---|---|---|---|---|
| A3 | 255.0 | 137.0 | 84.0 | 171.0 | 57.7 | 79.3 |
| A4 | 416.7 | 259.0 | 171.3 | 245.3 | 84.7 | 174.3 |
| **A11** | 281.0 | **98.0** | 79.7 | 201.3 | 34.0 | 64.0 |
| A12 | 626.7 | 411.7 | 278.7 | 348.0 | 149.3 | 262.3 |

Net rank-1 gain (fixed − broken): A3 +118, A4 +158, A11 +183, A12 +215. A11 improves the rank of 1,679 queries and worsens 322 (5.2 : 1),
against A3's 2.7 : 1 and A4's 2.3 : 1. A12 improves 1,524 and worsens 940 (1.6 : 1).

## 4. Check for a held-out-exclusion shortcut (A12)

**The concern.** Training never sees held-out nodes (val and test endpoints). The ~5k trainable DOID nodes are pushed away from the
training anchors on every step, and A12's train MRR is 0.99, so it memorises them. The gold target is always held-out, so a
model could rank it above the trainable candidates without having learned anything about the pair, and the global pool is 59%
trainable (4,984 of 8,465 DOID nodes).

**The check.** [scripts/srga_pool_diagnostic.py](../../scripts/srga_pool_diagnostic.py) re-scores each query with the candidate
pool cut three ways (the gold target is always kept): all; held-out only (3,481 DOID nodes the loss never saw: no
candidate has been trained against, so a memorisation shortcut cannot help); and trainable only (where a shortcut helps most).
The "all" column reproduces the evaluator exactly.

MRR, mean ± std over seeds (A0 and A1 have no seeds):

| model | val: all | val: held-out only | val: trainable only | test: all | test: held-out only | test: trainable only |
|---|---|---|---|---|---|---|
| A0 | 0.4125 | 0.4805 | 0.5331 | 0.4475 | 0.5195 | 0.5421 |
| A1 | 0.4555 | 0.5268 | 0.5779 | 0.4922 | 0.5690 | 0.5908 |
| A3 | 0.4784 ± 0.0042 | 0.5383 ± 0.0085 | 0.6166 ± 0.0084 | 0.4957 ± 0.0024 | 0.5671 ± 0.0016 | 0.6085 ± 0.0033 |
| A4 | 0.4791 ± 0.0183 | 0.5435 ± 0.0217 | 0.6134 ± 0.0147 | 0.5084 ± 0.0104 | 0.5723 ± 0.0104 | 0.6223 ± 0.0091 |
| A11 | 0.4924 ± 0.0031 | 0.5433 ± 0.0062 | 0.6382 ± 0.0129 | 0.5234 ± 0.0030 | 0.5945 ± 0.0025 | 0.6462 ± 0.0009 |
| A12 | 0.5169 ± 0.0209 | **0.5931 ± 0.0100** | 0.6129 ± 0.0156 | 0.5361 ± 0.0073 | **0.6199 ± 0.0041** | 0.6320 ± 0.0077 |
| A13 | 0.4733 ± 0.0116 | 0.5435 ± 0.0115 | 0.6021 ± 0.0131 | 0.4997 ± 0.0078 | 0.5685 ± 0.0066 | 0.6136 ± 0.0068 |

**Reading.**
- A12's lead over A3 *grows* when the pool is restricted to held-out nodes (val +0.055, test +0.053, against +0.039 and +0.040 on the
  full pool), and it has the highest held-out-only MRR everywhere.
- On the trainable-only pool, the one a shortcut would help most, A12 is *not* ahead: A11 is higher (0.6382 vs 0.6129 val,
  0.6462 vs 0.6320 test).
- So A12's gain is not coming from the held-out exclusion. It ranks held-out targets better against equally unseen
  competitors. Its weakness is the opposite: it is relatively worse against the memorised trainable nodes.
- Every model scores higher in the smaller pools, as expected, so pools should be compared within a column, not across.
- The diagnostic is not a proof of no shortcut at all: it rules out the specific one above.
  This holds for A3, A4, A11 and A13 as well, since the same exclusion applies to all.

## 5. Gate 5.5 and what the controls say

| condition | result |
|---|---|
| A11 ≥ A3 − 1 seed std on val | 0.4924 vs 0.4784 − 0.0042 = 0.4742: **yes** (A11 is +0.014 above A3, about 3 stds) |
| A13 (rewired) < A11 | 0.4733 vs 0.4924: **yes** (A11 beats A13 on all 3 seeds, 91/48, 77/58, 75/56 queries) |

**Gate 5.5: GO.** A11 is the paper-compatible supervised variant.

- **A11 vs A3 (does the GCN branch add to supervision?).** Yes. Val +0.014 MRR, test +0.028 global and +0.016 candidates, a
  paired win on every seed, and 98 broken against 137. Unlike A4 on val, this gain was visible on val and held on test.
- **A11 vs A4 (undirected paper GCN vs the directional relational adapter).** A11 is better on global test MRR (+0.015)
  and has far fewer breaks (98 vs 259). A4 is level on candidates and doubles the fixes where SapBERT's confuser is a near
  neighbour (171 vs 80). On val, the pairs split evenly (65/75, 74/74, 58/86).
- **A13 vs A11 (graph or capacity?).** The graph. A rewired graph loses −0.019 val MRR and wins fewer pairs on every seed. The
  effect is smaller than the A5 vs A4 drop in Phase 5.3 (−0.108), because the GCN reaches the neighbours through a shared
  symmetric average and the SapBERT block is the same in both, so wrong neighbours cost less.
- **A12 vs A11 (what the SapBERT floor is worth).** It costs MRR (−0.013 global, −0.036 candidates on test) and buys safety: A11
  breaks 98 of 1,123 SapBERT-correct queries, A12 breaks 412. A12 has by far the largest overfit (train MRR 0.99).
- **A12 vs Sprint 4.** The Sprint 4 supervised GCN (triplet loss, DeepWalk, 128-d, no floor) scored 0.223 global test MRR. A12,
  the same kind of GCN with full-softmax InfoNCE, no DeepWalk, no graph-view loss and 256-d, scores 0.536. So the Sprint 4
  failure was mainly the objective, not the architecture and not the missing floor. This is not a clean single-factor
  ablation, since A12 changed the loss, the embedding width and removed DeepWalk together.

## 6. Limitations

- **Val is small** (211 pairs). The A11–A4 and A12-vs-A3 val differences are inside noise, and only the A13 drop and A11's low
  loss counts are clear. The test readout (3,280 pairs) is the more reliable comparison.
- **A12's seed variance is the largest** (val std 0.021; Hits@1 std 0.036), and its epochs choice was weak (section 3.1) and sits at
  the edge of the grid.
- **A12 vs the plan's selection rule.** A12 has the highest test MRR but a 37% break rate on SapBERT-correct queries and no floor.
  Whether MRR or safety matters more is the developer's call, and I make no claim that A12 replaces A3 or A11.
- **A13 was not read on test**, as planned.
- **The shortcut check (section 4)** rules out one specific mechanism, not every leak. The training is also transductive:
  test nodes take part in message passing as features only.
- The GCN uses the symmetrized edge list with duplicates (as the paper's `gcn` encoder does), while `relational` collapses them.
- Seeds are not bit-reproducible on this machine. `ruff` is not installed, so lint was not run.

## 7. Files

- Modified: `src/cga/pipelines/gcl_train/{models,pipeline}.py`, `tests/unit/pipelines/gcl_train/{test_models,test_pipeline}.py`,
  `scripts/srga_paired_ranks.py`, `configs/experiments/ncit_doid_srga_compare.yaml`, `README.md`, `dev_docs/sprints/sprint_5.md`
  (Phase 5.5 boxes ticked).
- Created: `scripts/srga_pool_diagnostic.py`, `configs/experiments/ncit_doid_srga_gcn_sel.yaml`,
  `configs/experiments/ncit_doid_srga_gcn.yaml`, this log.
- Untracked artifacts: model lineages `srga-gcn-l0.25-s<seed>`, `srga-gcn-rewired-l0.25-s<seed>`, `gcn-full-e{50,100,200}-s42`,
  `gcn-full-e200-s{43,44}` under `models/gcl/ncit_doid_srga/` (all dated 20261015; A11 s42/43/44 `T201253Z-1016474d` /
  `T201327Z-087885c8` / `T201359Z-3c8831a6`; A13 `T201310Z-14c498a5` / `T201343Z-b68619d2` / `T201415Z-01d3ca24`;
  A12 e200 `T201502Z-8c3ad332` / `T201822Z-eab07537` / `T201843Z-0b73214d`), and the evaluate runs `20261015T201530Z-55822502`
  (A12 epochs), `20261015T201905Z-c395a133` (val table), `20261015T203126Z-cfe2330e` (test readout).

## 8. Commands

```
uv run pytest tests/unit -q

# A11 (seed 43 shown). A13: add the two edge_rewire --set lines. A12: add residual_floor=false and use epochs 200.
uv run cga run gcl_train --dataset ncit_doid_srga --set seed=43 --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=gcn --set gcl_train.model.lambda=0.25 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.model.model_id=srga-gcn-l0.25-s43
#   A13 extra: --set gcl_train.model.edge_rewire.enabled=true --set gcl_train.model.edge_rewire.seed=43 --set gcl_train.model.model_id=srga-gcn-rewired-l0.25-s43
#   A12 extra: --set gcl_train.model.residual_floor=false --set gcl_train.train.epochs=200 --set gcl_train.model.model_id=gcn-full-e200-s43

uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_gcn_sel
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_gcn
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_compare

uv run python scripts/srga_paired_ranks.py --dataset ncit_doid_srga --split val --group A3="srga-mlp-l0.25-s{seed}" --group A4="srga-rel-l0.25-s{seed}" --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}" --group A13="srga-gcn-rewired-l0.25-s{seed}"
uv run python scripts/srga_stratified.py --dataset ncit_doid_srga --split test --group A3="srga-mlp-l0.25-s{seed}" --group A4="srga-rel-l0.25-s{seed}" --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}"
uv run python scripts/srga_pool_diagnostic.py --dataset ncit_doid_srga --split test --group A3="srga-mlp-l0.25-s{seed}" --group A11="srga-gcn-l0.25-s{seed}" --group A12="gcn-full-e200-s{seed}"
```
