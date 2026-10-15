# Sprint 5, Phase 5.2: Trainable SRGA core without edges (A3)

Plan: [sprint_5.md](../sprints/sprint_5.md). Previous phase: [sprint_5_1.md](sprint_5_1.md). Lineage `ncit_doid_srga`.
No test split is read in this phase (the single test readout is Phase 5.4).

## 1. Model architecture

`ResidualAdapterEncoder(adapter="mlp")` in [models.py](../../src/cga/pipelines/gcl_train/models.py). It has no graph input.

```
SapBERT store (frozen, 768-d)
        |  x̂ = x / ||x||                         (detached: never trained)
        |--------------------------------------------------------------+
        v                                                              |
  Linear(768 -> 256) -> ReLU -> LayerNorm(256) -> Dropout(0.2)         |
        -> Linear(256 -> 256)            = h   (last layer: W ~ N(0, 1e-3^2), b = 0)
        |                                                              |
        v                                                              v
   ĥ = h / ||h||  --- x sqrt(lambda) ---> [ x̂ (768) ;  sqrt(lambda) * ĥ (256) ]  =  z   (1024-d, exported)
```

- Trainable parameters: 263,168 (`768*256+256` + `2*256` + `256*256+256`). SapBERT block and lambda are fixed.
- lambda = **0.25**, the Phase 5.1 winner (fixed, global, not learned).
- Export layout is unchanged: `z` is saved as `node_embeddings`, and the `gold_alignment` evaluator scores it by cosine.

### Equations

Score between a source concept `a` (NCIT) and a target concept `d` (DOID), by dot product on `z`:

```
z_a · z_d  =  x̂_a · x̂_d  +  lambda * ĥ_a · ĥ_d          (cosine of z ranks the same, up to per-node norms)
s(a, d)    =  cos(z_a, z_d)
```

Loss (full-softmax InfoNCE, [losses.py](../../src/cga/pipelines/gcl_train/losses.py) `infonce_full`), over the distinct
train pairs `P = {(a, p)}` and temperature `tau`:

```
L_fwd = - mean_{(a,p) in P}  log  exp(s(a,p)/tau) / sum_{d in D \ G(a)\{p}} exp(s(a,d)/tau)
L_rev = - mean_{(a,p) in P}  log  exp(s(p,a)/tau) / sum_{n in N \ G^-1(p)\{a}} exp(s(p,n)/tau)
L     = L_fwd + w * L_rev          (w = reverse_weight = 0.5)
```

- `D` = all non-held-out DOID nodes (8,465 minus test/val endpoints); `N` = all non-held-out NCIT nodes.
- `G(a)` = every gold target of `a` in **any** split (train, val, test); `G^-1(p)` = every gold source of `p`. They are
  masked to `-inf`, so a valid alternative is never scored as a negative.
- There are no sampled negatives (the `examples.parquet` negatives are ignored) and no graph-view triplet loss (`alpha`
  is ignored). Positives are the distinct `(anchor_id, text_pos_id)` train rows: 1,174 pairs.
- Optimizer: Adam, lr 1e-3, weight decay 1e-4, full batch.

Note on initialisation: `ĥ` is length-normalised, so the 1e-3 last-layer init makes `h` small but `ĥ` still unit length.
At step 0, `z` is SapBERT plus a `sqrt(lambda)`-scaled block that is a smooth function of `x̂` (not a tiny perturbation).
`lambda = 0` is the only setting that makes `z` exactly SapBERT (tested: zero gradient on the adapter).

## 2. What was implemented

- [models.py](../../src/cga/pipelines/gcl_train/models.py): `mlp` adapter (above); constructor takes `text_dim` and
  `adapter_dropout`; `relational` is still rejected. `DualEncoder.forward` accepts and ignores `edge_type`, so the trainer
  has one call signature.
- [losses.py](../../src/cga/pipelines/gcl_train/losses.py): `FullSoftmaxData`, `build_full_softmax_data` (candidate sets,
  gold masks, de-duplicated pairs; raises if a pair endpoint is held-out) and the registered `infonce_full`.
  It takes the whole embedding matrix and returns `(total, forward, reverse)`, so the trainer branches on its name.
- [trainer.py](../../src/cga/pipelines/gcl_train/trainer.py): `train_model(..., edge_type, full_data)`. For
  `infonce_full` the graph view is skipped; `metrics.jsonl` records `loss_text` = forward term and `loss_graph` =
  reverse term (kept so the loss-curve plot works).
- [pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py): when `gcl_train.loss.name == infonce_full`, the load
  step reads the ingest `alignments.parquet` (all splits, restricted to the universe) for the gold mask, and the train
  step builds the candidate sets from the universe `held_out` flags. `loss.temperature` (default 0.05) and
  `loss.reverse_weight` (default 0.5) are read. Other loss names do not read alignments, so existing runs are unchanged.
- Tests (430 pass; 10 new): `infonce_full` equals a hand-computed value on a 2 x 4 fixture; masked gold targets leave
  the softmax and carry no gradient; held-out nodes are never candidates and held-out pair endpoints raise; the reverse
  weight is respected; the MLP shapes, init and dropout (eval is deterministic and ignores edges); `lambda = 0` gives zero
  gradient on the adapter; a 40-epoch trainer run lowers the loss and finite metrics are written; an end-to-end
  pipeline run. The `triplet`/`infonce` tests are unchanged and pass.
- Configs: [ncit_doid_srga_mlp_sel.yaml](../../configs/experiments/ncit_doid_srga_mlp_sel.yaml) (tau x epochs selection)
  and [ncit_doid_srga_mlp.yaml](../../configs/experiments/ncit_doid_srga_mlp.yaml) (the 3 seeds).

## 3. Results

All numbers: global mode. `gold_val` has 211 pairs; `gold_train` has 1,195 pairs (informational: it shows whether the
under-fitting seen in Sprint 4 is gone, and is not used to choose anything).

### Selection on val (seed 42, lambda 0.25), evaluate run `20261015T165152Z-3e3b7e38`

| tau | epochs | val MRR | val Hits@1 | val Hits@10 | train MRR |
|---|---|---|---|---|---|
| SapBERT (A0) | - | 0.4125 | 0.2938 | 0.6398 | 0.4395 |
| 0.05 | **50** | **0.4765** | 0.3507 | 0.7299 | 0.6685 |
| 0.05 | 100 | 0.4505 | 0.3318 | 0.6872 | 0.8059 |
| 0.05 | 200 | 0.3977 | 0.2938 | 0.6114 | 0.8941 |
| 0.1 | 50 | 0.4351 | 0.3081 | 0.6872 | 0.5412 |
| 0.1 | 100 | 0.4245 | 0.3081 | 0.6540 | 0.6306 |
| 0.1 | 200 | 0.3832 | 0.2844 | 0.5782 | 0.7059 |

Chosen: **tau = 0.05, 50 epochs**. More epochs fit the train pairs much better (train MRR 0.67 to 0.89) and lower val
MRR, so the model over-fits. 50 epochs is the smallest value in the plan's grid, so the true optimum may be lower.

### Chosen setting, 3 seeds, evaluate run `20261015T165533Z-a3588e37`

| model | val MRR | val Hits@1 | val Hits@5 | val Hits@10 | train MRR |
|---|---|---|---|---|---|
| SapBERT (A0) | 0.4125 | 0.2938 | 0.5498 | 0.6398 | 0.4395 |
| A1 (nb, l=0.25 dir) | 0.4555 | 0.3365 | 0.6019 | 0.6919 | 0.4911 |
| A3 seed 42 | 0.4756 | 0.3460 | 0.6303 | 0.7393 | 0.6370 |
| A3 seed 43 | 0.4764 | 0.3460 | 0.6351 | 0.7251 | 0.6887 |
| A3 seed 44 | 0.4832 | 0.3555 | 0.6635 | 0.7346 | 0.6452 |
| **A3 mean +- std** | **0.4784 +- 0.0042** | 0.3491 +- 0.0055 | 0.6430 +- 0.0179 | 0.7330 +- 0.0072 | 0.6569 +- 0.0278 |

(std is the sample std over 3 seeds.)

Models (`models/gcl/ncit_doid_srga/`, dated 20261015): `srga-mlp-l0.25-s42` `T165407Z-7619df52`, `-s43`
`T165432Z-64cb0e30`, `-s44` `T165458Z-396155cf`. The 6 selection runs are `srga-mlp-sel-t<tau>-e<epochs>`.

Training loss (50 epochs, seeds 42/43/44): 6.19/6.21/6.16 at epoch 1 to 3.61/3.29/3.53 at epoch 50. No NaN.
The loss rose on 4, 1 and 3 of the 49 epoch-to-epoch steps (dropout noise early in training); the trend is down.

## 4. Acceptance and gate 5.2: GO

- Loss decreases with no NaN: yes.
- Training-set MRR above SapBERT's: yes (0.657 vs 0.440, mean of 3 seeds).
- Val MRR (mean of 3 seeds) >= A0: yes, 0.4784 vs 0.4125 (+0.066).
- A3 vs A1: A3 is higher on val by +0.023 MRR, about 5 seed std. This is a single val set of 211 pairs.

Reading: supervision alone, with no structure, already beats SapBERT and also the training-free neighbour mean. So the
bar for Phase 5.3 is higher: A4 must beat A3 (0.4784 +- 0.0042), not only A1, by more than the seed std, with the
rewired control A5 clearly below A4.

## 5. Caveats

- Selection used one seed and 211 val pairs, so the tau/epoch choice is noisy. The over-fitting trend (more epochs ->
  lower val MRR) is large and consistent across both temperatures, so it is likely real.
- The optimum epoch count may be below 50 (edge of the plan's grid); not explored.
- The seed-42 selection run (val MRR 0.4765) and the later seed-42 run of the same setting (0.4756) differ slightly, so
  runs are not bit-reproducible on this machine.
- Paired win/loss counts against A0 need per-query ranks, which the evaluator does not export; they are left to 5.3/5.4.
- Training is transductive (val/test nodes are in the forward pass as features, never in a loss term).
- `ruff` is not installed, so lint was not run.

## 6. Files

Modified: `src/cga/pipelines/gcl_train/{models,losses,trainer,pipeline}.py`, tests in
`tests/unit/pipelines/gcl_train/{test_models,test_losses,test_trainer,test_pipeline}.py`,
`dev_docs/sprints/sprint_5.md` (Phase 5.2 boxes ticked).
Created: `configs/experiments/ncit_doid_srga_mlp_sel.yaml`, `configs/experiments/ncit_doid_srga_mlp.yaml`, this log.
Untracked artifacts: 9 new model lineages under `models/gcl/ncit_doid_srga/` and two evaluation runs.

## 7. Commands

```
uv run pytest tests/unit -q

# one selection run (tau in {0.05, 0.1}, epochs in {50, 100, 200})
uv run cga run gcl_train --dataset ncit_doid_srga --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=mlp --set gcl_train.model.lambda=0.25 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.model.model_id=srga-mlp-sel-t0.05-e50
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_mlp_sel

# chosen setting, seeds 42/43/44
uv run cga run gcl_train --dataset ncit_doid_srga --set seed=43 --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=mlp --set gcl_train.model.lambda=0.25 --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05 --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50 --set gcl_train.model.model_id=srga-mlp-l0.25-s43
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_mlp
```

Use `0.0001`, not `1e-4`, in `--set`: the YAML parser reads `1e-4` as a string.
