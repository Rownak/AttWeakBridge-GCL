# Sprint 12.5 — Execution log: gated residual graph adapter over frozen text

Phase 12.5 of [sprint_12.md](../sprints/sprint_12.md); idea in [gcl_method_update.md](../design_docs/gcl_method_update.md) §3.
All planned runs were trained; every check passed. Evaluation, plots and the paper table are Phase 12.6.

## 1. What was implemented

**Model code only** (data, loss, evaluators and the existing encoders are unchanged).

- `GatedResidualEncoder` / `build_gated_residual_encoder` in [models.py](../../src/cga/pipelines/gcl_train/models.py), selected by `gcl_train.model.encoder: residual_gated`.
  - Frozen text `x` (768-d, detached), `Â` = `D^-1/2 (A + I) D^-1/2` over the same symmetrised intra-graph `edge_index` (`_propagate`; an isolated node keeps its own features), `s` = DeepWalk (128-d).
  - Text variant: `z = LN(x + g ⊙ f_t(Â X))`, `f_t = Linear(768,128)–ReLU–Linear(128,768)`, `g = σ(W_g [x ‖ ÂX] + b_g)` with `W_g: Linear(1536, 768)`, per node and per dimension.
  - Dual variant (`graph_branch: true`): `z = LN(x + g^t ⊙ f_t(ÂX) + g^s ⊙ f_s(ÂS))`, `g^s = σ(W_s [x ‖ ÂS] + b_s)`, `f_s = Linear(128,128)–ReLU–Linear(128,768)`; the gates arbitrate instead of concatenating.
  - `LN` = parameter-free `layer_norm` over 768 dims (`norm: none` removes it; kept only as the sanity fallback).
  - Init: last layer of every `f` at std 1e-3 / zero bias, gate weights std 1e-3, gate bias `gate_bias_init = -4` (σ = 0.018). Untrained, the model is `LN(x)`.
  - `gate`: `learned` | `none` (g = 1) | `fixed_zero` (g = 0, **no parameters**, so the pipeline's existing zero-parameter path skips the optimiser and writes an empty `metrics.jsonl`).
  - `gate_values()` returns the per-node mean gate (`(n,)`, or `(n, 2)` text / graph for the dual variant); `gate_parameters()` returns the gate projections.
- [pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py): builds the encoder from `adapter_dim` (default 128), `gate`, `graph_branch`, `gate_bias_init`, `norm`; Adam gets two parameter groups for this encoder (gate weights with `gate_weight_decay`, default 1e-4; everything else with the global `optim.weight_decay`, 0.0); `ExportStep` writes `node_embeddings/gate.npy` when the model has a learned gate, and the manifest lists it as `node_embeddings_gate`.
- [scripts/gate_diagnostics.py](../../scripts/gate_diagnostics.py): per graph role, mean / median / p05 / p95 of the gate, share below 0.05 and above 0.95, Spearman correlation with undirected degree (distinct neighbours from the pinned ingest `edges.parquet`), plus `gate_hist.png` and `gate_vs_degree.png`.
- [configs/experiments/attack_cwe_ent_gated_sanity.yaml](../../configs/experiments/attack_cwe_ent_gated_sanity.yaml): G4 vs frozen mpnet-chunk with the 12.3 protocol (used only for the sanity check).

**Files created:** `scripts/gate_diagnostics.py`, `tests/unit/pipelines/gcl_train/test_gated_residual.py`, `tests/unit/test_gate_diagnostics.py`, `configs/experiments/attack_cwe_ent_gated_sanity.yaml`.
**Files modified:** `src/cga/pipelines/gcl_train/models.py`, `src/cga/pipelines/gcl_train/pipeline.py`, `dev_docs/sprints/sprint_12.md` (Phase 12.5 added earlier; boxes ticked now).
Outputs: `models/gcl/attack_cwe_ent/ent-g*-gated*/`, `artifacts/paper/attack_cwe_ent/sprint_12_gated/gate/{g3,g5}/`, evaluate run `artifacts/evaluations/attack_cwe_ent/20261028T160520Z-c7e8fd6f/` (sanity).

## 2. Tests

`tests/unit/pipelines/gcl_train/test_gated_residual.py` (12 cases): `fixed_zero` equals `LN(x)` exactly with 0 parameters; untrained `learned` has mean gate ≈ σ(−4) and output within 1e-2 of `LN(x)`; `none` has no gate parameters and equals `LN(x + f(ÂX))`; the dual model with both gates forced to 0 equals `LN(x)` (gate `(n, 2)`); isolated nodes keep their features through `_propagate` and nothing is NaN, connected pair weights 1/2; gate-bias init and the gate parameter group; shapes, finite output, gradients reach the gate only for `learned`; same seed gives the same output and `x` receives no gradient; unknown `gate` / `norm` rejected; existing encoders unchanged; two pipeline end-to-end tests (learned text variant exports `gate.npy` and the manifest output; dual exports `(n, 2)`; `fixed_zero` has 0 trainable parameters and no `gate.npy`). `tests/unit/test_gate_diagnostics.py` (6 cases): degree, statistics, per-role report with degree correlation, dual columns, row-mismatch error, figures written.

`uv run pytest tests/unit -q`: **636 passed** (was 618).

## 3. Training runs

Common to every run: dataset `attack_cwe_ent`, `--experiment unsup_u2`, text store mpnet-chunk `20261027T194928Z-8cf38e55`, DeepWalk `20261027T135210Z-3b18ba0c`, ingest `20261027T194643Z-5d3f8baf`, `encoder: residual_gated`, `adapter_dim 128`, `gate_bias_init −4`, `gate_weight_decay 1e-4`, `norm: layernorm`, loss `infonce` (temperature 0.1, `alpha 0.5`), Adam lr 1e-3, 100 epochs, CPU, one thread (`OMP_NUM_THREADS=1`). Bottom-5 pseudo-pairs: sampler `s11c-rank_topk-k5`, training data run `20261027T205523Z-ac890ca3`; hard-negative pairs: `s11c-rank_topk-k5-xhard`, run `20261028T023558Z-decc373b`.

| Id | Model id | Run id | Seed | Gate / branch | Pairs | Params | Loss ep 1 → 100 |
|---|---|---|---|---|---|---|---|
| G2 | `ent-g2-gated-mpnetc-s42` | `20261028T154339Z-6f4468f5` | 42 | none / text | bottom-5 | 197,504 | 8.441 → 6.961 |
| G2 | `ent-g2-gated-mpnetc-s43` | `20261028T154339Z-24ab4772` | 43 | none / text | bottom-5 | 197,504 | 8.441 → 6.940 |
| G2 | `ent-g2-gated-mpnetc-s44` | `20261028T154339Z-6c92f9c0` | 44 | none / text | bottom-5 | 197,504 | 8.441 → 6.951 |
| G3 | `ent-g3-gated-mpnetc-s42` | `20261028T153145Z-bd1e8273` | 42 | learned / text | bottom-5 | 1,377,920 | 8.441 → 6.593 |
| G3 | `ent-g3-gated-mpnetc-s43` | `20261028T154340Z-2f19bae8` | 43 | learned / text | bottom-5 | 1,377,920 | 8.441 → 6.585 |
| G3 | `ent-g3-gated-mpnetc-s44` | `20261028T154339Z-6f657c79` | 44 | learned / text | bottom-5 | 1,377,920 | 8.441 → 6.584 |
| G4 | `ent-g4-gated-mpnetc-s42` | `20261028T153136Z-3749de5e` | 42 | fixed 0 / text | bottom-5 | 0 | no training |
| G5 | `ent-g5-gated-mpnetc-s42` | `20261028T154340Z-fdcd6c82` | 42 | learned / dual | bottom-5 | 2,182,400 | 8.441 → 6.170 |
| G5 | `ent-g5-gated-mpnetc-s43` | `20261028T154340Z-2bc03e86` | 43 | learned / dual | bottom-5 | 2,182,400 | 8.441 → 6.154 |
| G5 | `ent-g5-gated-mpnetc-s44` | `20261028T154340Z-bd006915` | 44 | learned / dual | bottom-5 | 2,182,400 | 8.441 → 6.169 |
| G3-hard | `ent-g3-gated-hard-mpnetc-s42` | `20261028T154340Z-da58604c` | 42 | learned / text | hard (M = 50) | 1,377,920 | 8.689 → 7.206 |
| G5-hard | `ent-g5-gated-hard-mpnetc-s42` | `20261028T154340Z-18f16e45` | 42 | learned / dual | hard (M = 50) | 2,182,400 | 8.689 → 6.945 |
| G3 rerun | `ent-g3-gated-rerun-mpnetc-s42` | `20261028T154340Z-fae791e1` | 42 | learned / text | bottom-5 | 1,377,920 | 8.441 → 6.593 |

The epoch-1 loss is the same (8.441) for all runs of one pair set and every seed because the untrained model is `LN(x)` at every seed (the gate and `f` start at near-constant values).

**Checks per run (all 13 trained runs plus G4):** manifest `status: success`; 100 `metrics.jsonl` rows and loss lower at epoch 100; no NaN; `node_embeddings` 2,927 × 768, finite, index equals the universe of `20261027T205523Z-ac890ca3`; the resolved config shows `encoder: residual_gated`, the intended `gate` and `graph_branch`, the training data run and sampler id, and the seed. G4 has 0 trainable parameters, so `metrics.jsonl` is empty and the embeddings are `LN(x)` (max difference 0.0 against `layer_norm` of the pinned mpnet-chunk store). Trainable-parameter counts follow from the layer sizes (G3 = 197,504 for `f_t` + 1,180,416 for `W_g`; G5 adds `f_s` 115,584 and `W_s` 688,896 = 804,480).

**Reproducibility:** the G3 seed-42 rerun (`…fae791e1`, a different model id so it does not overwrite the original) has the same `vectors.npy` SHA-256 as the original (`0e0f99c677af…`): bit-identical. Different seeds give different embeddings (G3 s43 `637854eb…`, s44 `efc2146e…`).

Trained in parallel (11 runs at once, one thread each, G3 / G4 seed 42 first); each run took about 15–30 min on CPU. Parallel execution did not change results (the rerun check above).

## 4. Gate diagnostics (G3 and G5, seed 42, bottom-5)

Initial gate σ(−4) = 0.018; `artifacts/paper/attack_cwe_ent/sprint_12_gated/gate/{g3,g5}/` (`gate_diagnostics.md` / `.json`, `gate_hist.png`, `gate_vs_degree.png`).

| Model | Gate | Graph | mean | median | p05 | p95 | share < 0.05 | share > 0.95 | Spearman(degree) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| G3 | text | ATT&CK | 0.0870 | 0.0887 | 0.0450 | 0.1232 | 0.083 | 0.000 | −0.479 |
| G3 | text | CWE | 0.1032 | 0.1019 | 0.0656 | 0.1474 | 0.006 | 0.000 | −0.155 |
| G5 | text | ATT&CK | 0.0954 | 0.0965 | 0.0633 | 0.1238 | 0.004 | 0.000 | −0.274 |
| G5 | text | CWE | 0.1331 | 0.1330 | 0.0836 | 0.1853 | 0.000 | 0.000 | 0.295 |
| G5 | graph | ATT&CK | 0.1730 | 0.1755 | 0.1279 | 0.2121 | 0.000 | 0.000 | −0.062 |
| G5 | graph | CWE | 0.1500 | 0.1486 | 0.1083 | 0.1992 | 0.000 | 0.000 | 0.163 |

Mean gate over all nodes for the other runs: G3 s43 / s44 0.088 / 0.094 (max 0.19); G5 s43 text / graph 0.108 / 0.177, s44 0.109 / 0.194; G3-hard 0.061; G5-hard text / graph 0.075 / 0.179.

Reading (descriptive; no tuning): the gate neither collapsed to 0 nor saturated to 1. It rose from 0.018 to about 0.09–0.19 and stays in a narrow band (p95 ≤ 0.21), i.e. the models use a small, intermediate share of the graph signal, and in the dual model the DeepWalk gate is larger than the text gate. Higher-degree ATT&CK nodes get a lower gate in G3 (Spearman −0.48); for CWE the relation is weak in G3 and positive in G5 (+0.30), so there is no single degree trend. The hard-negative runs use a smaller gate (G3 0.061, G5 text 0.075). The gate magnitude by itself does not show whether structure helps; that is judged by the Phase 12.6 evaluation.

## 5. Sanity: G4 against frozen mpnet-chunk

Evaluate run `20261028T160520Z-c7e8fd6f` (config `attack_cwe_ent_gated_sanity`, 12.3 protocol, methods frozen mpnet-chunk and G4).

- Top-10 cosine-neighbour overlap between `LN(x)` and frozen `x`, within each graph role over the universe: **0.999** (ATT&CK), **0.9993** (CWE).
- The 12 neighbourhood columns of the 12.4 table agree to 4 decimals except 1e-4 differences in three P@10 cells (ATT&CK `structural_r1` 0.1588 vs 0.1589, textual ATT&CK 0.1907 vs 0.1908, textual CWE 0.2421 vs 0.2422). The silhouette column differs by 0.0016 (0.0542 vs 0.0558), the largest difference. It is also the column that scores the frozen store on more nodes (Phase 12.3 caveat), so this is not a pure LayerNorm effect. `n_queries` is identical for both methods in every cell.
- Maximum absolute difference over all 13 columns = **0.0016 < 0.01**, so the `norm: none` control was not needed and was not trained.

## 6. Slow-training risk

The criterion was a G3 mean gate below 0.05 at epoch 100. G3 seed 42 reached 0.092 (s43 0.088, s44 0.094), so the 300-epoch G3 run was **not** needed and not run. Nothing else was tuned.

## 7. Commands

```
uv run pytest tests/unit -q                                   # 636 passed

# one training run (all 13 differ only in these --set values); the driver scripts were kept in the session scratchpad
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
  --set seed=<42|43|44> --set device=cpu --set gcl_train.loss.name=infonce \
  --set gcl_train.model.encoder=residual_gated --set gcl_train.model.model_id=<ent-gN-gated[-hard]-mpnetc-sS> \
  --set gcl_train.model.gate=<learned|none|fixed_zero> --set gcl_train.model.graph_branch=<true|false> \
  --set gcl_train.model.adapter_dim=128 \
  --set gcl_train.inputs.training_data_sampler_id=<s11c-rank_topk-k5 | s11c-rank_topk-k5-xhard> \
  --set gcl_train.inputs.training_data_run=<20261027T205523Z-ac890ca3 | 20261028T023558Z-decc373b>

uv run python scripts/gate_diagnostics.py --run models/gcl/attack_cwe_ent/ent-g3-gated-mpnetc-s42/20261028T153145Z-bd1e8273 \
  --ingest data/processed/attack_cwe_ent/20261027T194643Z-5d3f8baf --out artifacts/paper/attack_cwe_ent/sprint_12_gated/gate/g3
uv run python scripts/gate_diagnostics.py --run models/gcl/attack_cwe_ent/ent-g5-gated-mpnetc-s42/20261028T154340Z-fdcd6c82 \
  --ingest data/processed/attack_cwe_ent/20261027T194643Z-5d3f8baf --out artifacts/paper/attack_cwe_ent/sprint_12_gated/gate/g5
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_gated_sanity \
  --set evaluate.inputs.ingest_run=20261027T194643Z-5d3f8baf
```

## 8. Limits and assumptions

- Design choices not fixed by the idea (all declared in the sprint file, none tuned): per-dimension gate with the per-node score as its mean, bottleneck 128, gate bias −4, gate weight std 1e-3, `gate_weight_decay` 1e-4, parameter-free LayerNorm. The gate weight init at 1e-3 is my choice to make the initial gate exactly σ(bias); other values may train faster or slower.
- The G3 rerun uses a different `model_id` (`…-rerun-…`), so its config hash differs from the original; every other config value is identical.
- G4 has no trainable path, so it has no loss curve; its `metrics.jsonl` is empty by design (existing zero-parameter behaviour).
- Training loss values are not a quality measure across models (different output spaces); quality is judged in Phase 12.6.
- Seeds 43 / 44 exist only for G2, G3 and G5; the seed-42 rows are the ones compared with A0 / A1 / A2.
- Gold-based evaluation of the gated models is outside this sprint.

## 9. Status

Phase 12.5 acceptance met: the gated encoder is selectable by config only, every run above has a pinned run id and passed the checks, no existing run or default behaviour changed (existing tests pass; A0 / A1 / A2 untouched). Next: Phase 12.6.
