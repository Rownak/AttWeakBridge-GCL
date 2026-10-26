# Sprint 9, Phase 9.5: SapBERT residual with SapBERT / DeepWalk GCN branches

Plan: [sprint_9.md](../sprints/sprint_9.md). Previous: [9.4](sprint_9_4.md). Decision: report only.

## 0. Outcome in brief

- **Score fusion helps, trained residuals do not.** Fusing frozen SapBERT with the 9.4 DW2-I embeddings (no training) beats frozen SapBERT: w = 0.25 gives MRR 0.4681 and F1 0.4367 (frozen: 0.4475 / 0.4136), w = 0.1 gives F1 0.4374. DW2-I therefore carries signal SapBERT lacks.
- **None of the three trained residual rows beats frozen SapBERT** (MRR 0.4475, F1 0.4136): R1 0.4144 / 0.3725, R2 0.4295 / 0.3948, R3 0.4109 / 0.3674 (MRR / F1).
- **DeepWalk adds nothing on top of the text GCN:** R3 is below R1 by 0.0035 MRR (inside the seed std ~0.008). R2 (DeepWalk branch alone) is the best trained row, 0.015 MRR above R1.
- **The training-free neighbour residual (A1) is still the best:** MRR 0.4922, F1 0.4601, above every trained row and every fusion.

## 1. What changed

- `gcl_train.model.graph_branch: true` (default off) adds a second residual to `residual_adapter`: a GCN over the graph (DeepWalk) store, built like the `gcn` adapter, appended as `sqrt(lambda) * normalize(h_graph)`: `z = [x̂ ; text residual ; graph residual]`. `adapter: none` drops the text residual (R2). Refused: `none` without the branch; the branch with `residual_floor: false`. Recorded in the manifest summary. Files: `src/cga/pipelines/gcl_train/models.py`, `pipeline.py`.
- `scripts/run_unsup_matrix.sh` priority 6 (`res-r1`, `res-r2`, `res-r3`), each seed on the `sap-rank_topk-k5` run built with that seed. `scripts/unsup_summary.py` `--rows residual`.
- Configs: `ncit_doid_unsup_fusion_test.yaml`, `ncit_doid_unsup_residual_test.yaml`.
- Tests: graph-branch layout, off-by-default equality, `none` / invalid options (models), end-to-end training and manifest (pipeline). The gcl_train tests pass (80).

## 2. Runs

- **Fusion check** (evaluation `20261026T162838Z-89e91493`): `sap-u0` plus `SapBERT + w × DW2-I` for w in {0.1, 0.25, 0.5, 1} (fusion method kind; ranks by `cos_S + w · cos_D`), each with the three `dw-u2-i-s<seed>` models. The whole grid is reported.
- **R0 / A1:** R0 is frozen SapBERT. A1 is a parameter-free `residual_adapter` (`neighbor_mean`, directional, λ 0.25), run `res-a1` (`20261026T162822Z-9a1cc189`).
- **R1, R2, R3:** 9 runs (3 seeds), `residual_adapter`, λ 0.25 per branch, `adapter_dim` 256, InfoNCE (temperature 0.1), Adam 1e-3, 100 epochs, last epoch, SapBERT pairs of the seed. Nothing re-tuned. Log `logs/sprint_9_5_train.log`.
- **Evaluation, once:** `20261026T181041Z-10f16c38` (`ncit_doid_unsup_residual_test`).

## 3. Results (NCIT–DOID test, 3,280 pairs)

Fusion grid (3 seeds of DW2-I):

| w | Hits@1 | Hits@10 | MRR | P | R | F1 |
|---|---|---|---|---|---|---|
| frozen SapBERT | 0.3424 | 0.6537 | 0.4475 | 0.4663 | 0.3716 | 0.4136 |
| 0.1 | 0.3560 ± 0.0028 | 0.6925 ± 0.0009 | 0.4679 ± 0.0019 | 0.4937 | 0.3927 | 0.4374 ± 0.0021 |
| 0.25 | 0.3491 ± 0.0033 | 0.7052 ± 0.0015 | 0.4681 ± 0.0024 | 0.4954 | 0.3904 | 0.4367 ± 0.0021 |
| 0.5 | 0.3225 ± 0.0025 | 0.6935 ± 0.0029 | 0.4452 ± 0.0025 | 0.4692 | 0.3638 | 0.4098 ± 0.0053 |
| 1.0 | 0.2780 ± 0.0016 | 0.6444 ± 0.0016 | 0.3983 ± 0.0034 | 0.4270 | 0.3206 | 0.3663 ± 0.0061 |

Residual rows (mean ± std over 3 seeds):

| Variant | Pseudo-pairs | Loss | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 SapBERT (frozen) | – | – | 0.3424 | 0.6537 | 0.4475 | 0.4663 | 0.3716 | 0.4136 | 2614.0 | 1219.0 |
| R0 SapBERT + training-free neighbour residual (A1) | – | – | 0.3817 | 0.7122 | 0.4922 | 0.5125 | 0.4174 | 0.4601 | 2671.0 | 1369.0 |
| R1 SapBERT + GCN(SapBERT) | SapBERT top-k | InfoNCE | 0.3165 ± 0.0035 | 0.6002 ± 0.0015 | 0.4144 ± 0.0024 | 0.4218 ± 0.0020 | 0.3335 ± 0.0023 | 0.3725 ± 0.0022 | 2593.7 ± 5.5 | 1094.0 ± 7.5 |
| R2 SapBERT + GCN(DeepWalk) | SapBERT top-k | InfoNCE | 0.3270 ± 0.0061 | 0.6273 ± 0.0092 | 0.4295 ± 0.0058 | 0.4499 ± 0.0072 | 0.3517 ± 0.0065 | 0.3948 ± 0.0069 | 2564.3 ± 7.5 | 1153.7 ± 21.4 |
| R3 SapBERT + both | SapBERT top-k | InfoNCE | 0.3109 ± 0.0076 | 0.6033 ± 0.0078 | 0.4109 ± 0.0079 | 0.4154 ± 0.0126 | 0.3294 ± 0.0095 | 0.3674 ± 0.0108 | 2601.0 ± 7.9 | 1080.3 ± 31.0 |

## 4. Reading

- A trained residual under SapBERT top-k pairs and InfoNCE ends up below frozen SapBERT, in line with 9.3: the objective rewards reproducing SapBERT's neighbourhoods, and the extra capacity adds noise. No variation of this objective improved on U0, so the target gain was not found here.
- The fusion result shows the structure signal exists in DW2-I but is useful only as a bounded, untrained score term (w ≤ 0.25); at w ≥ 0.5 it hurts. This is the same pattern as A1: a bounded graph term helps, a learned replacement does not.
- Fusion is not an unsupervised-protocol violation (no gold in w's grid beyond reporting it), but the best w is read off test; the grid is reported in full and w was not used to select anything further. A held-out, gold-free way to choose w is not provided.

## 5. Checks (gate 9.5)

- `graph_branch` off reproduces the earlier residual path (unit test: identical output with and without the explicit flag).
- R0 equals the 9.3 U0 (MRR 0.4475, F1 0.4136) and A1 equals the Sprint 5/8 reference (MRR 0.4922, F1 0.4601).
- Beating frozen SapBERT by more than the seed std: no trained row.

## 6. Disclosures and caveats

- Gold test pairs were read by the two `evaluate` runs only; nothing in training, pairing or the fusion weights read them, but the fusion grid is evaluated on test (all four weights reported).
- The pseudo-pairs come from SapBERT (UMLS-pretrained), and pairs include test-split nodes (transductive), as in 9.2–9.4.
- DeepWalk is not comparable across graphs; its branch can only help through the pairs.

## 7. Commands

```
uv run cga run evaluate --dataset ncit_doid_unsup --experiment ncit_doid_unsup_fusion_test
PREFIX=res- SAMPLER_PREFIX=sap- bash scripts/run_unsup_matrix.sh ncit_doid_unsup 6
uv run cga run evaluate --dataset ncit_doid_unsup --experiment ncit_doid_unsup_residual_test
```
