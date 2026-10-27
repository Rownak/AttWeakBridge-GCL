# Sprint 11: Dual-encoder GCN on Enterprise ATT&CK × CWE (training only)

Plan: [sprint_11.md](../sprints/sprint_11.md). No code changes; `evaluate` not run.

## 0. Outcome in brief

- 11.1 gate passed; 11.2 gate passed (both runs, 100 epochs, loss down, 2,927 embeddings).
- **Determinism check failed:** rerunning triplet with seed 42 gave different node embeddings (see §4). Not patched, per sprint scope.

## 1. Phase 11.1: Pseudo-pairs

- Inputs used unchanged: ingest `20261026T224948Z-0f7e98dc`, SecRoBERTa store `20261027T010723Z-8bc08adc`, DeepWalk store `20261026T225954Z-655383ae`.
- Run `artifacts/training_data/attack_cwe_ent/s11-rank_topk-k5/20261027T025748Z-86bccf61` (seed 42, `rank_topk`, k=5, `held_out_policy: none`).
- Universe 2,927 nodes, 0 held out; 14,635 examples (all `rank_topk`, all split `train`). All anchor/pos/neg ids resolve in the universe.
- `uv run pytest tests/unit -q`: 574 passed.

## 2. Phase 11.2: Training (seed 42, `unsup_u2`, `encoder: gcn`, 328,576 trainable params)

| Run | Run id | Loss epoch 1 | Loss epoch 100 | Output |
|---|---|---|---|---|
| `ent-dual-gcn-triplet-s42` | `20261027T025828Z-18d30b07` | 0.7906 | 0.1267 | `models/gcl/attack_cwe_ent/ent-dual-gcn-triplet-s42/20261027T025828Z-18d30b07/` |
| `ent-dual-gcn-infonce-s42` | `20261027T025838Z-048e799f` | 8.8463 | 5.0546 | `models/gcl/attack_cwe_ent/ent-dual-gcn-infonce-s42/20261027T025838Z-048e799f/` |

- Both: 100 `metrics.jsonl` rows, `node_embeddings` (2,927 x 128), `training_data_run_id` = `20261027T025748Z-86bccf61`, `universe_sha256` = `b11f6656e300…`.
- Each run took about 2-5 s.

## 3. Commands

```
uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
  --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-pt_SecRoBERTa \
  --set training_data.sampler.sampler_id=s11-rank_topk-k5
RUN=20261027T025748Z-86bccf61
uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
  --set gcl_train.model.model_id=ent-dual-gcn-<loss>-s42 \
  --set gcl_train.inputs.training_data_sampler_id=s11-rank_topk-k5 \
  --set gcl_train.inputs.training_data_run=$RUN --set gcl_train.loss.name=<triplet|infonce>
```

## 4. Open issue: training is not reproducible

- Rerun of triplet (`20261027T025903Z-18d30b07`, extra run directory kept) vs the first run: same index, but vectors differ (mean abs diff 0.35, max 4.01, per-row cosine mean 0.02). Epoch-1 loss already differs (0.7906 vs 0.8031), so the difference is at initialisation or batching, not a late drift.
- Likely cause: `cga.core.seeds.seed_everything` is defined but never called anywhere in `src/`, so the torch RNG is unseeded. The `seed` value is only recorded.
- Not fixed here (sprint says no code changes). Until fixed, the two models are one random draw each and reruns will not reproduce them. Suggested follow-up: call `seed_everything(ctx.seed)` in the `gcl_train` run start, then repeat the determinism check.
- The 11.2 checkbox for determinism stays open.

---

# Phases 11.4-11.5: Evaluation on the gold sets

No pipeline code changed. `scripts/unsup_summary.py` gained the row set `attack_ent_dual` and passes the selected row set to `--lift`.

## 5. What was done

- Config `configs/experiments/attack_cwe_ent_dual_test.yaml`: `ent-d-trip`, `ent-d-nce`, references `ent-s-r1`, `ent-s-r2-s42`, `ent-rand-s42/43/44`, and the four evaluations of Sprint 10.6 (`gold_full`, `prf_full`, `gold_pool`, `prf_pool`; same pool file).
- **`ent-d-trip` is pinned to its first run `20261027T025828Z-18d30b07`.** `LATEST` of that model id points to the determinism rerun (`…025903Z`), a different random draw (section 4). `ent-d-nce` has one run.

| Gold source | Ingest run | Evaluate run |
|---|---|---|
| stix | `20261026T224906Z-a0725a37` | `20261027T033353Z-b9cb22b0` |
| capec | `20261026T224923Z-1b4b5b3d` | `20261027T033403Z-38c6248c` |
| intersection | `20261026T224935Z-dee50c06` | `20261027T033413Z-35db93f8` |
| union | `20261026T224948Z-0f7e98dc` | `20261027T033422Z-7b5ea126` |

Repeat of union: `20261027T033432Z-7b5ea126`.

## 6. Checks

- **Reproduction:** for the reference rows (`ent-s-r1`, `ent-s-r2-s42`, `ent-rand-s42/43/44`; 120 rows per source) against the 10.6 runs: all rows matched on both sides, max absolute difference 0.0, `n_queries` equal, for stix, capec, intersection and union.
- **Repeat of union:** 168 of 168 rows identical (max difference 0.0). The evaluation is deterministic; the models are not (section 4).
- Unit tests: `uv run pytest tests/unit -q` green (see final count in the session).

## 7. Results (single seed for the dual rows; random = mean of 3)

MRR / F1 per row and gold source. Full per-source tables (Hits@1/10, P, R, #Pred, #Correct, #Gold) are in `artifacts/evaluations/attack_cwe_ent/20261027T033422Z-7b5ea126/sources_summary_dual_full.md` and `…_pool.md`.

**All 944 CWEs (`*_full`)**

| Row | stix | capec | intersection | union |
|---|---|---|---|---|
| Dual GCN, triplet | 0.0110 / 0.0056 | 0.0140 / 0.0029 | 0.0131 / 0.0071 | 0.0130 / 0.0027 |
| Dual GCN, InfoNCE | 0.0246 / 0.0056 | 0.0270 / 0.0120 | 0.0298 / 0.0142 | 0.0251 / 0.0055 |
| SecRoBERTa (frozen) | 0.0251 / 0.0115 | 0.0253 / 0.0092 | 0.0286 / 0.0147 | 0.0234 / 0.0083 |
| GCN(SecRoBERTa) residual | 0.0149 / 0.0106 | 0.0164 / 0.0083 | 0.0181 / 0.0134 | 0.0151 / 0.0075 |
| Random | 0.0065 / 0.0000 | 0.0070 / 0.0000 | 0.0065 / 0.0000 | 0.0070 / 0.0000 |

**151-CWE pool (`*_pool`)**

| Row | stix | capec | intersection | union |
|---|---|---|---|---|
| Dual GCN, triplet | 0.0460 / 0.0000 | 0.0634 / 0.0064 | 0.0530 / 0.0076 | 0.0591 / 0.0058 |
| Dual GCN, InfoNCE | 0.0627 / 0.0298 | 0.0645 / 0.0063 | 0.0696 / 0.0300 | 0.0618 / 0.0087 |
| SecRoBERTa (frozen) | 0.0705 / 0.0531 | 0.0790 / 0.0509 | 0.0805 / 0.0606 | 0.0734 / 0.0462 |
| GCN(SecRoBERTa) residual | 0.0364 / 0.0111 | 0.0486 / 0.0151 | 0.0413 / 0.0140 | 0.0456 / 0.0138 |
| Random | 0.0389 / 0.0053 | 0.0375 / 0.0076 | 0.0390 / 0.0068 | 0.0378 / 0.0070 |

**MRR lift over random (union):** full view: triplet 1.9x, InfoNCE 3.6x, frozen 3.3x, residual 2.2x. Pool view: triplet 1.6x, InfoNCE 1.6x, frozen 1.9x, residual 1.2x.

`#Correct` on the pool, union: triplet 2, InfoNCE 3, frozen 16, residual 5 (random 2.7) of 628 gold pairs.

## 8. Findings

- **Neither dual-encoder row clearly beats frozen SecRoBERTa.** On the pool it is lower on MRR and F1 everywhere; on the full view only InfoNCE edges ahead on MRR in two sources. The closest is InfoNCE on the full view (MRR 0.0298 vs 0.0286 on intersection, 0.0270 vs 0.0253 on capec: slightly higher, within what one seed can show; frozen is higher on stix and union, and on F1 in 3 of 4 sources).
- **InfoNCE is higher than triplet** on MRR in every source and view (about 2x on the full view) and on F1 in most. Triplet is near random on the full view (1.7-2.0x) and about 1.2-1.7x on the pool, where F1 is at or below random.
- **Dual vs the residual adapter:** the dual InfoNCE row is above the 10.5 residual adapter on MRR everywhere on the full view and on the pool; F1 is not clearly separable.
- On the pool, the dual rows are barely above random (1.2-1.8x); only frozen SecRoBERTa has an F1 clearly above random (0.046-0.061, 8-16 correct).

## 9. Limits

- Single seed, and the training is not reproducible (section 4): the dual rows are one random draw each; differences of a few thousandths of MRR should not be read.
- 3 gold techniques lie outside the training universe for stix and union (10.6 limit); the evaluation still ranks them.
- F1 rests on few correct pairs (1-5 for the dual rows).
- The pool is built from the union gold, so it is optimistic; read it next to the random rows.
- The summary tables show Hits@1 and Hits@10 only (script columns); Hits@5 is in `metrics.parquet`.
- Models were trained once, with no tuning.

## 10. Commands

```
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_dual_test \
  --set evaluate.inputs.ingest_run=<ingest run of that source>
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run 20261027T033422Z-7b5ea126 \
  --rows attack_ent_dual --ranking-eval gold_<full|pool> --prf-eval prf_<full|pool> --lift \
  --out-name sources_summary_dual_<full|pool>.md \
  --source-run stix=20261027T033353Z-b9cb22b0 --source-run capec=20261027T033403Z-38c6248c \
  --source-run intersection=20261027T033413Z-35db93f8 --source-run union=20261027T033422Z-7b5ea126
```
