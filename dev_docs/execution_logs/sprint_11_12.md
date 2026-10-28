# Sprint 11.12: Ablation of the mpnet-chunk InfoNCE dual (single-branch encoders)

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [11.11](sprint_11_11.md) (the dual GCNs on chunked mpnet vectors).

## 0. Outcome in brief

- **The DeepWalk branch does not add to the 11.11 gain.** The text-only GCN (A1) is above the full dual (A0) in 5 or 6 of 6 ranking cells in every source and in 15 to 17 of 24 set cells (section 5). Union, full view: Recall@10 0.2049 against 0.1630, F1 0.0286 against 0.0208; pair Hits@10 is level (0.1440 against 0.1472). The one place A0 is higher is the pool view F1 (0.0507 against 0.0423; 18 against 15 correct pairs).
- **The gain does not need the graph at all.** The text-only model with no graph convolution (A3, one linear map of the mpnet vector, trained with the same InfoNCE pseudo-pair loss) is already above frozen mpnet-chunk in 6 of 6 ranking cells and 22 to 23 of 24 set cells in every source. Union Hits@10 0.1248 against 0.1056, Recall@10 0.1599 against 0.1471, F1 0.0311 against 0.0164 (12 against 6 correct pairs). The graph convolution on top of it (A1 against A3) adds recall at k >= 10 (union Recall@10 +0.045) and little at the top of the ranking (union MRR +0.0002).
- **The graph-only encoders (A2, A4) work, but only through the mpnet pseudo-pairs.** Both are above random in every cell of every source and far above frozen DeepWalk (union MRR 0.0417 and 0.0389 against 0.0105; Recall@10 0.1134 and 0.0758 against 0.0092). They stay below frozen mpnet-chunk (union MRR 0.0606), and they never see text at the input. What they learn is the text-derived pairing of nodes, written into the DeepWalk space.
- **Graph convolution helps both branches at k >= 10** (A1 against A3, A2 against A4: Recall@10 above in all four sources) but not at the top of the ranking (full-view MRR differences between -0.013 and +0.009).
- **Checks passed:** the A0 retrain with the new code is bit-identical to the 11.11 model; A1 retrain is bit-identical; the A0, frozen mpnet-chunk and random rows reproduce the 11.11 numbers exactly (360 rows per source, max difference 0.0); repeat union evaluation identical (720 of 720 rows); `n_queries` is equal across the 10 methods in every evaluation.
- **Absolute level is still low** (best union Precision@10 0.046 against the ceiling 0.323); every statement above is from one seed.

## 1. Code change (minimal, default unchanged)

`DualEncoder` ([models.py](../../src/cga/pipelines/gcl_train/models.py)) gains `branch` (`both` | `text` | `graph`) and `message_passing` (bool); `build_dual_encoder` and the `gcn` path of `gcl_train` ([pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py)) read them with `model_cfg.get(...)`, defaults `both` and `true`. Unused towers, skips and fusion inputs are not created, and only the used towers are constructed, so an ablation does not consume random numbers for dropped modules. For the default, module registration order, parameter shapes and construction order are the original ones. Proof: the default still has 328,576 parameters, the default and the explicit `both` / `true` models have identical `state_dict` keys and values for the same seed (unit test), and the A0 retrain reproduces the 11.11 embeddings bit for bit (section 3).

**Manifest:** unchanged. The branch options are recorded in `config.resolved.yaml` (already an output of every run) and are visible in the model id. I did not add manifest fields; the A0 model, trained before the options existed, simply has no `branch` key in its config.

**Tests:** `tests/unit/pipelines/gcl_train/test_models.py` gained 8 cases: default unchanged (parameter count and state dict), shapes and parameter counts of the four new variants, `branch: text` independent of the DeepWalk input and `branch: graph` independent of the text input, `message_passing: false` independent of `edge_index` (and the GCN variant not), unknown branch rejected. `gcl_train` tests: 87 pass.

## 2. Runs

All use the 11.11 pseudo-pairs `s11c-rank_topk-k5/20261027T205523Z-ac890ca3` (manifest: mpnet-chunk `…20261027T194928Z-8cf38e55`, DeepWalk `…20261027T135210Z-3b18ba0c`, union ingest `20261027T194643Z-5d3f8baf`, 2,927 nodes, 14,635 examples), InfoNCE (temperature 0.1), `alpha 0.5`, hidden / out 128, Adam 1e-3, 100 epochs, seed 42, CPU, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

| Row | Model id | Run | Trainable params | `loss` epoch 1 → 100 | `loss_text` 1 → 100 | `loss_graph` 1 → 100 |
|---|---|---|---|---|---|---|
| A0 full dual (11.11) | `ent-dual-gcn-infonce-mpnetc-s42` | `20261027T205648Z-2aeb3259` | 328,576 | 8.855 → 5.800 | 10.685 → 5.191 | 7.025 → 6.408 |
| A0 retrain (gate) | `ent-abl-full-gcn-nce-mpnetc-s42` | `20261027T232219Z-0bded1cb` | 328,576 | same | same | same |
| A1 text-only GCN | `ent-abl-text-gcn-nce-mpnetc-s42` | `20261027T232407Z-adc619c8` (rerun `…233417Z-adc619c8`) | 246,272 | 8.849 → 6.108 | 8.932 → 5.051 | 8.766 → 7.165 |
| A2 graph-only GCN | `ent-abl-graph-gcn-nce-mpnetc-s42` | `20261027T232541Z-d91ebd8e` | 82,432 | 8.773 → 6.595 | 11.090 → 7.373 | 6.457 → 5.817 |
| A3 text-only, no GCN | `ent-abl-text-lin-nce-mpnetc-s42` | `20261027T232717Z-166677e3` | 114,944 | 8.725 → 6.505 | 8.612 → 4.923 | 8.838 → 8.086 |
| A4 graph-only, no GCN | `ent-abl-graph-lin-nce-mpnetc-s42` | `20261027T233035Z-b8df0fe0` | 33,024 | 8.656 → 7.023 | 10.825 → 8.267 | 6.488 → 5.779 |

Each manifest was checked: 100 `metrics.jsonl` rows, loss lower at epoch 100, 2,927 embedding rows, training data run as pinned, `branch` and `message_passing` in the resolved config as set. Parameter counts equal the ones asserted in the unit tests.

The single loss mixes a text-side and a graph-side term (`alpha 0.5`), both computed on the one output embedding. The single-branch runs still train on both terms. The columns show who fits what: the text models fit the text terms best (A3 `loss_text` 4.92) and the graph models fit the graph terms best (A4 `loss_graph` 5.78), and each is worse on the other side.

## 3. Reproduction gate and determinism

- **Gate:** A0 retrained with the new code and the unchanged config (only the model id differs) has node embeddings equal to the 11.11 InfoNCE model (`np.array_equal` on the 2,927 x 128 matrix: True). Only then were A1–A4 trained.
- **Determinism:** A1 trained twice with the same seed gives identical embeddings (True).

## 4. Evaluation

`configs/experiments/attack_cwe_ent_mpnet_ablation_eval.yaml`: methods `ent-mpnet-chunk` (frozen), `ent-dw` (frozen DeepWalk, `20261027T135210Z-3b18ba0c`), `ent-a1-text`, `ent-a2-graph`, `ent-a3-text-lin`, `ent-a4-graph-lin`, `ent-a0-full` (the 11.11 model) and `ent-rand-s42/43/44`; the six evaluations of 11.11 (cosine only; CSLS is not applicable).

| Source | 11.10 ingest | Evaluate run |
|---|---|---|
| stix | `20261027T194605Z-44eeb7c4` | `20261027T233834Z-2ab57969` |
| capec | `20261027T194618Z-cba9bfdb` | `20261027T233858Z-06e3251d` |
| intersection | `20261027T194631Z-950ba779` | `20261027T233925Z-e5ac91c7` |
| union | `20261027T194643Z-5d3f8baf` | `20261027T233942Z-ebf4efff` (repeat `20261027T234010Z-ebf4efff`) |

Tables are in `artifacts/evaluations/attack_cwe_ent/20261027T233942Z-ebf4efff/`: `sources_summary_ablation_{full,pool,set_full,set_pool}.md`.

**Checks:** all four pass.
- A0, frozen mpnet-chunk and the three random rows equal the 11.11 numbers: 360 values per source, max difference 0.0 (stix, capec, intersection, union).
- Repeat of the union run: 720 of 720 rows identical.
- `n_queries` is equal across the 10 methods in every evaluation (stix 284 / 287 / 93, capec 566 / 566 / 174, intersection 221 / 221 / 78, union 625 / 628 / 183 for ranking / assignment / set). Frozen DeepWalk covers the 2,927-node universe, so it is on the same candidate set (no separate run needed).

## 5. Results (union gold; other sources in the tables)

Full view (all 944 candidate CWEs). **Two different Hits@10 appear below.** *Pair Hits@10* is the `gold_alignment` metric: each (technique, CWE) gold pair is one query, with the sibling filter. *Set Hits@10* is the `gold_set_retrieval` metric: each technique is one query and counts as a hit if any of its gold CWEs is in the top 10. Set Hits@10 is the one in the 11.11 set tables (union InfoNCE 0.3443 full, 0.5956 pool); Pair Hits@10 is the one in the 11.11 ranking tables. Recall@10 and Precision@10 are set metrics.

| Row | Hits@1 | Pair Hits@10 | MRR | P | R | F1 | #Pred / #Correct | Set Hits@10 | Set Recall@10 | Set Prec@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | 0.0288 | 0.1056 | 0.0606 | 0.0583 | 0.0096 | 0.0164 | 103 / 6 | 0.2951 | 0.1471 | 0.0355 |
| DeepWalk (frozen) | 0.0032 | 0.0128 | 0.0105 | 0.0000 | 0.0000 | 0.0000 | 176 / 0 | 0.0383 | 0.0092 | 0.0044 |
| A1 text-only GCN | 0.0288 | 0.1440 | 0.0682 | 0.0775 | 0.0175 | 0.0286 | 142 / 11 | **0.3661** | **0.2049** | **0.0464** |
| A2 graph-only GCN | 0.0112 | 0.0976 | 0.0417 | 0.0301 | 0.0064 | 0.0105 | 133 / 4 | 0.2350 | 0.1134 | 0.0317 |
| A3 text-only, no GCN | **0.0352** | 0.1248 | 0.0680 | 0.0839 | 0.0191 | **0.0311** | 143 / 12 | 0.3279 | 0.1599 | 0.0410 |
| A4 graph-only, no GCN | 0.0160 | 0.0624 | 0.0389 | 0.0345 | 0.0080 | 0.0129 | 145 / 5 | 0.1803 | 0.0758 | 0.0213 |
| A0 full dual | 0.0208 | **0.1472** | 0.0640 | 0.0571 | 0.0127 | 0.0208 | 140 / 8 | 0.3443 | 0.1630 | 0.0459 |
| Random (3 seeds) | 0.0000 | 0.0101 | 0.0070 | 0.0194 | 0.0042 | 0.0070 | 186 / 0 | 0.0346 | 0.0128 | 0.0035 |

Pool view (candidate pool built from gold, optimistic; same columns):

| Row | Hits@1 | Pair Hits@10 | MRR | P | R | F1 | #Pred / #Correct | Set Hits@10 | Set Recall@10 | Set Prec@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | 0.0544 | 0.2496 | 0.1201 | 0.1538 | 0.0159 | 0.0289 | 65 / 10 | 0.5191 | 0.2905 | 0.0749 |
| DeepWalk (frozen) | 0.0176 | 0.0832 | 0.0494 | 0.0189 | 0.0032 | 0.0054 | 106 / 2 | 0.1639 | 0.0538 | 0.0257 |
| A1 text-only GCN | **0.0800** | **0.3104** | **0.1533** | 0.1829 | 0.0239 | 0.0423 | 82 / 15 | **0.6284** | **0.4003** | **0.0962** |
| A2 graph-only GCN | 0.0368 | 0.1984 | 0.0927 | 0.0649 | 0.0080 | 0.0142 | 77 / 5 | 0.4098 | 0.2441 | 0.0579 |
| A3 text-only, no GCN | 0.0640 | 0.2864 | 0.1367 | 0.1807 | 0.0239 | 0.0422 | 83 / 15 | 0.5410 | 0.3329 | 0.0869 |
| A4 graph-only, no GCN | 0.0256 | 0.1648 | 0.0806 | 0.0357 | 0.0048 | 0.0084 | 84 / 3 | 0.3825 | 0.2016 | 0.0552 |
| A0 full dual | 0.0704 | 0.3024 | 0.1490 | **0.2195** | **0.0287** | **0.0507** | 82 / 18 | 0.5956 | 0.3620 | 0.0951 |
| Random (3 seeds) | 0.0048 | 0.0699 | 0.0378 | 0.0194 | 0.0042 | 0.0070 | 137 / 2.7 | 0.1967 | 0.0707 | 0.0226 |

Gold count is 628 for every row; the recall ceiling is 0.2962.

**Set-based (per technique), k = 10.** One query per technique: Hits@10 = share of techniques with at least one gold CWE in the top 10, Recall@10 = mean share of a technique's gold CWEs found in the top 10, Precision@10 = mean share of the top 10 that is gold. The left three columns are the full view (all 944 candidate CWEs), the right three the pool view (pool built from gold, optimistic). The precision ceiling is the best Precision@10 a perfect ranking could reach (min(number of gold CWEs, 10) / 10, averaged over techniques); it is the same in both views. Best value per column among the seven non-random rows is in bold; random is the mean of three seeds. Union first, then the three other sources.

**union** (183 techniques scored)

| Row | Hits@10 | Recall@10 | Precision@10 | Hits@10 pool | Recall@10 pool | Precision@10 pool |
|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | 0.2951 | 0.1471 | 0.0355 | 0.5191 | 0.2905 | 0.0749 |
| DeepWalk (frozen) | 0.0383 | 0.0092 | 0.0044 | 0.1639 | 0.0538 | 0.0257 |
| A1 text-only GCN | **0.3661** | **0.2049** | **0.0464** | **0.6284** | **0.4003** | **0.0962** |
| A2 graph-only GCN | 0.2350 | 0.1134 | 0.0317 | 0.4098 | 0.2441 | 0.0579 |
| A3 text-only, no GCN | 0.3279 | 0.1599 | 0.0410 | 0.5410 | 0.3329 | 0.0869 |
| A4 graph-only, no GCN | 0.1803 | 0.0758 | 0.0213 | 0.3825 | 0.2016 | 0.0552 |
| A0 full dual | 0.3443 | 0.1630 | 0.0459 | 0.5956 | 0.3620 | 0.0951 |
| Random (3 seeds, mean) | 0.0346 | 0.0128 | 0.0035 | 0.1967 | 0.0707 | 0.0226 |
| *precision ceiling at k = 10* | – | – | 0.3230 | – | – | 0.3230 |

**stix** (93 techniques scored)

| Row | Hits@10 | Recall@10 | Precision@10 | Hits@10 pool | Recall@10 pool | Precision@10 pool |
|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | 0.2688 | 0.1241 | 0.0333 | 0.4839 | 0.2652 | 0.0688 |
| DeepWalk (frozen) | 0.0000 | 0.0000 | 0.0000 | 0.0968 | 0.0322 | 0.0108 |
| A1 text-only GCN | **0.4086** | **0.2463** | **0.0516** | **0.6452** | **0.4360** | 0.1011 |
| A2 graph-only GCN | 0.2258 | 0.1267 | 0.0323 | 0.3978 | 0.2600 | 0.0591 |
| A3 text-only, no GCN | 0.2796 | 0.1435 | 0.0355 | 0.4839 | 0.3076 | 0.0796 |
| A4 graph-only, no GCN | 0.1828 | 0.0717 | 0.0247 | 0.3978 | 0.2233 | 0.0634 |
| A0 full dual | 0.3548 | 0.1745 | 0.0473 | 0.6237 | 0.4322 | **0.1032** |
| Random (3 seeds, mean) | 0.0251 | 0.0071 | 0.0025 | 0.2007 | 0.0812 | 0.0229 |
| *precision ceiling at k = 10* | – | – | 0.3054 | – | – | 0.3054 |

**capec** (174 techniques scored)

| Row | Hits@10 | Recall@10 | Precision@10 | Hits@10 pool | Recall@10 pool | Precision@10 pool |
|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | 0.2989 | 0.1521 | 0.0351 | 0.4828 | 0.2826 | 0.0707 |
| DeepWalk (frozen) | 0.0402 | 0.0096 | 0.0046 | 0.1609 | 0.0539 | 0.0259 |
| A1 text-only GCN | **0.3736** | **0.2136** | **0.0454** | **0.6264** | **0.4105** | **0.0925** |
| A2 graph-only GCN | 0.2299 | 0.1152 | 0.0310 | 0.4080 | 0.2497 | 0.0563 |
| A3 text-only, no GCN | 0.3276 | 0.1640 | 0.0402 | 0.5402 | 0.3379 | 0.0851 |
| A4 graph-only, no GCN | 0.1667 | 0.0689 | 0.0195 | 0.3736 | 0.1991 | 0.0529 |
| A0 full dual | 0.3391 | 0.1682 | 0.0431 | 0.5805 | 0.3630 | 0.0897 |
| Random (3 seeds, mean) | 0.0326 | 0.0131 | 0.0033 | 0.1839 | 0.0686 | 0.0213 |
| *precision ceiling at k = 10* | – | – | 0.3057 | – | – | 0.3057 |

**intersection** (78 techniques scored)

| Row | Hits@10 | Recall@10 | Precision@10 | Hits@10 pool | Recall@10 pool | Precision@10 pool |
|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | 0.2949 | 0.1416 | 0.0346 | 0.4359 | 0.2517 | 0.0641 |
| DeepWalk (frozen) | 0.0000 | 0.0000 | 0.0000 | 0.0897 | 0.0302 | 0.0103 |
| A1 text-only GCN | **0.4487** | **0.2881** | **0.0526** | **0.6667** | **0.4753** | **0.0987** |
| A2 graph-only GCN | 0.2308 | 0.1287 | 0.0333 | 0.4231 | 0.2804 | 0.0603 |
| A3 text-only, no GCN | 0.2821 | 0.1564 | 0.0346 | 0.4872 | 0.3083 | 0.0782 |
| A4 graph-only, no GCN | 0.1667 | 0.0610 | 0.0231 | 0.4103 | 0.2410 | 0.0641 |
| A0 full dual | 0.3590 | 0.1989 | 0.0436 | 0.6282 | 0.4610 | **0.0987** |
| Random (3 seeds, mean) | 0.0214 | 0.0076 | 0.0021 | 0.1838 | 0.0828 | 0.0218 |
| *precision ceiling at k = 10* | – | – | 0.2833 | – | – | 0.2833 |

**Difference columns, union** (positive = first minus second; full-view MRR, pair Hits@10, set Recall@10, F1; pool-view MRR, pair Hits@10, set Recall@10, F1):

| Comparison | MRR | Hits@10 | Rec@10 | F1 | MRR pool | Hits@10 pool | Rec@10 pool | F1 pool |
|---|---|---|---|---|---|---|---|---|
| Branch effect A0 − A1 (add DeepWalk to text) | −0.0042 | +0.0032 | −0.0419 | −0.0077 | −0.0043 | −0.0080 | −0.0383 | +0.0085 |
| Branch effect A0 − A2 (add text to graph) | +0.0224 | +0.0496 | +0.0497 | +0.0103 | +0.0563 | +0.1040 | +0.1179 | +0.0365 |
| GCN effect, text: A1 − A3 | +0.0002 | +0.0192 | +0.0450 | −0.0026 | +0.0166 | +0.0240 | +0.0674 | +0.0001 |
| GCN effect, graph: A2 − A4 | +0.0027 | +0.0352 | +0.0376 | −0.0024 | +0.0121 | +0.0336 | +0.0426 | +0.0058 |
| Training gain, text: A3 − frozen | +0.0074 | +0.0192 | +0.0128 | +0.0147 | +0.0166 | +0.0368 | +0.0424 | +0.0133 |
| Training gain, text: A1 − frozen | +0.0077 | +0.0384 | +0.0578 | +0.0122 | +0.0332 | +0.0608 | +0.1098 | +0.0134 |
| Graph gain: A2 − frozen DeepWalk | +0.0312 | +0.0848 | +0.1042 | +0.0105 | +0.0432 | +0.1152 | +0.1903 | +0.0087 |
| Graph gain: A4 − frozen DeepWalk | +0.0285 | +0.0496 | +0.0666 | +0.0129 | +0.0311 | +0.0816 | +0.1478 | +0.0030 |

The same differences for the other three sources were computed with a scratch script outside the repo (section 8). The signs of the branch and training-gain rows agree in every source; the GCN-effect rows on MRR and F1 do not (full-view MRR A2 − A4 is −0.0127 in stix and −0.0073 in intersection).

**Cell counts** (ranking cells: Hits@1, Hits@10, MRR in the full and pool views = 6; set cells: Hits, Recall, Precision at k = 5, 10, 15, 20 in both views = 24; "above" counts strictly higher, in the order stix, capec, intersection, union):

| Statement | Ranking cells above | Set cells above |
|---|---|---|
| A1 above A0 | 6, 5, 6, 5 | 15, 17 (2 tied), 16, 17 |
| A0 above A1 | 0, 1, 0, 1 | 9, 5, 8, 7 |
| A3 above frozen mpnet-chunk | 6, 6, 6, 6 | 23, 22, 22, 23 |
| A1 above frozen mpnet-chunk | 5, 5, 5, 5 (1 tied each) | 24, 23, 24, 21 |
| A0 above frozen mpnet-chunk | 5, 5, 5, 5 | 23, 21, 21, 21 |
| A1 above A3 | 5, 5, 5, 5 | 23, 20, 24, 20 |
| A2 above A4 | 4, 5, 4, 5 | 18, 24, 22, 24 |
| A2 above frozen DeepWalk; A4 above frozen DeepWalk | 6 in every source (both) | 24 in every source (both) |
| A2 above random; A4 above random | 6 in every source (both) | 24 in every source (both) |
| A0 above A2; A1 above A2 | 6 in every source (both) | 24 in every source (both) |

## 6. Report (no tuning)

1. **Does the text-only GCN (A1) match the full dual (A0)?** Yes, and slightly better. A1 is above A0 in 5 or 6 of 6 ranking cells in each source and in 15 to 17 of 24 set cells. The 11.11 InfoNCE gain over frozen mpnet therefore does not come from DeepWalk. The only metrics where A0 is clearly higher are the pool-view F1 (union +0.0085, 3 more correct pairs) and, by a tiny margin, union Hits@10 (+0.0032); A1 is higher for Recall@10 in both views and for full-view F1 and MRR. With one seed, differences of this size (a few hundredths) are not established.
2. **Can the graph-only encoder carry alignment? (A2 / A4)** Yes, through the pseudo-pairs. Both are above random in all 6 ranking and 24 set cells of every source, and above frozen DeepWalk in all of them. Union MRR 0.0417 (A2, 5.9x random) and 0.0389 (A4) against 0.0105 for frozen DeepWalk (1.5x). The two graphs are disjoint, their DeepWalk vectors are not in a shared space, and frozen DeepWalk is close to random across graphs; the trained map is what creates a shared space. This is distillation of the mpnet text signal into structural coordinates, not independent structural evidence. A2 and A4 stay below frozen mpnet-chunk in MRR and Recall@10 and below A0 and A1 in every cell. They are not a constant list: the top-10 lists of A2 and A4 contain 313 and 359 distinct CWEs and reach 25 and 19 gold CWEs (section 7). I did not test which structural cue (degree, neighbourhood, position) the map uses.
3. **Does the graph convolution help? (A1 against A3, A2 against A4)** Mostly at k >= 10. A1 is above A3 in 5 of 6 ranking and 20 to 24 of 24 set cells per source (union Recall@10 +0.045, Hits@10 +0.019), but union MRR is level (+0.0002) and F1 slightly lower (−0.0026). A2 is above A4 in 4 or 5 of 6 ranking and 18 to 24 of 24 set cells; MRR is mixed in sign across sources.
4. **Does the 11.11 gain need the graph at all?** No. A3 is a linear map of the mpnet vector with no graph input, and it is above frozen mpnet-chunk in every ranking cell in all four sources and 22 to 23 of 24 set cells. So the training on the mpnet pseudo-pairs with the InfoNCE loss is the main source of the 11.11 gain; the graph convolution adds recall at larger k (A1 against A3) and DeepWalk adds nothing measurable (A0 against A1). In particular the 11.11 statement "the dual beats frozen text" should be read as "the InfoNCE-trained projection beats frozen text".
5. **Top-rank behaviour:** Hits@1 is not improved by training in the full view for A0 and A1 (0.0208 and 0.0288 against 0.0288); only A3 reaches 0.0352. The gains are at k >= 5 and in recall.

## 7. Hub check (union gold, 183 techniques with a gold CWE in the shared pool of 944 CWEs; cosine; hardware = CWE-1200 to 1399, 13.0% of the pool)

| Row | Set Recall@10 (check) | Distinct CWEs in top-10 lists | Share of slots held by the 10 most frequent CWEs | Hardware share of slots | Gold CWEs reached | Five most frequent CWEs (count) |
|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | 0.1471 | 277 | 26.4% | 22.8% | 29 | 509 (63), 1278 (54), 537 (54), 114 (53), 1334 (53) |
| DeepWalk (frozen) | 0.0092 | 598 | 6.8% | 10.2% | 8 | 560 (15), 33 (14), 509 (14), 28 (12), 29 (12) |
| A0 full dual | 0.1630 | 377 | 15.4% | 20.5% | 40 | 114 (39), 1273 (29), 520 (28), 379 (28), 522 (28) |
| A1 text GCN | 0.2049 | 388 | 15.1% | 19.9% | 38 | 114 (39), 1278 (31), 1391 (29), 1334 (27), 15 (27) |
| A2 graph GCN | 0.1134 | 313 | 18.5% | 19.7% | 25 | 114 (46), 15 (40), 73 (37), 384 (34), 1313 (33) |
| A3 text linear | 0.1599 | 383 | 15.1% | 18.4% | 40 | 114 (38), 1334 (31), 520 (30), 15 (27), 1278 (26) |
| A4 graph linear | 0.0758 | 359 | 17.0% | 22.6% | 19 | 114 (51), 507 (37), 494 (32), 509 (30), 1313 (29) |

The Recall@10 column equals the set table, which checks that this script scores the same lists. Training spreads the lists (distinct CWEs 277 → 377 to 388; top-10 share 26% → 15%) for every trained text model, and the text-only models reach as many gold CWEs as the full dual (38 and 40 against 40). CWE-114 is the most frequent CWE in every trained row (39 to 51 of 1,830 slots). The graph-only rows are more concentrated and reach fewer gold CWEs (25, 19). Hardware share stays near 20%, above the 13% of the pool.

## 8. Limits

- **Single seed and one pseudo-pair draw.** No seed variation was run; differences of about 0.01 (for example A0 against A1, A1 against A3 in MRR) are not tested against seed noise. Direction counts across four sources and two views are not independent tests: the sources share most gold pairs, and the pool view is built from gold.
- **The loss still contains both the text and the graph terms in all single-branch runs.** A model that sees only the text input is still trained with graph-neighbour targets, and the other way round.
- **The graph-only result depends on the mpnet pseudo-pairs.** It says nothing about what DeepWalk alone could align without them.
- 3 techniques (stix, union) lie outside the 2,927-node universe and are not scored for any method, as in 11.11.
- Gold is transitive through CAPEC; the pool is optimistic; assignment F1 rests on 3 to 18 correct pairs; absolute precision is low (best union Precision@10 0.046, ceiling 0.323).
- `checks_eval1112.py`, `hub1112.py` and `check1112.py` are scratch analysis scripts outside the repo; the numbers in sections 3, 5 and 7 come from them.

## 9. Files

- Modified: `src/cga/pipelines/gcl_train/models.py`, `src/cga/pipelines/gcl_train/pipeline.py`, `tests/unit/pipelines/gcl_train/test_models.py`, `scripts/unsup_summary.py` (row set `attack_ent_mpnet_ablation`), `dev_docs/sprints/sprint_11.md`.
- New: `configs/experiments/attack_cwe_ent_mpnet_ablation_eval.yaml`, `dev_docs/execution_logs/sprint_11_12.md`.

## 10. Commands

```
uv run pytest tests/unit/pipelines/gcl_train -q

RUN=20261027T205523Z-ac890ca3
# gate (A0 retrain; branch=both, message_passing=true) and A1..A4:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
  --set seed=42 --set device=cpu --set gcl_train.loss.name=infonce \
  --set gcl_train.model.model_id=<ent-abl-{full,text,graph}-{gcn,lin}-nce-mpnetc-s42> \
  --set gcl_train.model.branch=<both|text|graph> --set gcl_train.model.message_passing=<true|false> \
  --set gcl_train.inputs.training_data_sampler_id=s11c-rank_topk-k5 --set gcl_train.inputs.training_data_run=$RUN

for each gold source (11.10 ingest run from section 4):
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_mpnet_ablation_eval \
    --set evaluate.inputs.ingest_run=<ingest run of that source>
U=20261027T233942Z-ebf4efff
SRC="--source-run stix=20261027T233834Z-2ab57969 --source-run capec=20261027T233858Z-06e3251d \
     --source-run intersection=20261027T233925Z-e5ac91c7 --source-run union=$U"
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_mpnet_ablation \
  --ranking-eval gold_<full|pool> --prf-eval prf_<full|pool> --lift --out-name sources_summary_ablation_<full|pool>.md $SRC
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_mpnet_ablation \
  --set-eval set_<full|pool> --out-name sources_summary_ablation_set_<full|pool>.md $SRC
uv run pytest tests/unit -q
```
