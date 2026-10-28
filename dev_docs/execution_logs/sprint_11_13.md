# Sprint 11.13: Pseudo-pair `top_k` and loss ablation (pool view, set-based, k = 10)

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [11.12](sprint_11_12.md) (single-branch ablation), [11.11](sprint_11_11.md).

## 0. Outcome in brief

- **InfoNCE beats triplet everywhere.** For A0, A1 and A2, at both `top_k` values, the InfoNCE model is above its triplet counterpart in all five reported metrics (largest gap: Hits@10 and Recall@10, 0.12 to 0.20 lower with triplet). The triplet A0 and A1 are also below frozen mpnet-chunk in Hits@10 (0.40 to 0.44 against 0.5191).
- **`top_k` = 10 does not help the text-carrying models.** A0 and A1 with InfoNCE are lower at k = 10 than at k = 5 in all five metrics (Hits@10 −0.022 and −0.033, Recall@10 −0.008 and −0.009). Only the graph-only model A2 gains slightly at k = 10 (Hits@10 +0.011 with InfoNCE, +0.022 with triplet), while its Recall@10 is level.
- **The 11.12 ordering holds for InfoNCE:** A1 is above A0 in all five metrics at both k, and A0 is above A2 in all of them. With triplet the A1 against A0 comparison is mixed (A1 higher in Hits@1 and Hits@5, A0 higher in Recall@10 and Precision@10, Hits@10 equal at k = 5).
- **Best row:** A1, InfoNCE, k = 5 (Hits@10 0.6284, Recall@10 0.4003, Precision@10 0.0962), the same model as in 11.12.
- **Checks passed:** the five reused or frozen rows equal the 11.12 `set_pool` values exactly (60 values, max difference 0.0); the repeat evaluation is identical (168 of 168 rows); `n_queries` is 183 for all 14 rows.
- Single seed, one gold source, the optimistic pool view; see section 6.

## 1. Protocol

Union gold only (11.10 ingest `20261027T194643Z-5d3f8baf`); `gold_set_retrieval`, split `test`, `target_pool_union.json` (pool view only); one query per technique (183 scored); cosine; reported: Hits@1, Hits@5, Hits@10, Recall@10, Precision@10 only.

Fixed for every trained model: mpnet-chunk text store `20261027T194928Z-8cf38e55`, DeepWalk store `20261027T135210Z-3b18ba0c`, `unsup_u2` (hidden / out 128, dropout 0, Adam lr 1e-3, weight decay 0, 100 epochs), `alpha 0.5`, seed 42, CPU with one thread, `message_passing: true`. Varied: `top_k` (it applies to the positives and the bottom-`k` negatives together, on the text and the graph side), loss, `branch`. No code change in this phase.

## 2. Pseudo-pairs

| Sampler id | Run | Universe | Examples | Distinct cross-graph pairs | Of which gold (union) |
|---|---|---|---|---|---|
| `s11c-rank_topk-k5` (11.11) | `20261027T205523Z-ac890ca3` | 2,927 | 14,635 | 13,828 | 62 (0.45%) |
| `s11c-rank_topk-k10` (new) | `20261028T013100Z-7a027e89` | 2,927 | 29,270 | 27,191 | 84 (0.31%) |

The k = 10 manifest lists the pinned ingest `20261027T194643Z-5d3f8baf`, text `20261027T194928Z-8cf38e55` and DeepWalk `20261027T135210Z-3b18ba0c` runs; protocol `unsupervised`, `held_out_policy: none`, 0 held-out nodes, 10 examples per anchor, seed 42. Gold pairs are counted as unordered pairs (the k = 5 value reproduces 11.11). Going from k = 5 to 10 adds 22 gold pairs but doubles the number of pairs, so the gold share falls.

## 3. Runs (exact configuration)

Loss parameters: InfoNCE temperature 0.1, triplet margin 1.0; all with `alpha 0.5`. "Reused" rows are the 11.11 / 11.12 models with the identical configuration (their retrains are bit-identical).

| Row id | Model id | Run | Branch | `top_k` / sampler | Loss | Training data run | Params | Examples | `loss` epoch 1 → 100 | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| a0-nce-k5 | `ent-dual-gcn-infonce-mpnetc-s42` | `20261027T205648Z-2aeb3259` | both | 5 / `s11c-rank_topk-k5` | InfoNCE | `20261027T205523Z-ac890ca3` | 328,576 | 14,635 | 8.855 → 5.799 | reused |
| a0-trip-k5 | `ent-dual-gcn-triplet-mpnetc-s42` | `20261027T205615Z-db28e05d` | both | 5 | triplet | same | 328,576 | 14,635 | 0.788 → 0.140 | reused |
| a0-nce-k10 | `ent-p13-a0-nce-k10-mpnetc-s42` | `20261028T013227Z-4ae182b9` | both | 10 / `s11c-rank_topk-k10` | InfoNCE | `20261028T013100Z-7a027e89` | 328,576 | 29,270 | 9.687 → 7.260 | new |
| a0-trip-k10 | `ent-p13-a0-trip-k10-mpnetc-s42` | `20261028T013624Z-a42425b9` | both | 10 | triplet | same | 328,576 | 29,270 | 0.804 → 0.176 | new |
| a1-nce-k5 | `ent-abl-text-gcn-nce-mpnetc-s42` | `20261027T232407Z-adc619c8` | text | 5 | InfoNCE | `20261027T205523Z-ac890ca3` | 246,272 | 14,635 | 8.849 → 6.108 | reused |
| a1-trip-k5 | `ent-p13-a1-trip-k5-mpnetc-s42` | `20261028T013128Z-3ec036f3` | text | 5 | triplet | same | 246,272 | 14,635 | 0.866 → 0.169 | new |
| a1-nce-k10 | `ent-p13-a1-nce-k10-mpnetc-s42` | `20261028T013822Z-b0e71713` | text | 10 | InfoNCE | `20261028T013100Z-7a027e89` | 246,272 | 29,270 | 9.639 → 7.373 | new |
| a1-trip-k10 | `ent-p13-a1-trip-k10-mpnetc-s42` | `20261028T014718Z-9d963909` | text | 10 | triplet | same | 246,272 | 29,270 | 0.877 → 0.203 | new |
| a2-nce-k5 | `ent-abl-graph-gcn-nce-mpnetc-s42` | `20261027T232541Z-d91ebd8e` | graph | 5 | InfoNCE | `20261027T205523Z-ac890ca3` | 82,432 | 14,635 | 8.773 → 6.595 | reused |
| a2-trip-k5 | `ent-p13-a2-trip-k5-mpnetc-s42` | `20261028T013157Z-8fceb08f` | graph | 5 | triplet | same | 82,432 | 14,635 | 0.758 → 0.218 | new |
| a2-nce-k10 | `ent-p13-a2-nce-k10-mpnetc-s42` | `20261028T014901Z-9ca690b4` | graph | 10 | InfoNCE | `20261028T013100Z-7a027e89` | 82,432 | 29,270 | 9.630 → 7.799 | new |
| a2-trip-k10 | `ent-p13-a2-trip-k10-mpnetc-s42` | `20261028T015756Z-a2df5fbd` | graph | 10 | triplet | same | 82,432 | 29,270 | 0.777 → 0.252 | new |

Each of the 12 manifests and resolved configs was checked: 100 metric rows, loss lower at epoch 100, 2,927 x 128 finite embeddings, 2,927 nodes, and the sampler id, training data run, loss, branch and `message_passing: true` as in the table. Losses of different `top_k` or loss types are not comparable with each other (different example sets and scales).

## 4. Evaluation

`configs/experiments/attack_cwe_ent_p13_eval.yaml` (14 pinned rows, one evaluation `set_pool`, `ks: [1, 5, 10]`). Evaluate run `20261028T015956Z-17707104`; repeat `20261028T020013Z-17707104`.

**Checks:** (1) the frozen mpnet-chunk, frozen DeepWalk, A0, A1 and A2 (InfoNCE, k = 5) rows equal the 11.12 `set_pool` values at k = 1, 5, 10: 60 values, max difference 0.0. (2) The repeat run is identical in all 168 rows. (3) `n_queries` = 183 for all 14 methods.

## 5. Comparison table (union gold, pool view, per technique)

Hits@1 and Hits@5 use the cutoffs 1 and 5; Hits@10, Recall@10 and Precision@10 use cutoff 10. Best value per column in bold. Frozen rows have no training.

| Model | `top_k` | Loss | Hits@1 | Hits@5 | Hits@10 | Recall@10 | Precision@10 |
|---|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | – | – | 0.1530 | 0.3716 | 0.5191 | 0.2905 | 0.0749 |
| DeepWalk (frozen) | – | – | 0.0328 | 0.1093 | 0.1639 | 0.0538 | 0.0257 |
| A0 full dual | 5 | InfoNCE | 0.1749 | 0.4754 | 0.5956 | 0.3620 | 0.0951 |
| A0 full dual | 10 | InfoNCE | 0.1475 | 0.4208 | 0.5738 | 0.3541 | 0.0891 |
| A0 full dual | 5 | triplet | 0.0601 | 0.2568 | 0.4372 | 0.2376 | 0.0683 |
| A0 full dual | 10 | triplet | 0.0601 | 0.2459 | 0.4153 | 0.2204 | 0.0601 |
| A1 text-only GCN | 5 | InfoNCE | **0.1858** | **0.4809** | **0.6284** | **0.4003** | **0.0962** |
| A1 text-only GCN | 10 | InfoNCE | 0.1803 | 0.4481 | 0.5956 | 0.3910 | 0.0923 |
| A1 text-only GCN | 5 | triplet | 0.0765 | 0.2951 | 0.4372 | 0.2275 | 0.0656 |
| A1 text-only GCN | 10 | triplet | 0.0656 | 0.2787 | 0.3989 | 0.2175 | 0.0579 |
| A2 graph-only GCN | 5 | InfoNCE | 0.1038 | 0.2514 | 0.4098 | 0.2441 | 0.0579 |
| A2 graph-only GCN | 10 | InfoNCE | 0.0929 | 0.2842 | 0.4208 | 0.2440 | 0.0628 |
| A2 graph-only GCN | 5 | triplet | 0.0273 | 0.1366 | 0.2514 | 0.1043 | 0.0377 |
| A2 graph-only GCN | 10 | triplet | 0.0219 | 0.1475 | 0.2732 | 0.1104 | 0.0377 |

Hits@k is the share of the 183 techniques with at least one gold CWE in the top k; one technique is 0.0055.

**Differences** (Hits@10 / Recall@10 / Precision@10):

| Model, loss | `top_k` 10 minus 5 |
|---|---|
| A0, InfoNCE | −0.0219 / −0.0079 / −0.0060 |
| A0, triplet | −0.0219 / −0.0172 / −0.0082 |
| A1, InfoNCE | −0.0328 / −0.0093 / −0.0038 |
| A1, triplet | −0.0383 / −0.0100 / −0.0077 |
| A2, InfoNCE | +0.0109 / −0.0001 / +0.0049 |
| A2, triplet | +0.0219 / +0.0061 / +0.0000 |

| Model, `top_k` | triplet minus InfoNCE |
|---|---|
| A0, 5 | −0.1585 / −0.1244 / −0.0268 |
| A0, 10 | −0.1585 / −0.1337 / −0.0290 |
| A1, 5 | −0.1913 / −0.1728 / −0.0306 |
| A1, 10 | −0.1967 / −0.1734 / −0.0344 |
| A2, 5 | −0.1585 / −0.1398 / −0.0202 |
| A2, 10 | −0.1475 / −0.1336 / −0.0251 |

## 6. Report and limits

- **Loss:** InfoNCE is above triplet in all 30 cells (3 models, 2 `top_k` values, 5 metrics). With triplet, A0 and A1 fall below frozen mpnet-chunk in Hits@10, Recall@10 and Precision@10. The triplet recipe here (margin 1.0, bottom-`k` easy negatives) is not competitive; I did not tune it.
- **`top_k`:** raising it from 5 to 10 lowers A0 and A1 on all three cutoff-10 metrics for both losses. A possible reason, not tested, is that the positives at ranks 6 to 10 are less reliable (the gold share among the pairs fell from 0.45% to 0.31%). For the graph-only A2 the change is small and mixed: Hits@5 and Hits@10 rise by 0.01 to 0.03, Recall@10 is level, Precision@10 is level or up.
- **Ordering of the branches:** with InfoNCE, A1 > A0 > A2 in every metric at both k, as in 11.12. With triplet the A1 against A0 order is mixed.
- **Limits:** single seed (differences of about 0.01 to 0.02, one to three techniques, are not tested); one gold source (union) and the optimistic pool view only, so this table says nothing about the full 944-candidate view; one draw of pseudo-pairs per `top_k`; `top_k` changes positives and negatives together, so their effects are not separated; only 183 of the 186 union techniques are scored (3 outside the dual universe); gold is transitive through CAPEC.

## 7. Files and commands

New: `configs/experiments/attack_cwe_ent_p13_eval.yaml`, this log. Modified: `dev_docs/sprints/sprint_11.md`. No source or test change. The scratch scripts (`train1113.sh`, `p13_report.py`) are outside the repo.

```
ING=20261027T194643Z-5d3f8baf
uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
  --set training_data.inputs.text_embeddings.embedder_id=sentence_transformer-multi_qa_mpnet_dot_chunk_desc \
  --set training_data.inputs.text_embeddings.run=20261027T194928Z-8cf38e55 \
  --set training_data.inputs.graph_embeddings.run=20261027T135210Z-3b18ba0c \
  --set training_data.sampler.k=10 --set training_data.sampler.sampler_id=s11c-rank_topk-k10
# 8 new models (A1 and A2 triplet at k=5; A0, A1, A2 x InfoNCE, triplet at k=10):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
  --set seed=42 --set device=cpu --set gcl_train.loss.name=<infonce|triplet> \
  --set gcl_train.model.model_id=ent-p13-<a0|a1|a2>-<nce|trip>-k<5|10>-mpnetc-s42 \
  --set gcl_train.model.branch=<both|text|graph> --set gcl_train.model.message_passing=true \
  --set gcl_train.inputs.training_data_sampler_id=s11c-rank_topk-k<5|10> \
  --set gcl_train.inputs.training_data_run=<20261027T205523Z-ac890ca3 for k5 | 20261028T013100Z-7a027e89 for k10>
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_p13_eval --set evaluate.inputs.ingest_run=$ING
```
