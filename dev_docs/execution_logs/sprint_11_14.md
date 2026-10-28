# Sprint 11.14: Negative-sampling ablation (positive-only and cross-modal hard negatives)

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [11.13](sprint_11_13.md) (the `top_k` and loss ablation, source of the reference rows).

## 0. Outcome in brief

- **Both variants were implemented and run** for A0 (full dual), A1 (text-only GCN) and A2 (graph-only GCN) with `top_k = 5` and InfoNCE: positive-only (in-batch negatives only, new loss `infonce_inbatch`) and positive + cross-modal hard negatives (new sampler option `cross_modal_hard`, existing `infonce`). Six new models, trained and evaluated on the union gold, pool view, per technique.
- **The negatives matter, and not in the same direction for every model.**
  - **A2 (graph-only):** both variants beat the bottom-5 reference in all five metrics (hard: Hits@10 +0.060, Recall@10 +0.037, Precision@10 +0.026; positive-only: +0.033, +0.021, +0.030). It is the only model for which removing the easy negatives helps everywhere. It stays below frozen mpnet-chunk in Hits@10 and Recall@10.
  - **A0 (full dual):** hard negatives give the best A0 at cutoff 10 (Hits@10 0.6120 against 0.5956, Recall@10 0.3724 against 0.3620; both about 3 techniques or less per 0.02), but lower Hits@1 and Hits@5. Positive-only is lower than the reference in Hits@5, Hits@10 and Recall@10 and higher in Hits@1 and Precision@10.
  - **A1 (text-only):** the reference is best. Hard negatives are lower in four of five metrics (Hits@10 −0.016, Recall@10 −0.017), and positive-only is clearly lower (Hits@10 −0.120, Recall@10 −0.088).
- **Positive-only is the weakest recall variant for A0 and A1, but it has the highest Hits@1 and Precision@10 among them.** Hard negatives recover most of the recall that positive-only loses (hard minus positive-only, Hits@10: +0.082 for A0, +0.104 for A1, +0.027 for A2).
- **The best row of the phase is still A1 with the bottom-5 reference negatives** (Hits@10 0.6284, Recall@10 0.4003). The 11.13 branch order (A1 above A0 above A2) holds for the reference; with hard negatives A1 is above A0 in 4 of 5 metrics (Hits@10 is tied at 0.6120); with positive-only the A1 against A0 order is mixed (A1 above in 1 of 5, 1 tied). A0 is above A2 in all five metrics for every variant.
- **Hard negatives are almost never false negatives:** 13 of 14,635 text negatives (0.09%) are gold partners of their anchor (reference: 0). No gold pair is read in training.
- **Checks passed:** default `top_k = 5` sampling with the new code reproduces the 11.11 `examples.parquet` exactly; the frozen and reference rows reproduce the 11.13 values exactly (60 values, max difference 0.0); the repeat evaluation is identical (132 of 132 rows); `n_queries` is 183 for all 11 rows; both rerun A1 models are bit-identical; 614 unit tests pass.
- Single seed, one gold source, the optimistic pool view; see section 8.

## 1. Implementation

**Hard negatives** (`src/cga/pipelines/training_data/samplers.py`, `similarity.py`, `pipeline.py`): `rank_topk` accepts `negatives: cross_modal_hard` with `hard_pool` (default 50, must be at least 1). For each anchor, positive and negative are still emitted in the existing row schema (one negative of each kind per positive), so nothing downstream changes.

- *Text side:* positives are the anchor's top-`k` text-similar nodes of the other graph. The candidate pool is the anchor's text ranks `k+1 … k+hard_pool`. For each positive `p`, in rank order, the negative is the unused pool node with the **lowest DeepWalk cosine to `p`** (both nodes are in the same graph). Ties go to the earlier pool node.
- *Graph side:* positives are the anchor's top-`k` DeepWalk neighbours (the anchor itself excluded). The pool is the anchor's DeepWalk ranks `k+1 … k+hard_pool`. The negative of each positive `q` is the unused pool node with the **lowest text cosine to `q`**. This needs same-graph text similarity, added as `compute_text_within_similarity` and computed only when the option is on.
- A pool node is used for at most one positive of an anchor, so the five negatives of an anchor are distinct. Existing options and defaults are unchanged.

**Positive-only loss** (`gcl_train/losses.py`, `trainer.py`, `pipeline.py`): `infonce_inbatch` registered next to `infonce`. For each row it scores the row's positive against every row's positive in the batch (full batch, all 14,635 rows), restricted to candidates of the same graph role as the row's positive. Masked out of the denominator: the row's own positive (it is the numerator) and every other pseudo-positive of the same anchor. A candidate that occurs in several rows is counted once per occurrence, as in `infonce`. It uses the same temperature (0.1), normalisation and row chunking as `infonce`, never reads `text_neg` or `graph_neg`, and `metrics.jsonl` has the same fields. The training data run is the unchanged `s11c-rank_topk-k5`; the explicit-negative columns in it are ignored.

**Tests (12 new, 614 in the suite pass):** hard negatives equal a brute-force argmin on random fixtures (both graphs); they are distinct, never the anchor or any of its positives, in the right graph; sampling is reproducible and keeps the `bottomk` positives row for row; only ids of the given matrices appear; a pool shorter than `k` caps the rows; missing inputs and `hard_pool < 1` raise. Within-graph text similarity skips held-out nodes. Positive-only loss equals a naive masked per-row reference (value and gradient), is the same chunked and unchunked, masks the anchor's own positives, is finite and reproducible, and training with `infonce_inbatch` gives identical losses and embeddings when the negative columns are changed; it raises without `node_roles`.

## 2. Hard-negative training data

`training_data` run `20261028T023558Z-decc373b` (sampler id `s11c-rank_topk-k5-xhard`), seed 42, same pinned ingest `20261027T194643Z-5d3f8baf`, text `20261027T194928Z-8cf38e55` and DeepWalk `20261027T135210Z-3b18ba0c`. Universe 2,927 (0 held out); 14,635 examples, 5 per anchor, all in split `train`.

| Check | Result |
|---|---|
| Default `top_k = 5` sampling with the new code, scratch run, compared with `s11c-rank_topk-k5/…205523Z-ac890ca3` | identical (all columns); the scratch run was deleted |
| Positive columns (anchor, text positive, graph positive) against `s11c-rank_topk-k5` | identical row for row |
| Negative equal to the anchor or to one of its positives | 0 of 29,270 |
| Text negatives in the other graph; graph negatives in the anchor's graph | all |
| Distinct text and graph negatives per anchor | yes, for every anchor |

**Diagnostics** (read-only; the gold is used here only for reporting; cosine; means over the 14,635 rows):

| | bottom-5 reference | cross-modal hard |
|---|---|---|
| Text negatives that are gold partners of their anchor | 0 | 13 (0.09%) |
| Text negative: DeepWalk cosine to its positive | 0.416 | 0.311 |
| Text negative: text cosine to the anchor | 0.131 | 0.419 |
| Graph negative: text cosine to its positive | 0.392 | 0.285 |
| Graph negative: DeepWalk cosine to the anchor | 0.115 | 0.544 |
| Distinct text negatives / graph negatives | 787 / 2,140 | 1,385 / 1,842 |

The hard negatives are much closer to the anchor in their own modality (text 0.419 against 0.131, DeepWalk 0.544 against 0.115) and farther from the positive in the other modality, as defined.

## 3. Runs (exact configuration)

Fixed for all: mpnet-chunk text store `20261027T194928Z-8cf38e55`, DeepWalk `20261027T135210Z-3b18ba0c`, `unsup_u2` (hidden / out 128, dropout 0, Adam lr 1e-3, 100 epochs), `alpha 0.5`, temperature 0.1, seed 42, CPU with one thread, `message_passing: true`, `top_k = 5`.

| Row | Model id | Run | Branch | Negatives | Loss | Sampler / training data run | Params | `loss` epoch 1 → 100 | Source |
|---|---|---|---|---|---|---|---|---|---|
| a0-nce-k5 | `ent-dual-gcn-infonce-mpnetc-s42` | `20261027T205648Z-2aeb3259` | both | bottom-5 | `infonce` | `s11c-rank_topk-k5` / `…205523Z-ac890ca3` | 328,576 | 8.855 → 5.799 | reused |
| a1-nce-k5 | `ent-abl-text-gcn-nce-mpnetc-s42` | `20261027T232407Z-adc619c8` | text | bottom-5 | `infonce` | same | 246,272 | 8.849 → 6.108 | reused |
| a2-nce-k5 | `ent-abl-graph-gcn-nce-mpnetc-s42` | `20261027T232541Z-d91ebd8e` | graph | bottom-5 | `infonce` | same | 82,432 | 8.773 → 6.595 | reused |
| a0-pos-k5 | `ent-p14-a0-pos-k5-mpnetc-s42` | `20261028T023713Z-eb4a8c99` | both | in-batch only | `infonce_inbatch` | `s11c-rank_topk-k5` / `…205523Z-ac890ca3` | 328,576 | 7.978 → 5.859 | new |
| a1-pos-k5 | `ent-p14-a1-pos-k5-mpnetc-s42` | `20261028T024655Z-1bedbfd1` | text | in-batch only | `infonce_inbatch` | same | 246,272 | 8.483 → 6.270 | new |
| a2-pos-k5 | `ent-p14-a2-pos-k5-mpnetc-s42` | `20261028T025100Z-042149f3` | graph | in-batch only | `infonce_inbatch` | same | 82,432 | 7.803 → 6.331 | new |
| a0-hard-k5 | `ent-p14-a0-hard-k5-mpnetc-s42` | `20261028T024245Z-3809a000` | both | cross-modal hard, M = 50 | `infonce` | `s11c-rank_topk-k5-xhard` / `…20261028T023558Z-decc373b` | 328,576 | 8.899 → 6.604 | new |
| a1-hard-k5 | `ent-p14-a1-hard-k5-mpnetc-s42` | `20261028T024917Z-8a12c39c` | text | cross-modal hard, M = 50 | `infonce` | same | 246,272 | 9.068 → 6.918 | new |
| a2-hard-k5 | `ent-p14-a2-hard-k5-mpnetc-s42` | `20261028T025322Z-47c176f6` | graph | cross-modal hard, M = 50 | `infonce` | same | 82,432 | 8.844 → 7.127 | new |

Each manifest and resolved config was checked: 100 metric rows, loss lower at epoch 100, 2,927 x 128 finite embeddings, 2,927 nodes, loss name, branch, sampler id and training data run as in the table. Loss values of different variants are not comparable (different denominators). **Determinism:** A1 was retrained once per variant with the same seed (`…025504Z-1bedbfd1` and `…025739Z-8a12c39c`); the node embeddings equal the evaluated runs exactly.

## 4. Evaluation

`configs/experiments/attack_cwe_ent_p14_eval.yaml` (11 pinned rows, one evaluation `set_pool`, `ks: [1, 5, 10]`, union pool). Evaluate run `20261028T025936Z-36368eac`; repeat `20261028T025944Z-36368eac`.

**Checks:** (1) frozen mpnet-chunk, frozen DeepWalk and the three reference rows equal the 11.13 values at k = 1, 5, 10: 60 values, max difference 0.0. (2) The repeat run is identical in all 132 rows. (3) `n_queries` = 183 for all 11 methods.

## 5. Comparison table (union gold, pool view, per technique)

Hits@1 and Hits@5 use cutoffs 1 and 5; Hits@10, Recall@10 and Precision@10 use cutoff 10. Best value per column in bold. One technique is 0.0055 in Hits@k.

| Model | Negatives | Hits@1 | Hits@5 | Hits@10 | Recall@10 | Precision@10 |
|---|---|---|---|---|---|---|
| mpnet-chunk (frozen) | – | 0.1530 | 0.3716 | 0.5191 | 0.2905 | 0.0749 |
| DeepWalk (frozen) | – | 0.0328 | 0.1093 | 0.1639 | 0.0538 | 0.0257 |
| A0 full dual | bottom-5 explicit (reference) | 0.1749 | 0.4754 | 0.5956 | 0.3620 | 0.0951 |
| A0 full dual | none explicit (in-batch only) | **0.1913** | 0.4208 | 0.5301 | 0.3270 | **0.1049** |
| A0 full dual | cross-modal hard | 0.1475 | 0.4317 | 0.6120 | 0.3724 | 0.0984 |
| A1 text-only GCN | bottom-5 explicit (reference) | 0.1858 | **0.4809** | **0.6284** | **0.4003** | 0.0962 |
| A1 text-only GCN | none explicit (in-batch only) | **0.1913** | 0.4262 | 0.5082 | 0.3125 | 0.1016 |
| A1 text-only GCN | cross-modal hard | 0.1530 | 0.4481 | 0.6120 | 0.3837 | 0.0995 |
| A2 graph-only GCN | bottom-5 explicit (reference) | 0.1038 | 0.2514 | 0.4098 | 0.2441 | 0.0579 |
| A2 graph-only GCN | none explicit (in-batch only) | 0.1257 | 0.3224 | 0.4426 | 0.2651 | 0.0874 |
| A2 graph-only GCN | cross-modal hard | 0.1202 | 0.3279 | 0.4699 | 0.2813 | 0.0842 |

**Differences** (Hits@1 / Hits@5 / Hits@10 / Recall@10 / Precision@10):

| Model | hard − reference | positive-only − reference | hard − positive-only |
|---|---|---|---|
| A0 | −0.0273 / −0.0437 / +0.0164 / +0.0104 / +0.0033 | +0.0164 / −0.0546 / −0.0656 / −0.0350 / +0.0098 | −0.0437 / +0.0109 / +0.0820 / +0.0454 / −0.0066 |
| A1 | −0.0328 / −0.0328 / −0.0164 / −0.0166 / +0.0033 | +0.0055 / −0.0546 / −0.1202 / −0.0878 / +0.0055 | −0.0383 / +0.0219 / +0.1038 / +0.0712 / −0.0022 |
| A2 | +0.0164 / +0.0765 / +0.0601 / +0.0372 / +0.0262 | +0.0219 / +0.0710 / +0.0328 / +0.0209 / +0.0295 | −0.0055 / +0.0055 / +0.0273 / +0.0163 / −0.0033 |

**Against frozen mpnet-chunk** (Hits@10 / Recall@10 / Precision@10): A0 hard +0.093 / +0.082 / +0.024; A0 positive-only +0.011 / +0.037 / +0.030; A1 hard +0.093 / +0.093 / +0.025; A1 positive-only −0.011 / +0.022 / +0.027; A2 hard −0.049 / −0.009 / +0.009; A2 positive-only −0.077 / −0.025 / +0.013. The reference rows are +0.077 / +0.072 / +0.020 (A0), +0.109 / +0.110 / +0.021 (A1) and −0.109 / −0.046 / −0.017 (A2).

## 6. Report (no tuning)

- **Effect of the negatives per model.** For the graph-only A2, either change helps in every metric, hard slightly more at cutoff 10 (Hits@10 +0.060 against +0.033) and positive-only more in Hits@1 and Precision@10. For A0 the picture is split: hard negatives help at cutoff 10 and hurt the top ranks; positive-only helps Hits@1 and Precision@10 and hurts Hits@5, Hits@10 and Recall@10. For the text-only A1 neither change beats the easy bottom-5 negatives except in Precision@10, where both are higher (+0.003, +0.006).
- **Hard against positive-only.** Hard negatives are above positive-only in Hits@5, Hits@10 and Recall@10 for all three models (3 of 5 metrics each) and below it in Hits@1 and Precision@10 (A2: Precision@10 −0.003, a fraction of a technique). So explicit hard negatives restore the recall that the in-batch-only loss gives up.
- **Branch order.** With the reference negatives A1 > A0 > A2 in all five metrics (as in 11.13). With positive-only, A1 is above A0 in 1 of 5 metrics (1 tied) and A0 is above A2 in all 5. With hard negatives A1 is above A0 in 4 of 5 (Hits@10 tied at 0.6120) and A0 is above A2 in all 5.
- **False negatives.** Only 13 of 14,635 hard text negatives (0.09%) are gold partners, so the hard variant is not mainly punished by false negatives; the lower A1 result is not explained by them (not tested further). The graph-side negatives cannot be gold partners (same graph).
- **Relation to frozen text.** A1 and A0 with hard negatives still beat frozen mpnet-chunk by +0.09 in Hits@10 and Recall@10; A1 positive-only is slightly below in Hits@10 (−0.011) and above in Recall@10 (+0.022) and Precision@10.

## 7. Files and commands

New: `configs/experiments/attack_cwe_ent_p14_eval.yaml`, this log. Modified: `src/cga/pipelines/training_data/samplers.py`, `similarity.py`, `pipeline.py`; `src/cga/pipelines/gcl_train/losses.py`, `trainer.py`, `pipeline.py`; tests `tests/unit/pipelines/training_data/test_samplers.py`, `test_similarity.py`, `tests/unit/pipelines/gcl_train/test_losses.py`, `test_trainer.py`; `dev_docs/sprints/sprint_11.md`. Scratch scripts (`train1114.sh`, `diag1114.py`, `p14_report.py`) are outside the repo.

```
uv run pytest tests/unit -q
ING=20261027T194643Z-5d3f8baf
uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
  --set training_data.inputs.ingest_run=$ING \
  --set training_data.inputs.text_embeddings.embedder_id=sentence_transformer-multi_qa_mpnet_dot_chunk_desc \
  --set training_data.inputs.text_embeddings.run=20261027T194928Z-8cf38e55 \
  --set training_data.inputs.graph_embeddings.run=20261027T135210Z-3b18ba0c \
  --set training_data.sampler.k=5 --set training_data.sampler.negatives=cross_modal_hard \
  --set training_data.sampler.hard_pool=50 --set training_data.sampler.sampler_id=s11c-rank_topk-k5-xhard
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
  --set seed=42 --set device=cpu --set gcl_train.model.message_passing=true \
  --set gcl_train.model.model_id=ent-p14-<a0|a1|a2>-<pos|hard>-k5-mpnetc-s42 \
  --set gcl_train.model.branch=<both|text|graph> \
  --set gcl_train.loss.name=<infonce_inbatch for pos | infonce for hard> \
  --set gcl_train.inputs.training_data_sampler_id=<s11c-rank_topk-k5 for pos | s11c-rank_topk-k5-xhard for hard> \
  --set gcl_train.inputs.training_data_run=<20261027T205523Z-ac890ca3 for pos | 20261028T023558Z-decc373b for hard>
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_p14_eval --set evaluate.inputs.ingest_run=$ING
```

## 8. Limits and assumptions

- **Interpretation of "hard".** Hard negatives are defined as near the anchor in the sampling modality (ranks `k+1 … k+50`) but far from the positive in the other modality. This was my reading of "low structural similarity / low text similarity"; the pool depth `M = 50` was fixed, not tuned. A different pool definition gives a different variant.
- **Positive-only keeps in-batch negatives by design:** the pseudo-positives of other anchors, restricted to the positive's graph. A pseudo-positive of another anchor can still be a true match of this anchor; only the anchor's own five are masked.
- Single seed; differences of 0.01 to 0.02 (one to three techniques) are not tested against seed noise. One gold source (union), the pool view only (optimistic, built from gold); gold is transitive through CAPEC; 3 of the 186 union techniques lie outside the dual universe.
- Positives are the frozen mpnet's own top-5 guesses (0.45% gold), unchanged here; only the negatives differ between rows.
