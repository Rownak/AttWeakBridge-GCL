# Sprint 3.2 — First end-to-end training run (execution log)

SapBERT + DeepWalk, `rank_topk` `k=5`, GCN then GAT. Inputs: ingest `20261011T134334Z-6d45b8af`
(with the Phase 3.0.0 NCIT intersection edges), `hf_mean_pool-sapbert`, `deepwalk-deepwalk/20261011T134446Z-d5aa89ed`.

## Task 1 — `training_data`

```
uv run cga run training_data --dataset ncit_doid \
  --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert \
  --set training_data.sampler.name=rank_topk --set training_data.sampler.k=5
```
Run `20261011T160509Z-77dcd0ed` -> `artifacts/training_data/ncit_doid/rank_topk-k5/`.

| Item | Value |
|---|---|
| Universe (`intersection`) | 24,221 nodes (24,227 minus the 6 isolated NCIT nodes with no DeepWalk vector) |
| `held_out` | 4,845 (20.0%: ingest's val+test node split; no alignments, so nothing else) |
| Similarity matrices | text 12,604 x 6,772, graph_1 12,604 x 12,604, graph_2 6,772 x 6,772 |
| `n_examples` | 96,880 (all `train` split; anchors come only from train nodes) |
| Runtime | 32 s |
| **Peak RAM** | **1.96 GB** (process-tree RSS, sampled every 0.5 s) on a 64 GB machine |

**Scale check (Phase 3.0.1 item): passed.** Similarity is built over train-split anchors only (12,604 NCIT x
6,772 DOID), not the full 15.8k x 8.5k, so memory is far below the ~1 GB-per-matrix estimate. No size-reduction
work is needed.

## Task 2 — `gcl_train` GCN and GAT

```
uv run cga run gcl_train --dataset ncit_doid --set gcl_train.inputs.training_data_sampler_id=rank_topk-k5 \
  --set gcl_train.model.model_id=gcn-sapbert-deepwalk --set gcl_train.train.epochs=100
uv run cga run gcl_train --dataset ncit_doid --set gcl_train.inputs.training_data_sampler_id=rank_topk-k5 \
  --set gcl_train.model.encoder=gat --set gcl_train.model.model_id=gat-sapbert-deepwalk --set gcl_train.train.epochs=100
```
| Model | Run | Embeddings | Time | Peak RAM |
|---|---|---|---|---|
| `gcn-sapbert-deepwalk` | `20261011T160602Z-6fc64509` | 24,221 x 128 | 22 s | 1.45 GB |
| `gat-sapbert-deepwalk` | `20261011T160626Z-e9f5267a` | 24,221 x 128 | 37 s | 1.49 GB |

Other settings are the pipeline defaults (`alpha=0.5`, triplet loss, margin 1.0, hidden/out dim 128, lr 1e-3,
GAT heads `[8, 1]`). Both runs trained on the GPU in well under a minute.

## Task 3 — Training sanity

Read from each run's `metrics.jsonl` (100 rows, one per epoch; fields `loss`, `loss_graph`, `loss_text`):

| Model | loss epoch 1 -> 100 | loss_graph | loss_text | NaN/inf |
|---|---|---|---|---|
| GCN | 0.686 -> 0.057 | 0.599 -> 0.051 | 0.772 -> 0.062 | none |
| GAT | 0.683 -> 0.047 | 0.606 -> 0.039 | 0.760 -> 0.054 | none |

Loss falls steadily for both and both views, and is still decreasing slowly at epoch 100 (not fully
converged; 100 epochs is the sprint's fixed budget). GAT reaches a lower training loss than GCN. Training loss
alone does not rank the models; that is `evaluate`'s job (Task 4).

## Task 4 — Experiment and plot configs

- [configs/experiments/ncit_doid_neighborhood_compare.yaml](../../configs/experiments/ncit_doid_neighborhood_compare.yaml):
  methods `deepwalk` (`deepwalk-deepwalk`), `sapbert` (`hf_mean_pool-sapbert`), `gcn` (`gcn-sapbert-deepwalk`),
  `gat` (`gat-sapbert-deepwalk`); `neighborhood` with views `structural_r1`, `structural_r2`, `textual`
  (reference `hf_mean_pool-bert_base`, `top_k: 10`), `ks: [1, 5, 10, 20]`; `cluster_quality` with
  `n_clusters: [2, 4, 8, 16, 32]`. Identical to `full_grid_compare.yaml` apart from the method list.
- [configs/plots/ncit_doid_neighborhood_compare.yaml](../../configs/plots/ncit_doid_neighborhood_compare.yaml):
  precision@k and silhouette line plots plus loss curves.

## Task 5 — `evaluate` and `plot`

```
uv run cga run evaluate --dataset ncit_doid --experiment ncit_doid_neighborhood_compare
uv run cga run plot --dataset ncit_doid --config configs/plots/ncit_doid_neighborhood_compare.yaml
```
- evaluate run `20261011T160719Z-0545f962`: 384 neighborhood rows + 40 cluster rows, **62 min** (`neighborhood`
  ~49 min, `cluster_quality` ~5 min, t-SNE for gcn/gat ~7 min), peak RAM 2.92 GB. Slow because the evaluator
  loops per query in Python over ~24k nodes; no change made.
- plot run `20261011T170935Z-17b8c2b3` (`artifacts/plots/ncit_doid/...`).
- Queries: 15,756 NCIT and 8,465 DOID per view (all nodes of both graphs, including train and held-out; see the
  sprint caveat: these are proxy relevance sets, not gold labels).

### Results (higher is better)

MRR:

| view | graph | DeepWalk | SapBERT | GCN | GAT |
|---|---|---|---|---|---|
| structural_r1 | NCIT | **0.722** | 0.473 | 0.615 | 0.634 |
| structural_r1 | DOID | **0.475** | 0.210 | 0.404 | 0.418 |
| structural_r2 | NCIT | **0.998** | 0.761 | 0.957 | 0.966 |
| structural_r2 | DOID | **0.998** | 0.608 | 0.890 | 0.918 |
| textual (bert_base ref) | NCIT | 0.331 | **0.484** | 0.405 | 0.402 |
| textual (bert_base ref) | DOID | 0.287 | **0.379** | 0.344 | 0.347 |

normalized precision@10:

| view | graph | DeepWalk | SapBERT | GCN | GAT |
|---|---|---|---|---|---|
| structural_r1 | NCIT | **0.860** | 0.480 | 0.678 | 0.706 |
| structural_r1 | DOID | **0.736** | 0.297 | 0.542 | 0.568 |
| structural_r2 | NCIT | **0.922** | 0.468 | 0.684 | 0.705 |
| structural_r2 | DOID | **0.927** | 0.414 | 0.657 | 0.691 |
| textual | NCIT | 0.120 | **0.161** | 0.131 | 0.131 |
| textual | DOID | 0.127 | **0.159** | 0.142 | 0.144 |

Silhouette (mean over `n_clusters` 2..32, both graphs): GAT **0.159**, GCN 0.153, DeepWalk 0.057, SapBERT 0.031
(GCN/GAT lead at every `n_clusters` on both graphs; the one near-tie is DOID at 32 clusters: DeepWalk 0.118, GCN 0.111, GAT 0.123).

### Reading
- The trained dual encoders sit **between** the two single-view baselines on neighborhood metrics: they gain on
  the textual view over DeepWalk and on the structural views over SapBERT, but do not beat the best baseline on
  either view. That is expected from a joint objective with `alpha=0.5`.
- They clearly win on **cluster quality** (silhouette ~3-5x the baselines).
- `structural_r2` MRR for DeepWalk is ~0.998 because 2-hop relevant sets are large; it saturates and discriminates
  little. Prefer `structural_r1` and precision@k when comparing.
- SapBERT is evaluated against the bert_base textual reference, so its textual lead partly reflects two text
  encoders agreeing, not task accuracy.
- GAT is slightly ahead of GCN on structural views and silhouette; they tie on the textual view.
- Results are single-seed, 100 epochs, and only a relative comparison (sprint caveat).

## Commands to reproduce
```
uv run cga run training_data --dataset ncit_doid --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=rank_topk --set training_data.sampler.k=5
uv run cga run gcl_train --dataset ncit_doid --set gcl_train.inputs.training_data_sampler_id=rank_topk-k5 --set gcl_train.model.model_id=gcn-sapbert-deepwalk --set gcl_train.train.epochs=100
uv run cga run gcl_train --dataset ncit_doid --set gcl_train.inputs.training_data_sampler_id=rank_topk-k5 --set gcl_train.model.encoder=gat --set gcl_train.model.model_id=gat-sapbert-deepwalk --set gcl_train.train.epochs=100
uv run cga run evaluate --dataset ncit_doid --experiment ncit_doid_neighborhood_compare
uv run cga run plot --dataset ncit_doid --config configs/plots/ncit_doid_neighborhood_compare.yaml
```
