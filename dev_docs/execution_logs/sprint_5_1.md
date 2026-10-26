# Sprint 5, Phase 5.1: SRGA output vector with a training-free adapter (A1)

Plan: [sprint_5.md](../sprints/sprint_5.md). Phase 5.0 had not been done when this phase started, and 5.1 needs its
`val` split and lineage, so its work is included here (section 0).

## 0. Phase 5.0 prerequisites (done here)

- `AlignmentSplit` gains `"val"` ([schemas.py](../../src/cga/data/schemas.py)); the parquet schema is unchanged.
- [ingest/pipeline.py](../../src/cga/pipelines/ingest/pipeline.py): new `carve_val_split(rows, fraction, seed)`, called by
  `IngestAlignmentsStep` when `dataset.alignments.val_split` is set. It samples distinct train `node_1_id`s (sorted, then
  `random.Random(seed)`) and relabels all their train rows `val`. The key is absent for every other dataset, so nothing
  changes there.
- [training_data/pipeline.py](../../src/cga/pipelines/training_data/pipeline.py): `_HELD_OUT_ALIGNMENT_SPLITS = {"test", "val"}`.
- [configs/datasets/ncit_doid_srga.yaml](../../configs/datasets/ncit_doid_srga.yaml): copy of the supervised config with
  `val_split: {fraction: 0.15, seed: 42}` under `alignments`.
- Ingest `20261015T154744Z-206881dd`: `n_alignments_by_split` = train 1195, val 211, test 3280.
  - A first ingest put the key under `dataset:` instead of `dataset.alignments:`, so no carve ran. It was deleted and redone.
- Embedding stores (sapbert, bert_base, deepwalk) copied to `artifacts/embeddings/ncit_doid_srga/`. Checks: `vectors.npy`,
  `index.parquet` and `status.parquet` are byte-identical to the Sprint 4 copies; `node_id` order equals the ingest;
  `input_sha256` matches the `label_synonyms_definition` `text_sha256` for 24,227/24,227 nodes (sapbert, bert_base).
- `training_data` run `20261015T160653Z-95a52c43` (`supervised-negatives_per_positive5`): 5,870 examples, 1,174 of 1,195
  train pairs usable (21 skipped), 6,935 held-out nodes.
- A0 evaluation, run `20261015T160528Z-327994a4` (`ncit_doid_srga_compare`): test equals Sprint 4 exactly (global MRR
  0.4475, candidates MRR 0.5193). A0 val is in the tables below.

## 1. What was implemented

- [models.py](../../src/cga/pipelines/gcl_train/models.py): `ResidualAdapterEncoder` (+ `build_residual_adapter_encoder`).
  - `z = [x̂ ; sqrt(lambda) * normalize(h)]`, with `x̂` the unit text vector (detached, so frozen).
  - `neighbor_mean`: `h` = mean of `x̂` over 1-hop neighbours; with `directional`, `h = [mean parents ; mean children]`
    and the whole `h` is normalized. Nodes without neighbours get `h = 0`.
  - `mlp` and `relational` raise `ValueError` (reserved for 5.2/5.3).
  - Duplicate `(src, dst)` edges are collapsed before averaging, so a pair stored under two `edge_sources` counts once.
- [gcl_train/pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py):
  - `LoadStep` builds `edge_type` beside the symmetrized `edge_index` (0 = stored child->parent, 1 = reversed). `gcn`/`gat`
    ignore it.
  - `TrainStep` picks the encoder from `gcl_train.model.encoder`. A model with 0 trainable parameters skips the optimizer
    loop (`epochs` ignored), writes an empty `metrics.jsonl`, and exports directly.
  - The manifest summary gains `n_trainable_params` for every run (0 for SRGA neighbour-mean).
- Direction convention: messages flow source->target. A type-0 edge (child->parent) delivers a child's vector to its parent,
  so it feeds the *children* mean; a type-1 edge feeds the *parents* mean.
- Configs: [ncit_doid_srga_compare.yaml](../../configs/experiments/ncit_doid_srga_compare.yaml) (A0, floor, winner;
  `gold_val` + `gold_test`) and [ncit_doid_srga_grid.yaml](../../configs/experiments/ncit_doid_srga_grid.yaml) (val only,
  all 10 grid models). The grid has its own file so no test number is read for non-winning configurations.
- Tests (420 pass, 20 new): val carve (deterministic, whole sources, test untouched, bad fraction), val endpoints held out
  and absent from examples, `gold_alignment` with `split: val`, encoder (lambda=0, columns, undirected mean, directional
  parent/child sides on a 3-node chain, isolated node, duplicate edges, zero params), pipeline (training skipped, `epochs`
  ignored, `n_trainable_params`).

## 2. Results

Val (211 pairs, global), evaluate run `20261015T161109Z-08e1534a`. `l` is lambda; `dir` is directional.

| model | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|
| sapbert (A0) | 0.2938 | 0.5498 | 0.6398 | 0.4125 |
| l=0 | 0.2938 | 0.5498 | 0.6398 | 0.4125 |
| l=0.1 | 0.2938 | 0.5829 | 0.6682 | 0.4171 |
| l=0.25 | 0.2844 | 0.5782 | 0.6730 | 0.4210 |
| l=0.5 | 0.2938 | 0.5640 | 0.6777 | 0.4162 |
| l=1 | 0.2559 | 0.5024 | 0.6398 | 0.3720 |
| l=0 dir | 0.2938 | 0.5498 | 0.6398 | 0.4125 |
| l=0.1 dir | 0.3223 | 0.5972 | 0.6682 | 0.4401 |
| **l=0.25 dir** | **0.3365** | **0.6019** | **0.6919** | **0.4555** |
| l=0.5 dir | 0.3270 | 0.5877 | 0.6777 | 0.4451 |
| l=1 dir | 0.2938 | 0.5308 | 0.6351 | 0.3991 |

Winner on val: **lambda = 0.25, directional** (MRR 0.4555 vs 0.4125, +0.043). Directional beats undirected at every
lambda > 0 (A6, training-free form). Undirected A1 is only +0.009 over A0 on val.

Test (3,280 pairs), evaluate run `20261015T161244Z-ed57d740`; read once, for the winner and the floor check only.

| mode | model | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---|---|---|---|---|
| global | sapbert (A0) | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| global | l=0 (floor) | 0.3424 | 0.5677 | 0.6537 | 0.4475 |
| global | **l=0.25 dir** | **0.3817** | **0.6235** | **0.7122** | **0.4922** |
| candidates | sapbert (A0) | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| candidates | l=0 (floor) | 0.3927 | 0.6686 | 0.7784 | 0.5193 |
| candidates | **l=0.25 dir** | **0.4277** | **0.7052** | **0.8210** | **0.5557** |

Model runs (`models/gcl/ncit_doid_srga/<model_id>/`, all dated 20261015): l0 `T160723Z-2ca8d4ed`, l0-dir `T160746Z-09f746b0`,
l0.1 `T160809Z-bcc79f1d`, l0.1-dir `T160830Z-d6a958b1`, l0.25 `T160852Z-d24e8d3c`, l0.25-dir `T160915Z-a21ef9d8`,
l0.5 `T160937Z-420f455d`, l0.5-dir `T161000Z-02a1117c`, l1 `T161022Z-3742c357`, l1-dir `T161044Z-2336f32f`.

## 3. Gate 5.1: GO

- Floor check passes: `srga-nb-l0` equals A0 on val and test in every metric.
- Best A1 (directional, lambda 0.25) beats A0 on val (+0.043 MRR) and on test (global +0.045, candidates +0.036). The
  design's undirected probe expected 0.469 global MRR; directional edges add to that, which gives the 5.3 relational
  adapter a reason to exist.
- Proceed to 5.2.

## 4. Caveats

- The grid is a single deterministic run (no training, so no seeds) on 211 val pairs. The directional gain is large
  (+0.03 MRR over undirected at lambda 0.25), but the choice of lambda among 0.1 to 0.5 is within noise; the optimum is not sharp.
- The undirected A1 was not read on test (test discipline), so the design's 0.469 is not re-confirmed there.
- In the directional vector `h` is normalized as a whole, so a node with only parents (or only children) puts all its
  residual weight on that side.
- `ruff` is not installed, so lint was not run.

## 5. Files

Modified: `src/cga/data/schemas.py`, `src/cga/pipelines/ingest/pipeline.py`,
`src/cga/pipelines/training_data/pipeline.py`, `src/cga/pipelines/gcl_train/{models,pipeline}.py`, and the tests
(`ingest`, `training_data`, `evaluate/test_gold_alignment.py`, `gcl_train/{test_models,test_pipeline}.py`),
`dev_docs/sprints/sprint_5.md` (Phase 5.0 and 5.1 boxes ticked).
Created: `configs/datasets/ncit_doid_srga.yaml`, `configs/experiments/ncit_doid_srga_{compare,grid}.yaml`, this log.
Untracked artifacts: `data/processed/ncit_doid_srga/`, `artifacts/{embeddings,training_data,evaluations}/ncit_doid_srga/`,
`models/gcl/ncit_doid_srga/`.

## 6. Commands

```
uv run pytest tests/unit -q
uv run cga run ingest --dataset ncit_doid_srga --set "ingest.split.fractions=[1.0,0.0,0.0]"
uv run cga run training_data --dataset ncit_doid_srga --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5
uv run cga run gcl_train --dataset ncit_doid_srga --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5 --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=neighbor_mean --set gcl_train.model.lambda=0.25 --set gcl_train.model.directional=true --set gcl_train.model.model_id=srga-nb-l0.25-dir
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_grid      # val only
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_compare   # val + test
```
