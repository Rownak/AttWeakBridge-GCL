# Sprint 9, Phase 9.1: Protocol, datasets and sampler options

Plan: [sprint_9.md](../sprints/sprint_9.md). Every new option is default-off; the full unit suite passes (532 tests). Nothing from Sprints 1–8 was re-run or changed.

## 0. Outcome in brief

- **Gate 9.1: GO.** All four conditions hold:

| gate condition | result |
|---|---|
| Tests green | 532 passed (`uv run pytest tests/unit -q`) |
| Pseudo-pairs byte-identical with and without `alignments.parquet` | Yes on the real NCIT–DOID data: `examples.parquet` and `universe.parquet` have the same sha256 with `alignments.parquet` present (run `20261026T024611Z-7f5b967a`) and hidden (`20261026T025243Z-7f5b967a`). Also a unit test. |
| U0 reproduces a plain `embedding_store` evaluation of `bert_base` | Yes, exactly (all 4 ranking and 6 P/R/F1 values, section 4). |
| ATT&CK–CWE runs end-to-end | Yes: ingest → embed stores → `training_data` → U1 / U2 `gcl_train` → gold-free `neighborhood` evaluation → `summary.md`. |

- **Read this before 9.2:** the 20-epoch smoke models are **worse than U0** on NCIT–DOID (U1 MRR 0.0142, U2 MRR 0.0043, against U0 0.0186). They are pipeline checks, not results, and nothing was changed because of them (the settings are fixed in advance). Whether 100 epochs changes the picture is what 9.2 measures.

## 1. What was implemented

- **Protocol guard** [protocol.py](../../src/cga/core/protocol.py) (new): `get_protocol`, `is_unsupervised`, `refuse`. `protocol` defaults to `supervised`; an unknown value raises `ConfigError`.
  - `training_data` ([pipeline.py](../../src/cga/pipelines/training_data/pipeline.py), `check_protocol`) refuses `sampler.name=supervised` and any `held_out_policy` other than `none`.
  - `gcl_train` ([pipeline.py](../../src/cga/pipelines/gcl_train/pipeline.py), `check_protocol`) refuses `loss.name=infonce_full` and `monitor.enabled=true`.
  - `evaluate` is not guarded: it must read gold test pairs.
- **`training_data.held_out_policy`** (`legacy` default | `none`). `none` marks every node eligible and never opens `alignments.parquet`. `stats.json` / the manifest summary now record `protocol`, `held_out_policy` and `n_pseudo_pairs` (distinct `(anchor, text positive)` pairs; for `rank_topk` it equals `n_examples`).
- **`rank_topk` options** [samplers.py](../../src/cga/pipelines/training_data/samplers.py):
  - `negatives: bottomk | random | hard`, `hard_offset` (default 10). `random` draws up to k columns outside the anchor's positives with the seeded RNG. `hard` takes text ranks `hard_offset+1 … hard_offset+k`. `hard_offset < k` raises `ConfigError`.
  - `positives: topk | mutual`. `mutual` keeps a positive only if the anchor is in the positive's own top-k, in both the text and the graph view; fewer than k may survive, and the rank-zip stops at the shortest list.
  - Non-default variants get their own sampler id (`rank_topk-k5-negrandom`, `…-neghard-off10`, `…-posmutual`), so `LATEST` pointers do not collide. The default id is unchanged.
- **`gold_alignment.sibling_filter_splits`** (default: all splits) in [gold_alignment.py](../../src/cga/pipelines/evaluate/evaluators/gold_alignment.py).
- **`residual_adapter` with row losses:** no code change was needed. A new test trains `adapter: mlp` with `triplet` and `infonce` (`alpha 1.0`) end to end (finite loss, finite embeddings, trainable parameters).
- **`gcl_train` manifest summary** records `protocol`, `held_out_policy` and `n_pseudo_pairs` (read from the `training_data` run's summary).
- **Datasets:** [ncit_doid_unsup.yaml](../../configs/datasets/ncit_doid_unsup.yaml) (alignments `train` + `test`, no `val_split`) and [attack_cwe_unsup.yaml](../../configs/datasets/attack_cwe_unsup.yaml) (`alignments: null`). Both set `protocol: unsupervised`.
- **Experiment configs:** [unsup_u1.yaml](../../configs/experiments/unsup_u1.yaml) and [unsup_u2.yaml](../../configs/experiments/unsup_u2.yaml) hold the fixed Sprint 9 settings (k 5, Adam 1e-3, 100 epochs, margin 1.0). [ncit_doid_unsup_smoke.yaml](../../configs/experiments/ncit_doid_unsup_smoke.yaml) and [attack_cwe_unsup_smoke.yaml](../../configs/experiments/attack_cwe_unsup_smoke.yaml) are the 9.1 readouts. The 9.2 files (`ncit_doid_unsup_test.yaml`, `attack_cwe_unsup_eval.yaml`) are left to 9.2.
- **Scripts:** [unsup_summary.py](../../scripts/unsup_summary.py) (mean ± sample std per seed group, `—` for missing gold columns, writes `summary.md` + `summary.csv` into the run) and [copy_embedding_store.py](../../scripts/copy_embedding_store.py) (copy without `shards/` and verify).
- **README:** new "Unsupervised ablations" section.

## 2. Data and stores

Ingest with `ingest.split.fractions=[1.0,0.0,0.0]`:

| dataset | ingest run | content |
|---|---|---|
| `ncit_doid_unsup` | `20261026T024157Z-3bde93ef` | 24,227 nodes (15,762 / 8,465), 32,874 edges, alignments train 1,406 + test 3,280, all nodes `train` |
| `attack_cwe_unsup` | `20261026T024142Z-91ff1328` | 1,217 nodes, no alignments, all nodes `train` |

ATT&CK–CWE: the node list has the same order as both earlier `attack_cwe` ingests. NCIT–DOID: counts equal `ncit_doid_srga_full`, and the copied store order matches the new `nodes.parquet` (above).

Stores copied without `shards/` by `copy_embedding_store.py`, which checks byte equality with the source, `AlignedMatrix.load` (manifest hashes), node order against the new `nodes.parquet`, and for text stores `input_sha256` against the ingest's text hashes:

| dataset | store | run | rows | checks |
|---|---|---|---|---|
| `ncit_doid_unsup` | `hf_mean_pool-bert_base` | `20261009T191700Z-1278c23a` | 24,227 | all pass; hash match 24,227 / 24,227 (`label_synonyms_definition`) |
| `ncit_doid_unsup` | `deepwalk-deepwalk` | `20261011T134446Z-d5aa89ed` | 24,221 | order ok (no text hash) |
| `attack_cwe_unsup` | `hf_mean_pool-bert_base` | `20260917T193723Z-536058b0` | 1,214 | all pass; hash match 1,214 / 1,214 (`description`) |
| `attack_cwe_unsup` | `deepwalk-deepwalk` | `20260917T142532Z-af282dba` | 1,137 | order ok |

The stores omit nodes (3 with empty text; 6 / 80 without a DeepWalk vector), so the training universe is 24,221 (NCIT–DOID) and 1,136 (ATT&CK–CWE). **No NCIT–DOID gold endpoint is among the 6 missing nodes**, so all 3,280 test pairs are scored for every method.

## 3. Smoke runs (seed 42, 20 epochs, fixed settings)

`training_data` (k 5, `held_out_policy: none`): NCIT–DOID `20261026T024611Z-7f5b967a` (121,105 examples, 0 held out), ATT&CK–CWE `20261026T024704Z-a60898d1` (5,680 examples). U1 and U2 share one run per dataset (same sampler).

| dataset | model | `gcl_train` run |
|---|---|---|
| NCIT–DOID | `u1-s42` (U1) | `20261026T024717Z-8dc20aca` |
| NCIT–DOID | `u2-s42` (U2) | `20261026T024758Z-e34112ba` |
| ATT&CK–CWE | `u1-s42` | `20261026T024815Z-1ee4340e` |
| ATT&CK–CWE | `u2-s42` | `20261026T024825Z-b17715d5` |

## 4. Readouts

NCIT–DOID gold test, 3,280 pairs, run `20261026T024853Z-3e2fa375` (`summary.md` in the run; global mode, sibling filter `[test]`, top-3 greedy one-to-one, threshold 0, no BioGITOM filters):

| model | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct |
|---|---|---|---|---|---|---|---|---|
| U0 frozen BERT | 0.0116 | 0.0308 | 0.0186 | 0.0712 | 0.0119 | 0.0204 | 548 | 39 |
| U1 (20 ep) | 0.0091 | 0.0223 | 0.0142 | 0.0593 | 0.0091 | 0.0158 | 506 | 30 |
| U2 (20 ep) | 0.0021 | 0.0070 | 0.0043 | 0.0063 | 0.0015 | 0.0025 | 798 | 5 |

**U0 check.** `bert_base` evaluated alone as an `embedding_store` (run `20261026T024955Z-3af8c8d4`) gives the same 10 values. The default sibling filter (all splits) and `[test]` also agree exactly, so on this gold set the restriction does not change U0.

ATT&CK–CWE, gold-free `neighborhood`, run `20261026T025034Z-153243c7` (every gold column is `—`). Hit@10 against the 1-hop structural view: U0 0.168 / 0.250 (graph_1 / graph_2), U1 0.177 / 0.244, U2 0.527 / 0.750. U2 raises structural agreement a lot and loses some textual agreement (graph_2 textual hit@10 1.00 → 0.64); this is one seed and 20 epochs.

## 5. Decisions, assumptions, limitations

- **`negatives` changes the text negatives only.** The plan defines `hard` on text ranks; `random` follows it, and the graph negatives stay bottom-k. If 9.2 should randomize both views, it is a small change.
- **`held_out_policy: none` ignores the node `split` too** (nobody is held out even if ingest produced val / test nodes). With the all-train ingest this is moot.
- **Extra guard (not in the plan):** under `protocol: unsupervised`, `gcl_train` also refuses a `training_data` run whose summary lacks `held_out_policy: none` (including runs from before this phase). A legacy run holds out gold test endpoints, so gold would have shaped the examples.
- **Test was read for the smoke models** (seed 42, 20 epochs; three evaluations on the same 3,280 pairs). It informed no choice. Phase 9.2's "read once per configuration" applies to the 100-epoch matrix rows; this should be disclosed with them.
- The latest `training_data` pointer for `ncit_doid_unsup/rank_topk-k5` is the no-alignments run `20261026T025243Z-7f5b967a`; it is byte-identical to the run the smoke models trained on.
- `monitor` is declared in `gcl_train` config but not implemented in code; the guard still refuses `enabled: true`.
- Mutual positives are computed lazily per candidate row, so the cost grows with the number of distinct positives (not measured at full size; 9.2 runs it).

## 6. Commands

```bash
uv run pytest tests/unit -q
uv run cga run ingest --dataset ncit_doid_unsup  --set "ingest.split.fractions=[1.0,0.0,0.0]"
uv run cga run ingest --dataset attack_cwe_unsup --set "ingest.split.fractions=[1.0,0.0,0.0]"
uv run python scripts/copy_embedding_store.py --src <store run dir> --dst-lineage <artifacts/embeddings/{dataset}/{embedder_id}> --ingest <ingest run dir> [--text-variant <variant>]
for ds in ncit_doid_unsup attack_cwe_unsup; do
  uv run cga run training_data --dataset $ds --experiment unsup_u2
  for u in u1 u2; do
    uv run cga run gcl_train --dataset $ds --experiment unsup_$u --set seed=42 --set gcl_train.model.model_id=$u-s42 --set gcl_train.train.epochs=20
  done
done
uv run cga run evaluate --dataset ncit_doid_unsup  --experiment ncit_doid_unsup_smoke
uv run cga run evaluate --dataset attack_cwe_unsup --experiment attack_cwe_unsup_smoke
uv run python scripts/unsup_summary.py --dataset ncit_doid_unsup --run 20261026T024853Z-3e2fa375
```

## 7. Files

New: `src/cga/core/protocol.py`, `configs/datasets/{ncit_doid,attack_cwe}_unsup.yaml`, `configs/experiments/{unsup_u1,unsup_u2,ncit_doid_unsup_smoke,attack_cwe_unsup_smoke}.yaml`, `scripts/{unsup_summary,copy_embedding_store}.py`, `tests/unit/core/test_protocol.py`, `tests/unit/test_unsup_summary.py`.
Modified: `src/cga/pipelines/training_data/{pipeline,samplers}.py`, `src/cga/pipelines/gcl_train/pipeline.py`, `src/cga/pipelines/evaluate/evaluators/gold_alignment.py`, `README.md`, and the tests `tests/unit/pipelines/{training_data/test_pipeline,training_data/test_samplers,gcl_train/test_pipeline,evaluate/test_gold_alignment}.py`.
