# Sprint 10.3: Train the GCN row

## What was done

Row 2, GCN(SecureBERT2), trained for 3 seeds on cross-graph SecureBERT2 pseudo-pairs. No new code: `scripts/run_unsup_matrix.sh` gained priority `7`; everything else is existing options.

- **Pseudo-pairs:** `rank_topk`, k 5, SecureBERT2 text-cosine (cross-graph), `held_out_policy: none`, one `training_data` run per seed, sampler id `sb2-rank_topk-k5`.
- **Model:** the Sprint 5 A12 module: `residual_adapter`, `adapter: gcn`, `residual_floor: false` (no frozen block, `z = h`, `adapter_dim` 256, 590,848 trainable parameters), base experiment `unsup_u1` with `gcl_train.loss.name=infonce`. Settings as in 9.2–9.5: temperature 0.1, Adam 1e-3, 100 epochs, last epoch. Each seed reads the `training_data` run built with that seed.
- **Ids:** `ent-g-sb2-s42/43/44`.

## Runs

| Seed | `training_data` run | `gcl_train` run | Final loss (epoch 100) |
|---|---|---|---|
| 42 | `20261027T004948Z-8a16e080` | `20261027T005009Z-abfb4112` | 0.2391 |
| 43 | `20261027T005037Z-f91190d7` | `20261027T005100Z-f014bd14` | 0.2343 |
| 44 | `20261027T005125Z-8cf80473` | `20261027T005147Z-4cba617a` | 0.2425 |

All six runs succeeded. Manifests show `protocol: unsupervised`, `held_out_policy: none`, `n_held_out_nodes: 0`, 14,635 pseudo-pairs per seed, and the same universe hash across seeds. Inputs: ingest `20261026T224948Z-0f7e98dc` (union lineage), SecureBERT2 store `20261026T225000Z-4f1f6686`, DeepWalk store `20261026T225954Z-655383ae` (required by `training_data` for its graph columns; not an input feature of row 2). Train log: `logs/sprint_10_3_train.log`.

## Universe and gold coverage

`universe_policy: intersection` (the default, as in Sprint 9) keeps nodes present in **both** the SecureBERT2 and DeepWalk stores: **2,927 nodes** (of 5,131 with text and 3,632 with a DeepWalk vector). Gold endpoints outside it cannot be embedded by the trained row:

| Source | Pairs | Pairs lost to the universe | Techniques / CWEs lost |
|---|---|---|---|
| stix | 287 | 3 | 3 / 2 |
| capec | 566 | 0 | 0 / 0 |
| intersection | 221 | 0 | 0 / 0 |
| union | 628 | 3 | 3 / 2 |

**Consequence for 10.4:** for `stix` and `union`, rows 2 and 3 have 3 fewer scorable pairs than row 1 (frozen SecureBERT2). Report `n_gold` per row, or note it beside the tables. I did not change `universe_policy`: `text_only` would drop DeepWalk from the universe, but `gcl_train` selects the graph store by universe and would fail on nodes without a DeepWalk vector, which needs a code change outside this phase.

## Issue found and fixed

My first run showed `protocol: supervised` in the manifests: in `configs/datasets/attack_cwe_ent.yaml` I had put `protocol: unsupervised` under `dataset:` (10.1/10.2), but the guard reads it at the top level. Training was still gold-free (`held_out_policy: none`), but I moved the key to the top level, deleted the six runs and rebuilt them. The ingest and embedding runs from 10.2 are unaffected (neither reads `protocol`).

## Files

Changed: `scripts/run_unsup_matrix.sh` (priority 7), `configs/datasets/attack_cwe_ent.yaml` (`protocol` at top level), `dev_docs/sprints/sprint_10.md` (checkboxes). Unit tests: 561 pass. Outputs: `artifacts/training_data/attack_cwe_ent/sb2-rank_topk-k5/`, `models/gcl/attack_cwe_ent/ent-g-sb2-s{42,43,44}/`.

## Commands

```
TEXT_EMBEDDER=hf_mean_pool-SecureBERT2 PREFIX=ent- SAMPLER_PREFIX=sb2- \
  bash scripts/run_unsup_matrix.sh attack_cwe_ent 7
```
