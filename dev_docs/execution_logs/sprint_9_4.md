# Sprint 9, Phase 9.4: DeepWalk-only rows

Plan: [sprint_9.md](../sprints/sprint_9.md). Previous: [9.3](sprint_9_3.md). Decision: report only.

## 0. Outcome in brief

- **Frozen DeepWalk is at chance** (DW0: Hits@10 0.0000, MRR 0.0007, F1 0.0000), as expected: NCIT and DOID share no edges, so coordinates are not comparable across graphs. MRR 0.0007 equals the Sprint 4 value.
- **A projection of DeepWalk cannot align it** (DW1 MRR 0.0024, DW1-I 0.0027): a per-node map of non-comparable coordinates has nothing to align.
- **The GCN of DeepWalk can, but only with InfoNCE:** DW2-I reaches Hits@10 0.3472, MRR 0.1552, F1 0.0794, with a structure-only encoder at inference. DW2 (triplet) is MRR 0.0306.
- **Control:** DW2-own (DeepWalk's own pairs) is at chance (MRR 0.0007), so the DW2-I signal comes from the SapBERT-derived pairs, not from structure on its own.
- **Every DeepWalk row is far below frozen SapBERT** (MRR 0.4475, F1 0.4136); the best, DW2-I, is 0.29 MRR below.

## 1. What changed

- `gcl_train.inputs.text_features: text | graph` (default `text`; earlier runs unchanged). `graph` feeds the DeepWalk store to the text tower, so the encoder never sees text. Recorded in the manifest summary (`text_features`). Unknown values raise `ConfigError`. Changed: `src/cga/pipelines/gcl_train/pipeline.py`.
- `scripts/run_unsup_matrix.sh`: priority 4 (dw-u1, dw-u1-i, dw-u2, dw-u2-i) and priority 5 (dw-u2-own); `SAMPLER_PREFIX` names the sampler lineage the rows read. Priority 4 pins each seed to the `sap-rank_topk-k5` run built with that seed (`pinned_run`), not to LATEST.
- `scripts/unsup_summary.py`: `--rows deepwalk` row set (`DEEPWALK_ROWS`) for the paper table and `--compare`.
- `configs/experiments/ncit_doid_unsup_deepwalk_test.yaml`: `dw-u0` and the 15 models, same two evaluations as 9.2/9.3.
- Tests: `text_features` default / `graph` / invalid (gcl_train), deepwalk row set (summary). The gcl_train and summary tests pass.

## 2. Runs

- Priority 4: 12 runs (4 rows x 3 seeds), pairs from the 9.3 SapBERT lineage `sap-rank_topk-k5`. Priority 5: 3 `training_data` runs (`dw-rank_topk-k5`, embedder `deepwalk-deepwalk`) and 3 `dw-u2-own` runs. Fixed 9.2 settings; nothing tuned. Model ids `dw-u1`, `dw-u1-i`, `dw-u2`, `dw-u2-i`, `dw-u2-own` (+ `-s<seed>`). Training log: `logs/sprint_9_4_train.log`.
- Evaluation, run once: `20261026T155212Z-70d12bef` (`uv run cga run evaluate --dataset ncit_doid_unsup --experiment ncit_doid_unsup_deepwalk_test`); `summary.md` from `unsup_summary.py --prefix dw- --rows deepwalk --text-label DeepWalk --compare 20261026T131132Z-5d0dc63e --compare-prefix sap- --compare-label SapBERT`.

## 3. Paper table (NCIT–DOID test, 3,280 pairs, mean ± std over 3 seeds)

| Variant | Pseudo-pairs | Loss | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepWalk (frozen) | – | – | 0.0000 | 0.0000 | 0.0007 | 0.0000 | 0.0000 | 0.0000 | 2398.0 | 0.0 |
| DeepWalk + projection | SapBERT top-k | Triplet | 0.0000 ± 0.0000 | 0.0025 ± 0.0008 | 0.0024 ± 0.0002 | 0.0004 ± 0.0000 | 0.0003 ± 0.0000 | 0.0004 ± 0.0000 | 2234.3 ± 25.0 | 1.0 ± 0.0 |
| DeepWalk + projection | SapBERT top-k | InfoNCE | 0.0002 ± 0.0002 | 0.0033 ± 0.0004 | 0.0027 ± 0.0002 | 0.0003 ± 0.0002 | 0.0002 ± 0.0002 | 0.0002 ± 0.0002 | 2369.0 ± 13.1 | 0.7 ± 0.6 |
| GCN of DeepWalk | SapBERT top-k | Triplet | 0.0089 ± 0.0006 | 0.0651 ± 0.0018 | 0.0306 ± 0.0011 | 0.0096 ± 0.0043 | 0.0051 ± 0.0023 | 0.0067 ± 0.0030 | 1725.0 ± 30.0 | 16.7 ± 7.6 |
| GCN of DeepWalk | SapBERT top-k | InfoNCE | 0.0695 ± 0.0043 | 0.3472 ± 0.0058 | 0.1552 ± 0.0053 | 0.0983 ± 0.0065 | 0.0666 ± 0.0045 | 0.0794 ± 0.0053 | 2221.0 ± 7.8 | 218.3 ± 14.7 |
| GCN of DeepWalk | DeepWalk top-k | Triplet | 0.0000 ± 0.0000 | 0.0008 ± 0.0008 | 0.0007 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1689.0 ± 11.5 | 0.0 ± 0.0 |

Against SapBERT (9.3, same rows; MRR / F1): U0 0.4475 / 0.4136 vs DW0 0.0007 / 0.0000; U1 0.4276 / 0.3978 vs 0.0024 / 0.0004; U1-I 0.4129 / 0.3725 vs 0.0027 / 0.0002; U2 0.1899 / 0.1184 vs 0.0306 / 0.0067; U2-I 0.4088 / 0.3156 vs 0.1552 / 0.0794.

## 4. Checks (gate 9.4)

- DW0 is near chance (MRR 0.0007, Hits@10 0). It equals the Sprint 4 frozen-DeepWalk cross-graph MRR (0.0007). I did not run a separate standalone `embedding_store` evaluation: `dw-u0` is itself an `embedding_store` method on the same 3,280 pairs, so that check would be the same computation.
- DW2-own is near chance (MRR 0.0007).
- `text_features` default leaves earlier runs unchanged (unit test; the default code path is the old one).

## 5. Disclosures and caveats

- **Test read:** the gold test pairs were read once, by `evaluate`; no gold in training, pair building, thresholds or the sibling filter.
- **DW1/DW2 pairs come from SapBERT**, which was pretrained on UMLS synonyms (see the 9.3 caveat). The rows are structure-only at inference, not independent of text. DW2-own is the no-text control.
- **Transductive:** the pseudo-pairs include test-split nodes (every node is eligible, as in 9.2/9.3). DW2-I's result is therefore best read as SapBERT's neighbour structure distilled into DeepWalk-based embeddings of the same nodes, not as structure discovering alignments; the DW2-own control supports this.
- Gold-free ATT&CK–CWE is out of scope for this phase.

## 6. Commands

```
PREFIX=dw- SAMPLER_PREFIX=sap- bash scripts/run_unsup_matrix.sh ncit_doid_unsup 4
TEXT_EMBEDDER=deepwalk-deepwalk PREFIX=dw- bash scripts/run_unsup_matrix.sh ncit_doid_unsup 5
uv run cga run evaluate --dataset ncit_doid_unsup --experiment ncit_doid_unsup_deepwalk_test
```
