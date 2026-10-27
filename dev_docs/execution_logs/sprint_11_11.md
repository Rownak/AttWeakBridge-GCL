# Sprint 11.11: Dual-encoder GCN on chunked mpnet text vectors

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [11.10](sprint_11_10.md) (description-only texts, chunked embeddings), [11.6](sprint_11_6.md) (the SecRoBERTa dual GCNs).

## 0. Outcome in brief

- **Trained four dual-encoder GCNs** (seed 42, CPU, one thread, 100 epochs): triplet and InfoNCE on the chunked mpnet store (`description_only` text), and the same two losses on SecRoBERTa `description_only` as a same-text control. All finish with a lower loss; a rerun of one model gives bit-identical embeddings.
- **The mpnet InfoNCE dual is the first dual in this sprint that beats its own frozen text vectors.** Union, full view: Hits@10 0.1472 against 0.1056, Recall@10 0.1630 against 0.1471, F1 0.0208 against 0.0164 (8 against 6 correct pairs). Pool view: MRR 0.1490 against 0.1201, Recall@10 0.3620 against 0.2905, F1 0.0507 against 0.0289 (18 against 10 correct). It is above frozen in 5 of 6 ranking cells in every source and in 21 to 23 of 24 set cells (the one ranking cell where it is not above is Hits@1 in the full view, in every source; union 0.0208 against 0.0288).
- **The mpnet triplet dual does not:** below frozen mpnet in all four sources (union: 0 of 6 ranking cells, 0 of 24 set cells above; Recall@10 0.0810 against 0.1471).
- **Encoder gap after training:** the mpnet duals are above the same-text SecRoBERTa duals in every ranking cell and in 24 of 24 set cells, in all four sources and for both losses. Union Recall@10: mpnet InfoNCE 0.1630 against 0.0306 for the control (5.3x; frozen gap 0.1471 against 0.0367, 4.0x).
- **The SecRoBERTa duals do not beat frozen SecRoBERTa** in any clear way, on either text (details in section 5).
- **Loss:** InfoNCE is above triplet for mpnet in every source (ranking 5 or 6 of 6, set 24 of 24); for the SecRoBERTa control it is mixed.
- **Checks passed:** the old-text references (frozen SecRoBERTa, both 11.6 duals, random) reproduce the 11.7 rows exactly (360 rows per source, max difference 0.0); repeat evaluation identical (864 of 864 rows); `n_queries` equal across all methods.
- **Still low in absolute terms:** best union Precision@10 is 0.046 against a ceiling of 0.323.

## 1. Inputs

| Input | Run |
|---|---|
| mpnet, chunked, `description_only` | `sentence_transformer-multi_qa_mpnet_dot_chunk_desc/20261027T194928Z-8cf38e55` |
| SecRoBERTa, `description_only` (control) | `hf_mean_pool-pt_SecRoBERTa_desc/20261027T200841Z-09088f9f` |
| DeepWalk (built from the 11.6 ingest) | `deepwalk-deepwalk/20261027T135210Z-3b18ba0c` |
| Ingest (nodes, edges) | union `20261027T194643Z-5d3f8baf` (11.10) |

The 11.10 ingests have the same node and edge id sets as the 11.6 ones (checked in 11.10) and DeepWalk does not read text, so the DeepWalk store stays valid. Nothing was re-ingested or re-embedded in this phase.

## 2. Pseudo-pairs (`training_data`, `rank_topk`, k = 5, seed 42, ingest pinned)

| Text store | Sampler id | Run | Universe | Examples |
|---|---|---|---|---|
| mpnet chunked | `s11c-rank_topk-k5` | `20261027T205523Z-ac890ca3` | 2,927 (0 held out) | 14,635 |
| SecRoBERTa `description_only` | `s11d-rank_topk-k5` | `20261027T205531Z-9cbd6584` | 2,927 (0 held out) | 14,635 |

Manifest checks: protocol `unsupervised`, `held_out_policy: none`, split `train` for all examples, 2,927 anchors, 5 examples per anchor, seed 42, the text and DeepWalk run ids as pinned. The universe equals that of 11.1 and 11.6.

### Pseudo-pair diagnostics (read-only; the gold is not used in training)

| | mpnet chunked (`s11c`) | SecRoBERTa `description_only` (`s11d`) | SecRoBERTa old text (11.6, `s11b`) |
|---|---|---|---|
| Distinct cross-graph pairs | 13,828 | 14,069 | 14,084 |
| Of them gold pairs (union, 625 gold pairs lie in the universe) | 62 (0.45%) | 18 (0.13%) | 22 (0.16%) |
| Share of the gold pairs covered | 9.9% | 2.9% | 3.5% |
| ATT&CK-anchored rows (9,915): distinct CWE positives | 468 | 350 | 397 |
| Share taken by the five most frequent CWEs | 21.9% | 32.5% | 31.3% |
| Positives in CWE ids 1200 to 1399 (the pool has 12.8%) | 18.8% | 36.5% | 52.7% |
| Most frequent CWE positives (rows) | CWE-506 (590), CWE-494 (495), CWE-1242 (388), CWE-507 (387), CWE-509 (313) | CWE-1300 (1,047), CWE-57 (753), CWE-391 (625), CWE-1259 (407), CWE-382 (387) | CWE-1331 (708), CWE-57 (689), CWE-1248 (666), CWE-1270 (546), CWE-1267 (496) |
| Gold techniques with at least one pseudo pair | 183 of 186 | 183 of 186 | 183 of 186 |

The 3 techniques without a pair lie outside the 2,927-node universe. The mpnet pairs are less concentrated and less hardware-heavy, and contain about three times as many gold pairs as the SecRoBERTa pairs, but gold pairs are still under 0.5% of them. (The 11.6 log's "53%" used the prefix CWE-12xx/13xx; the 52.7% above uses ids 1200 to 1399, the definition used from 11.9 on.)

## 3. Training (`gcl_train`, `unsup_u2`, seed 42, `device=cpu`, `OMP_NUM_THREADS=1`)

| Model id | Run | Loss epoch 1 | Loss epoch 100 |
|---|---|---|---|
| `ent-dual-gcn-triplet-mpnetc-s42` | `20261027T205615Z-db28e05d` | 0.7879 | 0.1397 |
| `ent-dual-gcn-infonce-mpnetc-s42` | `20261027T205648Z-2aeb3259` | 8.8549 | 5.7995 |
| `ent-dual-gcn-triplet-srbd-s42` | `20261027T205823Z-3a7bc511` | 0.8075 | 0.1281 |
| `ent-dual-gcn-infonce-srbd-s42` | `20261027T205855Z-a209bd62` | 8.9782 | 5.6841 |

- Output in `models/gcl/attack_cwe_ent/<model id>/<run id>/`. Each has 100 rows in `metrics.jsonl`, checkpoints at epochs 50 and 100, and `node_embeddings` of shape 2,927 x 128 with an index of 2,927 rows and no NaN. All manifests show 14,635 examples, 2,927 nodes, the pinned `training_data_run_id`, text run id and DeepWalk run id, and the same `universe_sha256` for the mpnet pair (`b11f6656e300…`).
- **Input dimensions** (checkpoint of the mpnet triplet model): the text GCN tower is 768 to 128 to 128, the graph GCN tower 128 to 128 to 128 (DeepWalk dimension 128), and the fusion layers take 512 inputs; 328,576 trainable parameters, the same for all four models and for the 11.6 duals. The plan's "768 + DeepWalk dimension" was wrong: the two towers are separate.
- **Determinism:** a rerun of the mpnet triplet model (`20261027T210042Z-db28e05d`, same seed) gives node embeddings identical to the first run (same index, max difference 0.0). `LATEST` of that model id points to the rerun; the evaluation config pins the first run.

## 4. Evaluation

Config `configs/experiments/attack_cwe_ent_mpnet_dual_eval.yaml`: the two mpnet-chunk duals and the frozen mpnet-chunk store; the control (frozen SecRoBERTa `description_only` and its two duals); the old-text references (`ent-s-r1`, `ent-d-trip` = `…-s42b` run `20261027T141524Z-5ac0fba7`, `ent-d-nce` = `…-s42b` run `20261027T141732Z-3f70b1a2`); `random-rand42/43/44`. All pinned by run id; the six evaluations of 11.9; cosine only; CSLS not applicable (as in 11.9).

| Gold source | 11.10 ingest | Evaluate run | Techniques scored (set) | Gold pairs scored |
|---|---|---|---|---|
| stix | `20261027T194605Z-44eeb7c4` | `20261027T210217Z-97f436c6` | 93 | 284 |
| capec | `20261027T194618Z-cba9bfdb` | `20261027T210227Z-60f05dc0` | 174 | 566 |
| intersection | `20261027T194631Z-950ba779` | `20261027T210240Z-a579bc8b` | 78 | 221 |
| union | `20261027T194643Z-5d3f8baf` | `20261027T210250Z-62e530e2` | 183 | 625 |

The duals only cover the 2,927 universe nodes, so, as in 11.7, 3 techniques (stix, union) and 3 pairs are not scored for any method, and the candidate set is the 944 CWEs in the universe. Values here are therefore comparable within this run, not with the 11.9 / 11.10 tables (186 techniques, 969 CWEs); the frozen mpnet-chunk row in this run is the like-for-like reference.

### Checks

- **Reproduction:** the old-text references (frozen SecRoBERTa, both 11.6 duals, random) equal the 11.7 runs (`…143203Z-359b85eb`, `…143214Z-460087ac`, `…143225Z-d9a8711e`, `…143236Z-91f56f41`) on the 360 rows per source that exist there (`gold_full`, `prf_full`, `set_full`, `set_pool`; the pool pair evaluations were not in 11.7): max difference 0.0, equal `n_queries`. The universe of the new duals is the same 2,927 nodes, so the candidate sets match.
- **Repeat:** union rerun `20261027T210302Z-62e530e2`: 864 of 864 rows identical.
- **Query set:** `n_queries` equal across methods in every evaluation (stix 284 / 287 / 93, capec 566 / 174, intersection 221 / 78, union 625 / 628 / 183 for ranking / assignment / set).
- **Tables** in `artifacts/evaluations/attack_cwe_ent/20261027T210250Z-62e530e2/`: `sources_summary_mpnetdual_full.md`, `sources_summary_mpnetdual_pool.md`, `sources_summary_mpnetdual_set_full.md`, `sources_summary_mpnetdual_set_pool.md`.

### Union results

**Ranking (per gold pair):**

| Method | MRR full | Hits@1 full | Hits@10 full | MRR pool | Hits@1 pool | Hits@10 pool |
|---|---|---|---|---|---|---|
| mpnet chunked, frozen | 0.0606 | 0.0288 | 0.1056 | 0.1201 | 0.0544 | 0.2496 |
| Dual, mpnet chunked, triplet | 0.0360 | 0.0096 | 0.0816 | 0.0938 | 0.0272 | 0.2192 |
| Dual, mpnet chunked, InfoNCE | 0.0640 | 0.0208 | 0.1472 | 0.1490 | 0.0704 | 0.3024 |
| SecRoBERTa `description_only`, frozen | 0.0160 | 0.0032 | 0.0288 | 0.0592 | 0.0144 | 0.1232 |
| Dual, control, triplet | 0.0209 | 0.0048 | 0.0432 | 0.0698 | 0.0208 | 0.1472 |
| Dual, control, InfoNCE | 0.0138 | 0.0016 | 0.0272 | 0.0601 | 0.0144 | 0.1408 |
| SecRoBERTa 11.9 text, frozen | 0.0247 | 0.0096 | 0.0368 | 0.0572 | 0.0160 | 0.1072 |
| Dual, 11.6, triplet | 0.0185 | 0.0048 | 0.0256 | 0.0591 | 0.0176 | 0.1088 |
| Dual, 11.6, InfoNCE | 0.0203 | 0.0080 | 0.0240 | 0.0523 | 0.0144 | 0.1056 |
| Random (3 seeds) | 0.0070 | 0.0000 | 0.0101 | 0.0378 | 0.0048 | 0.0699 |

**Assignment (top-3, greedy one-to-one), F1 with correct pairs of 628:** full view / pool view: mpnet frozen 0.0164 (6) / 0.0289 (10); mpnet triplet 0.0027 (1) / 0.0142 (5); mpnet InfoNCE 0.0208 (8) / 0.0507 (18); control frozen 0.0054 (2) / 0.0087 (3); control triplet 0.0027 (1) / 0.0116 (4); control InfoNCE 0.0052 (2) / 0.0115 (4); SecRoBERTa 11.9 text frozen 0.0246 (9) / 0.0206 (7); 11.6 triplet 0.0054 (2) / 0.0145 (5); 11.6 InfoNCE 0.0107 (4) / 0.0087 (3). Predicted counts, precision and recall for every source are in the tables. Each cell rests on 1 to 18 correct pairs.

**Set-based (per technique), k = 10 (precision ceiling 0.323), full view / pool view:**

| Method | Hits@10 | Recall@10 | Precision@10 | Hits@10 pool | Recall@10 pool | Precision@10 pool |
|---|---|---|---|---|---|---|
| mpnet chunked, frozen | 0.2951 | 0.1471 | 0.0355 | 0.5191 | 0.2905 | 0.0749 |
| Dual, mpnet chunked, triplet | 0.1913 | 0.0810 | 0.0268 | 0.4372 | 0.2376 | 0.0683 |
| Dual, mpnet chunked, InfoNCE | 0.3443 | 0.1630 | 0.0459 | 0.5956 | 0.3620 | 0.0951 |
| SecRoBERTa `description_only`, frozen | 0.0984 | 0.0367 | 0.0098 | 0.3497 | 0.1632 | 0.0399 |
| Dual, control, triplet | 0.0984 | 0.0260 | 0.0148 | 0.2678 | 0.1032 | 0.0426 |
| Dual, control, InfoNCE | 0.0656 | 0.0306 | 0.0082 | 0.2350 | 0.1196 | 0.0421 |
| SecRoBERTa 11.9 text, frozen | 0.1202 | 0.0663 | 0.0126 | 0.2842 | 0.1670 | 0.0355 |
| Dual, 11.6, triplet | 0.0656 | 0.0158 | 0.0087 | 0.2404 | 0.1046 | 0.0350 |
| Dual, 11.6, InfoNCE | 0.0765 | 0.0360 | 0.0082 | 0.2568 | 0.1271 | 0.0333 |

Recall@10 lift over random, full view (stix / capec / intersection / union): mpnet frozen 17.6x / 11.6x / 18.5x / 11.5x; mpnet InfoNCE 24.7x / 12.9x / 26.0x / 12.8x; mpnet triplet 14.6x / 6.5x / 14.2x / 6.3x; control frozen 5.3x / 2.8x / 5.1x / 2.9x; control triplet 5.9x / 1.5x / 3.3x / 2.0x; control InfoNCE 5.1x / 2.5x / 5.6x / 2.4x. Pool view, union: mpnet frozen 4.1x, InfoNCE 5.1x, triplet 3.4x, control frozen 2.3x, control triplet 1.5x, control InfoNCE 1.7x.

## 5. Report (no tuning)

"Above" counts are of 6 ranking cells (MRR, Hits@1, Hits@10 in the full and pool views) or of 24 set cells (Hits, Recall, Precision at k = 5, 10, 15, 20 in both views) per source, in the order stix / capec / intersection / union; ties are not counted as above.

- **Dual against frozen (gain from training = dual minus frozen), union:**

| Dual (against its frozen row) | MRR full | Hits@10 full | Recall@10 full | MRR pool | Recall@10 pool | F1 full |
|---|---|---|---|---|---|---|
| mpnet InfoNCE | +0.0035 | +0.0416 | +0.0159 | +0.0289 | +0.0715 | +0.0044 |
| mpnet triplet | -0.0246 | -0.0240 | -0.0661 | -0.0263 | -0.0528 | -0.0137 |
| control InfoNCE | -0.0023 | -0.0016 | -0.0061 | +0.0009 | -0.0436 | -0.0002 |
| control triplet | +0.0049 | +0.0144 | -0.0107 | +0.0106 | -0.0600 | -0.0027 |
| 11.6 InfoNCE | -0.0044 | -0.0128 | -0.0303 | -0.0049 | -0.0399 | -0.0140 |
| 11.6 triplet | -0.0061 | -0.0112 | -0.0505 | +0.0019 | -0.0624 | -0.0193 |

  - **mpnet InfoNCE** is above frozen mpnet in 5 / 5 / 5 / 5 of 6 ranking cells and in 23 / 21 / 21 / 21 of 24 set cells.
  - **mpnet triplet** is above in 3 / 0 / 1 / 0 of 6 ranking cells and 12 / 0 / 9 / 0 of 24 set cells, so it is above frozen in some cells only in stix and intersection (the two smallest sets) and nowhere in capec and union.
  - **Control duals** against frozen SecRoBERTa `description_only`: InfoNCE is above in 6 / 3 / 6 / 2 of 6 ranking cells and 17 / 4 / 21 / 5 of 24 set cells; triplet in 5 / 6 / 4 / 6 of 6 and 21 / 7 / 14 / 9 of 24. So the control duals gain in ranking quality at the top but not in Recall in the larger sets (capec, union).
  - **11.6 duals** against frozen SecRoBERTa: InfoNCE is above in 4 / 0 / 4 / 0 of 6 ranking cells and 12 / 2 / 12 / 2 of 24 set cells; triplet in 0 / 2 / 0 / 3 and 2 / 2 / 1 / 2.
- **mpnet duals against the same-text control duals:** above in 6 of 6 ranking cells and 24 of 24 set cells in every source, for both losses. The encoder gap grows after InfoNCE training (union Recall@10 frozen 0.1471 against 0.0367, 4.0x; InfoNCE 0.1630 against 0.0306, 5.3x) and shrinks after triplet training (0.0810 against 0.0260, 3.1x).
- **Effect of the text on the SecRoBERTa duals** (control against the 11.6 duals): triplet is above in 5 / 4 / 4 / 5 of 6 ranking cells and 23 / 17 / 21 / 19 of 24 set cells; InfoNCE is above in 1 / 3 / 1 / 3 of 6 and 6 / 6 / 4 / 6 of 24 (with 1 tie per source), so the shorter text helps the triplet dual and not the InfoNCE dual. I did not test why.
- **Which loss is higher:** for mpnet, InfoNCE is above triplet in 5 / 6 / 5 / 6 of 6 ranking cells and in 24 of 24 set cells in every source. For the control, InfoNCE is above triplet in 2 / 0 / 6 / 0 of 6 and 9 / 10 / 19 / 10 of 24; for the 11.6 duals in 6 / 3 / 6 / 2 and 23 / 18 / 24 / 18.
- **Hub check** (union, 183 techniques, 944 CWEs, top-10 lists; hardware range = CWE ids 1200 to 1399, pool share 13.0%). The probe's Recall@10 equals the evaluator value for every row.

| Store or model | Recall@10 | Distinct CWEs | Hardware-range share of slots | Gold CWEs reached | Five most frequent CWEs (lists of 183) |
|---|---|---|---|---|---|
| mpnet chunked, frozen | 0.1471 | 277 | 22.8% | 29 | CWE-509 (63), CWE-1278 (54), CWE-537 (54), CWE-114 (53), CWE-1334 (53) |
| Dual, mpnet chunked, InfoNCE | 0.1630 | 377 | 20.5% | 40 | CWE-114 (39), CWE-1273 (29), CWE-520 (28), CWE-379 (28), CWE-522 (28) |
| Dual, mpnet chunked, triplet | 0.0810 | 339 | 18.1% | 29 | CWE-1273 (44), CWE-202 (41), CWE-612 (36), CWE-507 (33), CWE-359 (33) |
| Dual, control, InfoNCE | 0.0306 | 358 | 30.8% | 10 | CWE-1299 (46), CWE-1426 (40), CWE-1300 (39), CWE-1259 (35), CWE-83 (34) |
| Dual, control, triplet | 0.0260 | 285 | 23.3% | 17 | CWE-1189 (40), CWE-1220 (39), CWE-1255 (37), CWE-647 (35), CWE-57 (34) |
| Dual, 11.6, InfoNCE | 0.0360 | 318 | 59.1% | 11 | CWE-1311 (60), CWE-1267 (59), CWE-1294 (57), CWE-1292 (56), CWE-1290 (54) |
| Dual, 11.6, triplet | 0.0158 | 290 | 38.7% | 12 | CWE-1292 (54), CWE-1317 (51), CWE-1290 (44), CWE-1302 (43), CWE-1316 (35) |

  The mpnet InfoNCE dual spreads its top-10 lists over more CWEs (377, the largest count), and its most frequent CWE is in 39 of 183 lists (the highest for the other models is 63); it reaches 40 distinct gold CWEs against 29 for frozen mpnet. The hardware / SoC hubs stay strong in the SecRoBERTa-based duals (59% of slots for the 11.6 InfoNCE dual). I did not check why the mpnet InfoNCE dual spreads wider.
- **Absolute level is still low:** the best Precision@10 is 0.046 (ceiling 0.323), Recall@10 0.163 (0.362 in the optimistic pool view).

## 6. Limits

- Single seed (42) per model, and one training-data draw per text store; the differences between duals of a few thousandths (for example MRR +0.0035 for mpnet InfoNCE) are not tested against seed variation. The larger gains (Hits@10 +0.042, pool Recall@10 +0.072, F1 with 8 against 6 and 18 against 10 correct pairs) are consistent across sources but come from one run.
- The mpnet and control models differ in encoder only (same text, same recipe), but the pseudo-pairs differ with the encoder. The 11.6 duals differ from the control in text only, for SecRoBERTa.
- The duals cover 2,927 nodes, so 3 gold techniques (stix, union) are not scored for any method and the candidate set is 944 CWEs.
- The pseudo-pairs contain at most 0.45% gold pairs and are the text encoder's own top-5; the gold plays no part in training. Gold is transitive through CAPEC; the pool is built from gold and is optimistic; assignment F1 rests on few correct pairs.
- CPU one-thread training is deterministic here (rerun identical); GPU or multi-thread runs would differ slightly (11.6).
- All duals are trained on the same universe and recipe with no tuning; a different margin, temperature, k or epoch count was not tried.
- Old stores, ingests and models are kept; the `LATEST` pointer of `ent-dual-gcn-triplet-mpnetc-s42` is the rerun (identical to the first run).

## 7. Files

- Modified: `scripts/unsup_summary.py` (row set `attack_ent_mpnet_dual`), `dev_docs/sprints/sprint_11.md`.
- New: `configs/experiments/attack_cwe_ent_mpnet_dual_eval.yaml`, this log. No pipeline code was changed in this phase.

## 8. Commands

```
ING=20261027T194643Z-5d3f8baf; DW=20261027T135210Z-3b18ba0c
uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
  --set training_data.inputs.ingest_run=$ING \
  --set training_data.inputs.text_embeddings.embedder_id=sentence_transformer-multi_qa_mpnet_dot_chunk_desc \
  --set training_data.inputs.text_embeddings.run=20261027T194928Z-8cf38e55 \
  --set training_data.inputs.graph_embeddings.run=$DW --set training_data.sampler.sampler_id=s11c-rank_topk-k5
# control: embedder_id=hf_mean_pool-pt_SecRoBERTa_desc, text run 20261027T200841Z-09088f9f, sampler id s11d-rank_topk-k5
RUN=<training_data run id>   # 20261027T205523Z-ac890ca3 (s11c) or 20261027T205531Z-9cbd6584 (s11d)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
  --set seed=42 --set device=cpu --set gcl_train.model.model_id=ent-dual-gcn-<loss>-<mpnetc|srbd>-s42 \
  --set gcl_train.inputs.training_data_sampler_id=<s11c|s11d>-rank_topk-k5 \
  --set gcl_train.inputs.training_data_run=$RUN --set gcl_train.loss.name=<triplet|infonce>
for each gold source (11.10 ingest run from section 4):
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_mpnet_dual_eval \
    --set evaluate.inputs.ingest_run=<ingest run of that source>
U=20261027T210250Z-62e530e2
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_mpnet_dual \
  --ranking-eval gold_<full|pool> --prf-eval prf_<full|pool> --lift --out-name sources_summary_mpnetdual_<full|pool>.md \
  --source-run stix=20261027T210217Z-97f436c6 --source-run capec=20261027T210227Z-60f05dc0 \
  --source-run intersection=20261027T210240Z-a579bc8b --source-run union=$U
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_mpnet_dual \
  --set-eval set_<full|pool> --out-name sources_summary_mpnetdual_set_<full|pool>.md <the same --source-run options>
uv run pytest tests/unit -q
```
