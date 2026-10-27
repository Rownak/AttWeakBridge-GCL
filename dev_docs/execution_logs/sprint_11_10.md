# Sprint 11.10: Description-only texts and chunked sentence embeddings

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [11.9](sprint_11_9.md) (truncating sentence encoders on the `description` text).

## 0. Outcome in brief

- **Text:** new variant `description_only` (ATT&CK `description`, CWE `description`; no CWE name, no extended description). Four re-ingests; nodes, edges and gold pairs are identical to the 11.6 ingests and the `description` / `full_metadata` texts are unchanged. Only the CWE text changes: median 377 to 152 characters. ATT&CK text is the same as before.
- **Chunking:** `SentenceTransformerEmbedder` has a `chunking` option (token-weighted mean of chunk vectors). It matters for few nodes: 116 ATT&CK nodes (111 with 2 chunks, 5 with 3; 105 are techniques or sub-techniques) and no CWE. In the union gold, 28 of 186 scored techniques are chunked.
- **Effect of chunking** (mpnet, same text): small and mostly positive. The top-10 list changes for 18 of those 28 techniques and for none of the other 158; union MRR 0.0603 against 0.0587, Recall@10 0.1447 against 0.1429.
- **Effect of the text:** mpnet changes little (union MRR 0.0587 against 0.0585, Recall@10 0.1429 against 0.1344, but Hits@10 0.1035 against 0.1178); SecRoBERTa loses a lot (union MRR 0.0158 against 0.0244, Recall@10 0.0361 against 0.0652); bge-base improves at the top ranks (union MRR 0.0626 against 0.0501, Hits@1 0.0414 against 0.0207) and loses some Recall@10 (0.1124 against 0.1196). The comparisons are in section 5; the direction is not the same for the three encoders.
- **Order on the new text:** both sentence encoders are above SecRoBERTa in every ranking and set cell. mpnet and bge split: bge is above mpnet on MRR and Hits@1 in the full view (4 of 4 sources), mpnet is above bge on Hits@10 (4 of 4) and on 20 to 24 of 24 set cells.
- **Bug caught on the way (fixed):** the base `embed.yaml` carries the legacy string `chunking: fixed`; my first `mpnet-trunc` run inherited it and was chunked. The checks showed it (section 3). Fixed with a strict `is True` test, a regression test and explicit `chunking: false` in the two 11.9 configs. The 11.9 stores are not affected.
- **Checks passed:** repeat embed identical; the 11.9 stores, scored on the new ingests, reproduce the 11.9 rows exactly (432 per source); repeat evaluation identical; `n_queries` equal across methods. 595 unit tests pass.
- **Still low in absolute terms:** best union Precision@10 is 0.035 against a ceiling of 0.319.

## 1. Code and configs

- `configs/datasets/attack_cwe_ent.yaml`: `description_only: {attack: [description], cwe: [description]}`; the other variants are unchanged.
- `src/cga/pipelines/embed/text_embedders.py`: `SentenceTransformerEmbedder(chunking=False)`. With `chunking=True`, `_embed_chunked` tokenizes with the model's tokenizer without truncation, splits into chunks of `max_seq_length` minus the special tokens (510 here), wraps each chunk with the tokenizer's special tokens and runs the model on token ids (no decode and re-tokenise), then returns the mean of the chunk vectors weighted by the number of content tokens, and the number of chunks. The special-token wrapper is read from the tokenizer (`build_inputs_with_special_tokens` no longer exists in transformers 5). A text that fits one chunk gives the unchunked vector (max difference 1.2e-6, padding noise).
- `src/cga/pipelines/embed/pipeline.py`: passes `chunking=embedder_cfg.get("chunking") is True`.
- `scripts/unsup_summary.py`: row set `attack_ent_desc`.
- Configs: `configs/pipelines/embed_mpnet_chunk.yaml`, `embed_bge_chunk.yaml` (`description_only`, `chunking: true`, `batch_size: 32`); `embed_mpnet.yaml` and `embed_bge.yaml` now say `chunking: false` explicitly; `configs/experiments/attack_cwe_ent_desc_eval.yaml`.
- Tests (+4, 595 pass): chunking (one chunk equals the unchunked vector; long text gives the token-weighted mean of its chunks and differs from the truncated vector; empty text stays a zero vector) and the pipeline option (`"fixed"` and `false` do not chunk, `true` does).
- Decisions as in the plan: the CWE name is excluded; chunk vectors are weighted by token count (SecRoBERTa's chunks are averaged without weights); ATT&CK text is not cleaned.

## 2. Re-ingest

| Gold source | 11.6 ingest | New ingest | Gold pairs |
|---|---|---|---|
| stix | `20261027T134239Z-8101c66c` | `20261027T194605Z-44eeb7c4` | 287 |
| capec | `20261027T134252Z-cbfb9a99` | `20261027T194618Z-cba9bfdb` | 566 |
| intersection | `20261027T134304Z-d983eac7` | `20261027T194631Z-950ba779` | 221 |
| union (`LATEST`) | `20261027T134320Z-32a16662` | `20261027T194643Z-5d3f8baf` | 628 |

- Checked per source: node ids (5,840), edge ids (22,979) and alignment ids are the same sets as in the 11.6 ingests; the `description` and `full_metadata` texts are identical; `texts.parquet` has 17,520 rows (3 variants).
- `description_only` (union ingest, nodes with text): ATT&CK 4,871 nodes, 709 empty (same as before), 0 `N/A`, text identical to `description` for all 4,871 (median 273 and mean 501 characters). CWE 969 nodes, no empty or `N/A`; median 377 to 152 characters and mean 517 to 161 against `description`; the old text began with the CWE name for all 969. Example: CWE-284 is now "The product does not restrict or incorrectly restricts access to a resource from an unauthorized actor."

## 3. Embedding (GPU: RTX 4090 Laptop, `torch 2.6.0+cu124`, `sentence-transformers 6.2.0`)

| Store | Embedder id | Run | Wall time |
|---|---|---|---|
| mpnet, chunked | `sentence_transformer-multi_qa_mpnet_dot_chunk_desc` | `20261027T194928Z-8cf38e55` (repeat `…201513Z-8cf38e55`) | 365 s |
| bge-base, chunked | `sentence_transformer-bge_base_en_v1_5_chunk_desc` | `20261027T195544Z-b50c75b4` | 361 s |
| mpnet, truncated | `sentence_transformer-multi_qa_mpnet_dot_trunc_desc` | `20261027T202343Z-1fe4eb2c` | 394 s |
| SecRoBERTa, `description_only` | `hf_mean_pool-pt_SecRoBERTa_desc` | `20261027T200841Z-09088f9f` | 378 s |

The truncated run uses `embed_mpnet.yaml` with `text_variant=description_only` and its own label; SecRoBERTa uses `embed.yaml` with `text_variant=description_only`.

- **Per store:** 5,131 x 768; the same node set as the new union ingest (nodes with text); 5,131 ok, 709 empty text, 0 failed; no NaN; no zero rows. Norms: mpnet 5.3 to 7.1 (un-normalised), bge 0.92 to 1.00 (a weighted mean of unit chunk vectors is not a unit vector, so chunked bge vectors are slightly shorter than 1 where a text has several chunks), SecRoBERTa 11.2 to 26.1.
- **Chunks:** mpnet and bge give the same distribution: 5,015 nodes with 1 chunk, 111 with 2, 5 with 3. All 116 multi-chunk nodes are ATT&CK (105 are `attack-pattern`); no CWE. SecRoBERTa has 79 nodes with more than one chunk (82 on the old text).
- **Chunked against truncated mpnet:** single-chunk nodes differ by at most 1.2e-6; the 116 multi-chunk nodes differ by 0.0031 to 0.349 (min / max of the largest component difference; median 0.100).
- **Repeat** (mpnet, chunked): same index, max difference 0.0.
- **Text-only checks:** `mpnet-trunc` against the 11.9 mpnet store: ATT&CK rows (same text) differ by at most 1.1e-6 (a different batch composition, because the CWE texts changed), all 969 CWE rows changed. SecRoBERTa: ATT&CK rows identical (0.0), all 969 CWE rows changed.
- **Bug found by these checks.** My first `mpnet-trunc` run was identical to the chunked store on all 5,131 rows and differed from the 11.9 store on ATT&CK rows by 0.35; its `n_chunks` showed 2 and 3. Cause: the base `configs/pipelines/embed.yaml` has the legacy string `chunking: fixed` (for `hf_mean_pool`), it merges into every embed config, and my first version read it as `bool("fixed") = True`. Fix: `chunking is True` only, plus the regression test and an explicit `chunking: false` in `embed_mpnet.yaml` and `embed_bge.yaml`. I deleted the wrong run directory and reran. The 11.9 stores were made before the option existed and have `n_chunks = 1` for every node, so they are unchanged.

## 4. Evaluation

Config `configs/experiments/attack_cwe_ent_desc_eval.yaml`: the four new stores; the 11.9 stores on the `description` text as references (`ent-mpnet`, `ent-bge`, `ent-s-r1`); `random-rand42/43/44`; the six evaluations of 11.9. All stores pinned by run id; cosine only; CSLS is not applicable (as in 11.9).

| Gold source | Evaluate run | Techniques scored (set) |
|---|---|---|
| stix | `20261027T203056Z-e2e6fcce` | 96 |
| capec | `20261027T203109Z-a12d8b7a` | 174 |
| intersection | `20261027T203121Z-960dbd04` | 78 |
| union | `20261027T203132Z-519bead2` | 186 |

- **Reproduction:** the 11.9 reference rows (`ent-mpnet`, `ent-bge`, `ent-s-r1` and the random rows; 432 rows per source) equal the 11.9 runs (`…191838Z-849137da`, `…191857Z-5c78e309`, `…191914Z-766d74b3`, `…191929Z-006fe93d`): max difference 0.0, equal `n_queries`, no unmatched row.
- **Repeat:** union rerun `20261027T203145Z-519bead2`: 720 of 720 rows identical.
- **Query set:** `n_queries` equal across methods in every evaluation (same numbers as 11.9).
- **Tables** in `artifacts/evaluations/attack_cwe_ent/20261027T203132Z-519bead2/`: `sources_summary_desc_full.md`, `sources_summary_desc_pool.md`, `sources_summary_desc_set_full.md`, `sources_summary_desc_set_pool.md`.

### Union results (union gold; the other sources are in the tables)

**Ranking (per gold pair):**

| Method | MRR full | Hits@1 full | Hits@10 full | MRR pool | Hits@1 pool | Hits@10 pool |
|---|---|---|---|---|---|---|
| mpnet, chunked (description only) | 0.0603 | 0.0287 | 0.1051 | 0.1198 | 0.0541 | 0.2484 |
| bge-base, chunked (description only) | 0.0626 | 0.0414 | 0.0892 | 0.1154 | 0.0605 | 0.2054 |
| mpnet, truncated (description only) | 0.0587 | 0.0271 | 0.1035 | 0.1189 | 0.0541 | 0.2468 |
| SecRoBERTa (description only) | 0.0158 | 0.0032 | 0.0287 | 0.0590 | 0.0143 | 0.1226 |
| mpnet, truncated (11.9 text) | 0.0585 | 0.0239 | 0.1178 | 0.1216 | 0.0462 | 0.2611 |
| bge-base, truncated (11.9 text) | 0.0501 | 0.0207 | 0.0908 | 0.1056 | 0.0414 | 0.2118 |
| SecRoBERTa (11.9 text) | 0.0244 | 0.0096 | 0.0366 | 0.0571 | 0.0159 | 0.1067 |
| Random | 0.0068 | 0.0000 | 0.0096 | 0.0376 | 0.0048 | 0.0695 |

**Assignment (top-3, greedy one-to-one):** F1 (correct pairs of 628), full view / pool view: mpnet chunked 0.0164 (6) / 0.0289 (10); bge chunked 0.0193 (7) / 0.0291 (10); mpnet truncated 0.0165 (6) / 0.0289 (10); SecRoBERTa new 0.0054 (2) / 0.0087 (3); 11.9 text: mpnet 0.0319 (12) / 0.0403 (14), bge 0.0269 (10) / 0.0230 (8), SecRoBERTa 0.0246 (9) / 0.0206 (7). Predicted counts for the new stores are 96 to 109 (full view) and 59 to 65 (pool view); precision, recall and counts for every source are in the tables. Each cell rests on 2 to 14 correct pairs.

**Set-based (per technique), Recall@10 and Precision@10 (ceiling 0.319), full view / pool view:**

| Method | Hits@10 | Recall@10 | Precision@10 | Hits@10 pool | Recall@10 pool | Precision@10 pool |
|---|---|---|---|---|---|---|
| mpnet, chunked | 0.2903 | 0.1447 | 0.0349 | 0.5108 | 0.2858 | 0.0737 |
| bge-base, chunked | 0.2634 | 0.1124 | 0.0296 | 0.4301 | 0.2262 | 0.0634 |
| mpnet, truncated | 0.2849 | 0.1429 | 0.0344 | 0.5054 | 0.2836 | 0.0737 |
| SecRoBERTa (description only) | 0.0968 | 0.0361 | 0.0097 | 0.3441 | 0.1606 | 0.0392 |
| mpnet (11.9 text) | 0.2957 | 0.1344 | 0.0382 | 0.5269 | 0.2995 | 0.0817 |
| bge-base (11.9 text) | 0.2419 | 0.1196 | 0.0296 | 0.4301 | 0.2253 | 0.0672 |
| SecRoBERTa (11.9 text) | 0.1183 | 0.0652 | 0.0124 | 0.2796 | 0.1643 | 0.0349 |

Recall@10 lift over random, full view (stix / capec / intersection / union): mpnet chunked 19.5x / 12.0x / 20.9x / 11.9x, bge chunked 12.2x / 8.9x / 11.2x / 9.2x, SecRoBERTa new 5.9x / 2.8x / 5.7x / 3.0x. Pool view: mpnet chunked 3.3x / 4.1x / 3.0x / 4.1x, bge chunked 2.5x / 3.3x / 2.4x / 3.3x, SecRoBERTa new 1.5x / 2.5x / 1.5x / 2.3x.

## 5. Report (no tuning)

Counts are "above" counts of 4 sources (ranking: MRR, Hits@1, Hits@10 in each view) or of 24 set cells per source (Hits, Recall, Precision at k = 5, 10, 15, 20 in both views); a tie is not counted as above.

- **Effect of chunking** (`mpnet-chunk` against `mpnet-trunc`, same text). Small. Chunking changes the vectors of 116 nodes, of which 28 are scored techniques in the union gold; the top-10 list changes for 18 of those 28 and for 0 of the other 158; Recall@10 on the 28 is 0.2341 against 0.2222 (one technique up, none down). MRR is above in 4 of 4 sources in both views and Hits@1 in 4 of 4 (full view) and 2 of 4 (pool); Hits@10 is above in 2 of 4 (not above in the other two: ties in the full view). Set cells: above in 14, tied in 8, below in 2 of 24 for capec and union; 6 / 12 / 6 for stix and intersection. For bge, chunking and the text change come together (next point), so the chunk effect alone is only measured for mpnet.
- **Effect of the text** (description only against the 11.9 `description` text).
  - mpnet (truncated both times): union MRR 0.0587 against 0.0585 and Hits@1 0.0271 against 0.0239; Hits@10 is lower in all 4 sources (full view; 0.1035 against 0.1178 in union) and pool MRR is lower in all 4. Set cells above in 7 to 11 of 24 and below in 13 to 14. A small loss at the lower ranks.
  - SecRoBERTa: clearly worse in the full view (MRR and Hits@1 and Hits@10 above in 0 of 4; union MRR 0.0158 against 0.0244, Recall@10 0.0361 against 0.0652; set cells above in 1 to 8 of 24). In the pool view Hits@10 is above in 3 of 4 sources, MRR in 1 of 4.
  - bge-base (chunking and text together): MRR above in 4 of 4 and Hits@1 above in 4 of 4, in both views (union MRR 0.0626 against 0.0501, Hits@1 0.0414 against 0.0207); Hits@10 above in 2 of 4. The set cells are mixed: above in 22 of 24 (stix), 17 (intersection), 8 (union) and only 4 (capec); union Recall@10 is lower (0.1124 against 0.1196).
  - So removing the CWE name and extended description is not neutral, and not in one direction: the name and extended text help SecRoBERTa clearly, help mpnet slightly, and for bge the shorter text puts the gold CWE nearer the top rank but reaches fewer of a technique's CWEs in the top 10. I did not test why.
- **Order on the new text.** `mpnet-chunk` and `bge-chunk` are both above `srb-new` in every ranking cell and in 24 of 24 set cells in every source. Between the two sentence encoders the picture is split: bge-chunk is above mpnet-chunk on MRR (0 of 4 sources for mpnet in the full view) and Hits@1 (0 of 4), and below on Hits@10 (mpnet above in 4 of 4, both views); mpnet-chunk is above in 20 to 24 of 24 set cells (capec 24, union 22, intersection 21, stix 20). In the pool view MRR is above for mpnet in 2 of 4.
- **Hubs** (union, 186 techniques, top-10 lists, hardware range = CWE ids 1200 to 1399 as in 11.9):

| Store | Recall@10 | Distinct CWEs | Hardware-range share of slots | Gold CWEs reached | Five most frequent CWEs (lists) |
|---|---|---|---|---|---|
| mpnet, chunked | 0.1447 | 281 | 22.7% | 29 | CWE-509 (63), CWE-114 (54), CWE-1278 (54), CWE-537 (54), CWE-1334 (53) |
| bge-base, chunked | 0.1124 | 219 | 11.6% | 22 | CWE-57 (106), CWE-65 (84), CWE-62 (82), CWE-61 (71), CWE-114 (68) |
| SecRoBERTa, description only | 0.0361 | 262 | 30.8% | 12 | CWE-1300 (102), CWE-1299 (96), CWE-1426 (69), CWE-83 (59), CWE-93 (57) |
| mpnet, truncated (11.9 text) | 0.1344 | 317 | 18.7% | 37 | CWE-1249 (71), CWE-214 (58), CWE-520 (54), CWE-509 (51), CWE-379 (43) |
| bge-base (11.9 text) | 0.1196 | 281 | 21.9% | 31 | CWE-497 (83), CWE-1249 (77), CWE-520 (71), CWE-1422 (55), CWE-639 (52) |
| SecRoBERTa (11.9 text) | 0.0652 | 263 | 59.9% | 15 | CWE-1311 (108), CWE-1302 (96), CWE-1267 (94), CWE-1290 (71), CWE-1427 (59) |

  The probe's Recall@10 equals the evaluator value for every store. With the short description-only text the hubs change but do not go away: SecRoBERTa's lists are still dominated by a few CWEs (CWE-1300 in 102 of 186 lists), and bge's by the path-equivalence CWEs (CWE-57 in 106 lists) with only 22 gold CWEs reached; mpnet-chunk has the largest number of distinct CWEs and gold CWEs reached of the new stores. I did not check why these CWEs attract the technique texts.
- **Absolute level is still low:** the best new-text union Precision@10 is 0.035 (ceiling 0.319), Recall@10 0.145 (0.286 in the optimistic pool view).

## 6. Limits

- One run per encoder. The three encoders differ in pretraining, pooling and normalisation, so the order is an order of whole models.
- The chunk effect is measured for mpnet only. For bge and SecRoBERTa the new stores combine the text change with their own chunking (bge: chunked; SecRoBERTa: its existing chunking, which averages without weights), and the 11.9 reference rows differ in both text and chunking. I did not run a truncated bge on the new text.
- Chunking touches only 116 ATT&CK nodes (28 scored techniques in the union), so its effect on the aggregated metrics is bounded by that.
- `description_only` removes the CWE name as well as the extended description; the effect of dropping the name alone was not isolated.
- The token-weighted mean is a choice; the unweighted average used by SecRoBERTa was not tried. The ATT&CK text keeps its `(Citation: …)` markers.
- Gold is transitive through CAPEC; the pool is built from gold and is optimistic; the assignment F1 rests on few correct pairs.
- Old stores, ingests and runs are kept (11.9 stores unchanged). The mislabelled first `mpnet-trunc` run was deleted.

## 7. Files

- Modified: `configs/datasets/attack_cwe_ent.yaml`, `configs/pipelines/embed_mpnet.yaml`, `configs/pipelines/embed_bge.yaml`, `src/cga/pipelines/embed/text_embedders.py`, `src/cga/pipelines/embed/pipeline.py`, `scripts/unsup_summary.py`, `tests/unit/pipelines/embed/test_text_embedders.py`, `tests/unit/pipelines/embed/test_pipeline.py`, `dev_docs/sprints/sprint_11.md`, `dev_docs/execution_logs/sprint_11_9.md` (the hardware-range definition: the probe used CWE ids 1200 to 1399, not "1190 or higher").
- New: `configs/pipelines/embed_mpnet_chunk.yaml`, `configs/pipelines/embed_bge_chunk.yaml`, `configs/experiments/attack_cwe_ent_desc_eval.yaml`, this log.

## 8. Commands

```
for k in stix capec intersection union; do
  uv run cga run ingest --dataset attack_cwe_ent --set ingest.split.fractions=[1.0,0.0,0.0] \
    --set dataset.alignments.equivalence.source=$k
done
uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_mpnet_chunk.yaml --set device=cuda
uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_bge_chunk.yaml --set device=cuda
uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_mpnet.yaml --set device=cuda \
  --set embed.inputs.text_variant=description_only --set embed.embedder.label=multi_qa_mpnet_dot_trunc_desc
uv run cga run embed --dataset attack_cwe_ent --set device=cuda \
  --set embed.inputs.text_variant=description_only --set embed.embedder.label=pt_SecRoBERTa_desc
for each gold source (ingest run from section 2):
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_desc_eval \
    --set evaluate.inputs.ingest_run=<ingest run of that source>
U=20261027T203132Z-519bead2
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_desc \
  --ranking-eval gold_<full|pool> --prf-eval prf_<full|pool> --lift --out-name sources_summary_desc_<full|pool>.md \
  --source-run stix=20261027T203056Z-e2e6fcce --source-run capec=20261027T203109Z-a12d8b7a \
  --source-run intersection=20261027T203121Z-960dbd04 --source-run union=$U
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_desc \
  --set-eval set_<full|pool> --out-name sources_summary_desc_set_<full|pool>.md <the same --source-run options>
uv run pytest tests/unit -q
```
