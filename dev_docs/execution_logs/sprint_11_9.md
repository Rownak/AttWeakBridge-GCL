# Sprint 11.9: Controlled text-embedding baselines (`sentence-transformers/multi-qa-mpnet-base-dot-v1`, then `BAAI/bge-base-en-v1.5`)

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [11.6](sprint_11_6.md), [11.7](sprint_11_7.md). Case study of the low scores: [data_analysis](../data_analysis/attack_cwe_low_set_scores_case_study.md).

## 0. Outcome in brief

Sections 1 to 8 describe the first encoder (mpnet). Section 9 repeats the phase with a second sentence encoder, `BAAI/bge-base-en-v1.5`, with the same code, ingest, texts, evaluators and checks.

- **New embedder kind `sentence_transformer`** (one sentence vector per text, truncated at the model's 512 tokens, no chunking, stored un-normalised) and the new dependency `sentence-transformers` 6.2.0. `torch` stays `2.6.0+cu124`.
- **mpnet store:** 5,131 x 768 vectors on the GPU, same node index as the SecRoBERTa store. A second GPU run is bit-identical (max difference 0.0).
- **Result:** frozen mpnet is higher than frozen SecRoBERTa on every ranking metric in all four gold sources and both views: MRR (full view, union) 0.0585 against 0.0244, Hits@10 0.1178 against 0.0366. On the set-based metrics it is higher at every k tested (5, 10, 15, 20): union Recall@10 0.1344 against 0.0652 (11.0x against 5.3x random).
- **Assignment metrics (top-3, one-to-one) are mixed:** F1 is higher for mpnet in capec and union and lower in stix and intersection. Each cell rests on 2 to 14 correct pairs, so I do not read a direction from F1.
- **Hubs:** mpnet's top-10 lists are far less dominated by the hardware / SoC CWEs (19% of top-10 slots against 60% for SecRoBERTa). It reaches 37 distinct gold CWEs in a top-10 list, SecRoBERTa 15.
- **Checks passed:** embed repeat identical; the 11.7 rows (420 per source) are reproduced exactly when mpnet is added to the 11.7 method set; the union evaluation repeat is identical; `n_queries` is equal across methods.
- **Still low in absolute terms:** union Precision@10 is 0.038 against a ceiling of 0.319, and the assignment recall is at most 0.025.
- **Second encoder (bge-base, section 9):** frozen bge-base is between the two: above SecRoBERTa and below mpnet on MRR and Hits@10 in every source and both views (union MRR, full view: mpnet 0.0585, bge 0.0501, SecRoBERTa 0.0244; union Recall@10: 0.1344, 0.1196, 0.0652). On the set-based metrics it is below mpnet in all 12 cells per source and view, and above SecRoBERTa in 8 to 12 of 12. Its repeat embed, its repeat evaluation and the reproduction of the mpnet, SecRoBERTa and random rows are all exact.

## 1. Code and dependency

- `uv add sentence-transformers` (`pyproject.toml`, `uv.lock`): adds `sentence-transformers==6.2.0` only; `torch 2.6.0+cu124` and `transformers 5.17.0` unchanged, `torch.cuda.is_available()` is `True`.
- `src/cga/pipelines/embed/text_embedders.py`: `SentenceTransformerEmbedder` with `embed(text)` and `embed_batch(texts)`. It loads the model once on `device`, skips empty texts (zero vector, `n_chunks = 0`), returns `n_chunks = 1` and does not normalise. The import is inside `__init__`, so the dependency is only needed for this kind. The dimension call accepts both names of the method (renamed in v6).
- `src/cga/pipelines/embed/pipeline.py`: the kind is added to `_TEXT_KINDS`, to the `Unknown embedder.kind` message and to the embedder selection in `_generate_text`. For this kind the pipeline encodes in batches of `embed.embedder.batch_size`; if a batch raises, each node of it is retried with `embed()`, so the per-node `failed` status still works. The `hf_mean_pool` path is unchanged (batch size 1, same statuses).
- Tests (+4, 591 pass): `tests/unit/pipelines/embed/test_text_embedders.py` (stub `SentenceTransformer`: un-normalised vector, zero vector and input order for empty text) and `test_pipeline.py` (batching and skipping empty text; fallback to per-node when a batch fails; embedder id).
- `scripts/unsup_summary.py`: row sets `attack_ent_mpnet` (mpnet, frozen SecRoBERTa, random) and, for the second encoder, `attack_ent_encoders` (bge-base, mpnet, SecRoBERTa, random).

## 2. Embedding

- GPU: NVIDIA GeForce RTX 4090 Laptop, 16,376 MiB; `torch 2.6.0+cu124`; `sentence-transformers 6.2.0`; `transformers 5.17.0`; Python 3.13.5.
- Config `configs/pipelines/embed_mpnet.yaml` (`batch_size: 32`); embedder id `sentence_transformer-multi_qa_mpnet_dot`; ingest `20261027T134320Z-32a16662` (`LATEST`).

| Run | Wall time | Result |
|---|---|---|
| `20261027T183826Z-2cd26dca` (pinned) | 357 s | 5,131 ok, 709 empty text, 0 failed |
| `20261027T184443Z-2cd26dca` (repeat; `LATEST`) | similar | same |

- **Checks:** same node ids and order as `hf_mean_pool-pt_SecRoBERTa/20261027T134403Z-8bc08adc`; shape 5,131 x 768, float32; no NaN; no zero rows; vector norms 5.30 to 7.12 (un-normalised, as intended). The 709 nodes with empty text get no vector, as in the SecRoBERTa store (same 5,131 stored nodes).
- **Repeat on the GPU:** same index, max absolute difference 0.0, arrays equal. I expected only a small difference; the fixed batch size and order give identical kernels. This is a property of this machine and these library versions, not a guarantee.
- **Time:** the GPU forward pass is a small part. Most of the 357 s is the per-node store writes (generate about 4.5 min, materialize about 80 s).
- **Truncation** (`max_seq_length` = 512 tokens; texts with description variant, token counts with the model's tokenizer):

| Graph | Nodes with text | Truncated | Share | Median tokens | p95 | Max |
|---|---|---|---|---|---|---|
| ATT&CK (`graph_1`) | 4,162 | 116 | 2.8% | 85 | 442 | 1,168 |
| CWE (`graph_2`) | 969 | 1 | 0.1% | 74 | 266 | 738 |

  Among the 867 `attack-pattern` nodes (techniques and sub-techniques), 105 (12.1%) are truncated; median 298 tokens. In the truncated texts a mean of 18% of the tokens is dropped. SecRoBERTa instead chunks and averages: 82 of its 5,131 nodes have more than one chunk. So the two encoders see the same text except for the tail of about 105 technique descriptions.
- The model was loaded unauthenticated from the Hub (a warning only); it is cached under `~/.cache/huggingface`.

## 3. Evaluation

Config `configs/experiments/attack_cwe_ent_mpnet_eval.yaml`: methods `ent-mpnet` (pinned `…183826Z-2cd26dca`), `ent-s-r1` (pinned `…134403Z-8bc08adc`), `ent-rand-s42/43/44`; evaluations `gold_full`, `prf_full`, `gold_pool`, `prf_pool`, `set_full`, `set_pool` (parameters as in 11.4 and 11.7; the pool file is `target_pool_union.json`). Cosine only.

| Gold source | Ingest | Evaluate run | Techniques scored (set) | Gold pairs |
|---|---|---|---|---|
| stix | `20261027T134239Z-8101c66c` | `20261027T185050Z-e2d67278` | 96 | 287 |
| capec | `20261027T134252Z-cbfb9a99` | `20261027T185101Z-f4c4ddf9` | 174 | 566 |
| intersection | `20261027T134304Z-d983eac7` | `20261027T185113Z-460f30b3` | 78 | 221 |
| union | `20261027T134320Z-32a16662` | `20261027T185125Z-773f1265` | 186 | 628 |

- Candidates are all 969 CWEs (full view) or the 151 pool CWEs. This is a wider candidate set than in 11.7 (944 / 93 / 183 techniques) because that run also scored the dual GCNs, which only cover 2,927 nodes. So the numbers here are not comparable with the 11.7 tables; they are comparable within this run.

### Checks

- **Reproduction:** the 11.9 method set is `ent-mpnet`, `ent-s-r1`, random, so the frozen and random rows are not directly comparable with 11.7. For a like-for-like check I ran the 11.7 config with `ent-mpnet` added (scratch copy, not kept; runs `20261027T185226Z-dde663f4`, `…185241Z-0fca8869`, `…185257Z-29ecd0d2`, `…185310Z-5de11fa9`, in source order). All 420 rows per source of the 11.7 methods match the 11.7 runs (`…143203Z-359b85eb`, `…143214Z-460087ac`, `…143225Z-d9a8711e`, `…143236Z-91f56f41`): max difference 0.0, equal `n_queries`, no unmatched row. The 11.4 reference runs used the old CWE texts, so they are not a reference here.
- **Repeat:** union rerun `20261027T185141Z-773f1265`: 360 of 360 rows identical (max difference 0.0).
- **Query set:** `n_queries` is equal for every method in every evaluation (stix 287 / 96, capec 566 / 174, intersection 221 / 78, union 628 / 186 for the pair / technique evaluations).

### Tables (all in `artifacts/evaluations/attack_cwe_ent/20261027T185125Z-773f1265/`)

`sources_summary_mpnet_full.md`, `sources_summary_mpnet_pool.md` (Hits@1, Hits@10, MRR, P, R, F1, #Pred, #Correct, #Gold, recall ceiling, MRR lift), `sources_summary_mpnet_set_full.md`, `sources_summary_mpnet_set_pool.md` (Hits / Recall / Precision at 5, 10, 15, 20, the precision ceiling, Recall@10 lift). Hits@5 is also in the evaluate metrics.

**Ranking (per gold pair, sibling-filtered), MRR, mpnet against SecRoBERTa (random in brackets):**

| View | stix | capec | intersection | union |
|---|---|---|---|---|
| full | 0.0615 vs 0.0299 (0.0063) | 0.0596 vs 0.0265 (0.0069) | 0.0662 vs 0.0372 (0.0063) | 0.0585 vs 0.0244 (0.0068) |
| pool | 0.1222 vs 0.0588 (0.0386) | 0.1253 vs 0.0608 (0.0375) | 0.1337 vs 0.0697 (0.0390) | 0.1216 vs 0.0571 (0.0376) |

Union Hits@1 / Hits@10, mpnet against SecRoBERTa: full 0.0239 / 0.1178 against 0.0096 / 0.0366; pool 0.0462 / 0.2611 against 0.0159 / 0.1067 (random pool Hits@10 is 0.0695).

**Assignment (top-3 per technique, greedy one-to-one), union:**

| View | Method | P | R | F1 | #Pred | #Correct | #Gold |
|---|---|---|---|---|---|---|---|
| full | mpnet | 0.0968 | 0.0191 | 0.0319 | 124 | 12 | 628 |
| full | SecRoBERTa | 0.0874 | 0.0143 | 0.0246 | 103 | 9 | 628 |
| pool | mpnet | 0.2090 | 0.0223 | 0.0403 | 67 | 14 | 628 |
| pool | SecRoBERTa | 0.1321 | 0.0111 | 0.0206 | 53 | 7 | 628 |

F1, mpnet against SecRoBERTa, full view: stix 0.0112 against 0.0341 (2 against 6 correct), capec 0.0349 against 0.0270 (12 against 9), intersection 0.0211 against 0.0436 (3 against 6), union 0.0319 against 0.0246. Pool view: stix 0.0238 against 0.0365, capec 0.0442 against 0.0227, intersection 0.0300 against 0.0467, union 0.0403 against 0.0206. The recall ceiling of the one-to-one printout is 0.2962 for union (printed "unique sources / #gold"); the true one-to-one ceiling is lower (gold-alignment analysis report: 0.16 to 0.20).

**Set-based (per technique), union, mpnet against SecRoBERTa (random):**

| View | Metric | @5 | @10 | @15 | @20 |
|---|---|---|---|---|---|
| full | Hits | 0.2043 vs 0.0860 | 0.2957 vs 0.1183 | 0.3387 vs 0.1559 | 0.3710 vs 0.1774 |
| full | Recall | 0.0864 vs 0.0419 | 0.1344 vs 0.0652 (0.0122) | 0.1753 vs 0.0847 | 0.2003 vs 0.0982 |
| full | Precision | 0.0462 vs 0.0172 | 0.0382 vs 0.0124 | 0.0323 vs 0.0111 | 0.0277 vs 0.0097 |
| pool | Hits | 0.3763 vs 0.1828 | 0.5269 vs 0.2796 | 0.5806 vs 0.3656 | 0.6183 vs 0.4462 |
| pool | Recall | 0.1757 vs 0.0993 | 0.2995 vs 0.1643 (0.0696) | 0.3522 vs 0.2270 | 0.4080 vs 0.2834 |
| pool | Precision | 0.0989 vs 0.0419 | 0.0817 vs 0.0349 | 0.0685 vs 0.0312 | 0.0618 vs 0.0312 |

Precision ceilings (union): 0.489 / 0.319 / 0.220 / 0.168 at k = 5 / 10 / 15 / 20. Recall@10 lift over random (full view): mpnet 18.1x / 11.0x / 18.8x / 11.0x, SecRoBERTa 12.0x / 5.5x / 13.4x / 5.3x (stix / capec / intersection / union); pool view: mpnet 3.4x / 4.5x / 3.5x / 4.3x, SecRoBERTa 2.0x / 2.5x / 2.2x / 2.4x. In every source and view, and at every k in the tables, mpnet is above SecRoBERTa on Hits, Recall and Precision.

**Cosine and CSLS:** cosine only. The evaluators have no CSLS option and `grep -ri csls src scripts` finds nothing, so CSLS is not applicable to this phase and no CSLS rows exist. The CSLS and mean-centring probe in the case-study report remains a separate follow-up.

## 4. Hubs and the cosine-versus-dot question (read-only probe, not part of the pipeline)

Union, 186 techniques, all 969 CWEs, top-10 lists:

| Store | Recall@10 (cosine) | Distinct CWEs in the lists | Hardware-range CWEs (id 1200 to 1399) among the 1,860 slots | Gold CWEs that reach any top-10 | Five most frequent CWEs (lists containing them) |
|---|---|---|---|---|---|
| mpnet | 0.1344 | 317 | 18.7% | 37 | CWE-1249 (71), CWE-214 (58), CWE-520 (54), CWE-509 (51), CWE-379 (43) |
| SecRoBERTa | 0.0652 | 263 | 59.9% | 15 | CWE-1311 (108), CWE-1302 (96), CWE-1267 (94), CWE-1290 (71), CWE-1427 (59) |

- The cosine Recall@10 of the probe equals the evaluator value (0.1344, 0.0652). The share of hardware-range CWEs in the pool is 12.8%; I defined "hardware range" as CWE ids 1200 to 1399 (the hardware / SoC weaknesses are mostly in this range), an approximation I did not check against the CWE views. (An earlier draft of this log said "1190 or higher"; the probe used 1200 to 1399.)
- mpnet's most frequent CWEs appear in at most 71 of 186 lists (SecRoBERTa: 108), so hubs are weaker but not gone (CWE-1249, CWE-214, CWE-520).
- **Native dot product** (probe): mpnet Recall@10 is 0.1176 with the raw dot product and 0.1344 with cosine; SecRoBERTa 0.0445 against 0.0652. Cosine is not the weaker choice here, so scoring by cosine does not explain a handicap for mpnet.
- **Context for 11.10:** on the 11.7 query set (the dual GCNs cover 183 of 186 union techniques), union Recall@10 in the like-for-like runs is mpnet 0.1366, SecRoBERTa 0.0663, InfoNCE dual 0.0360, triplet dual 0.0158.

## 5. Report (no tuning)

- **mpnet against SecRoBERTa, frozen:** mpnet is higher in every source and both views on MRR, Hits@1, Hits@10, Recall@k, Hits@k and Precision@k; about 2.4x SecRoBERTa's union MRR in the full view (0.0585 against 0.0244) and about 2.1x its Recall@10 (0.1344 against 0.0652). F1 of the top-3 assignment is mixed (higher in 2 of 4 sources in each view), and rests on 2 to 14 correct pairs per cell, so it does not contradict the ranking result.
- **Hubs:** the hardware / SoC hubs seen in the 11.6 and case-study analyses mostly disappear (19% of top-10 slots against 60%). Smaller hubs (CWE-1249, CWE-214) remain.
- **Truncation:** 105 of 867 technique descriptions (12%) are cut at 512 tokens, losing on average 18% of their tokens. The cut tail of long technique texts is not seen by mpnet; I did not test whether it matters.
- **Absolute level is still low:** union Precision@10 is 0.038 (ceiling 0.319); Recall@10 is 0.134, and the pool view, which is optimistic because it is built from gold, gives 0.300. Gold pairs that need CAPEC knowledge (for example Audio Capture and Screen Capture to CWE-267) are not reachable from the text.

## 6. Limits

- One encoder run per model; the mpnet and SecRoBERTa stores differ in pretraining data, objective, pooling (sentence encoder against token mean-pooling with chunks) and truncation, so the result measures "the encoder as a whole", not a single factor. It does not show why mpnet is better (a plausible but untested reason is its retrieval-style training; the reduced hub effect is observed, not shown to be the cause).
- Cosine instead of the model's native dot product; the probe above shows this does not favour SecRoBERTa.
- Gold is transitive through CAPEC ("exploits", not "equivalent"); the pool is built from gold and is optimistic (random reaches a set-based Hits@10 of 0.19 there).
- The numbers of this phase use all 969 CWEs, so they are not comparable with the 11.7 tables (944 CWEs, techniques covered by the dual GCNs); the like-for-like runs in section 3 are the bridge.
- No fine-tuning, no prefixes, no CSLS or centring, no training on mpnet vectors (Phase 11.10).
- Old stores, ingests and runs are kept. The mpnet runs `…183826Z-2cd26dca` and `…184443Z-2cd26dca` are identical; `LATEST` points to the second, the configs pin the first.

## 7. Files

- Modified: `pyproject.toml`, `uv.lock`, `src/cga/pipelines/embed/text_embedders.py`, `src/cga/pipelines/embed/pipeline.py`, `scripts/unsup_summary.py`, `tests/unit/pipelines/embed/test_text_embedders.py`, `tests/unit/pipelines/embed/test_pipeline.py`, `dev_docs/sprints/sprint_11.md`.
- New: `configs/pipelines/embed_mpnet.yaml`, `configs/experiments/attack_cwe_ent_mpnet_eval.yaml`, and for the second encoder `configs/pipelines/embed_bge.yaml`, `configs/experiments/attack_cwe_ent_encoders_eval.yaml`; this log.

## 8. Commands

```
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv add sentence-transformers
uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_mpnet.yaml --set device=cuda
for each gold source (ingest run from section 3):
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_mpnet_eval \
    --set evaluate.inputs.ingest_run=<ingest run of that source>
U=20261027T185125Z-773f1265
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_mpnet \
  --ranking-eval gold_<full|pool> --prf-eval prf_<full|pool> --lift --out-name sources_summary_mpnet_<full|pool>.md \
  --source-run stix=20261027T185050Z-e2d67278 --source-run capec=20261027T185101Z-f4c4ddf9 \
  --source-run intersection=20261027T185113Z-460f30b3 --source-run union=$U
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_mpnet \
  --set-eval set_<full|pool> --out-name sources_summary_mpnet_set_<full|pool>.md <the same --source-run options>
uv run pytest tests/unit -q

# second encoder (section 9)
uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_bge.yaml --set device=cuda
for each gold source (ingest run from section 3):
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_encoders_eval \
    --set evaluate.inputs.ingest_run=<ingest run of that source>
U=20261027T191929Z-006fe93d
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_encoders \
  --ranking-eval gold_<full|pool> --prf-eval prf_<full|pool> --lift --out-name sources_summary_encoders_<full|pool>.md \
  --source-run stix=20261027T191838Z-849137da --source-run capec=20261027T191857Z-5c78e309 \
  --source-run intersection=20261027T191914Z-766d74b3 --source-run union=$U
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_encoders \
  --set-eval set_<full|pool> --out-name sources_summary_encoders_set_<full|pool>.md <the same --source-run options>
```

## 9. Second encoder: `BAAI/bge-base-en-v1.5`

The phase is repeated with one more sentence encoder. Nothing else changes: the same `sentence_transformer` kind, ingest `20261027T134320Z-32a16662`, `description` texts, batch size 32, GPU, evaluators and checks. No retrieval instruction prefix is used for the queries (both sides are encoded alike), so the model is used without its recommended query instruction; I did not test the prefix.

### 9.1 Embedding

- Config `configs/pipelines/embed_bge.yaml`; embedder id `sentence_transformer-bge_base_en_v1_5`. The model's modules are Transformer, Pooling and Normalize.

| Run | Wall time | Result |
|---|---|---|
| `20261027T190733Z-d71a74dc` (pinned) | 310 s | 5,131 ok, 709 empty text, 0 failed |
| `20261027T191259Z-d71a74dc` (repeat; `LATEST`) | 312 s | same |

- **Checks:** same node ids and order as the SecRoBERTa store; 5,131 x 768; no NaN; no zero rows. The vectors are unit-norm (0.9999999 to 1.0000001) because the model normalises inside, so for this model the raw dot product equals the cosine (the mpnet vectors are un-normalised, norm 5.3 to 7.1).
- **Repeat on the GPU:** same index, max absolute difference 0.0, arrays equal.
- **Truncation:** `max_seq_length` is 512. With this model's tokenizer the counts equal those of mpnet: 116 of 4,162 ATT&CK texts (2.8%) and 1 of 969 CWE texts, 105 of 867 `attack-pattern` nodes (12.1%), mean share of tokens dropped 18%.

### 9.2 Evaluation

Config `configs/experiments/attack_cwe_ent_encoders_eval.yaml`: methods `ent-bge` (pinned `…190733Z-d71a74dc`), `ent-mpnet` (pinned `…183826Z-2cd26dca`), `ent-s-r1` (pinned `…134403Z-8bc08adc`), `ent-rand-s42/43/44`; the same six evaluations as in section 3. Cosine only; CSLS is not applicable (section 3).

| Gold source | Evaluate run | Techniques scored (set) |
|---|---|---|
| stix | `20261027T191838Z-849137da` | 96 |
| capec | `20261027T191857Z-5c78e309` | 174 |
| intersection | `20261027T191914Z-766d74b3` | 78 |
| union | `20261027T191929Z-006fe93d` | 186 |

- **Reproduction:** the `ent-mpnet`, `ent-s-r1` and random rows (360 per source) equal the section 3 runs (`…185050Z-e2d67278`, `…185101Z-f4c4ddf9`, `…185113Z-460f30b3`, `…185125Z-773f1265`): max difference 0.0, equal `n_queries`, no unmatched row. The candidate set is the same as in section 3, because all these stores cover all nodes.
- **Repeat:** union rerun `20261027T191948Z-006fe93d`: 432 of 432 rows identical.
- **Query set:** `n_queries` is equal for every method in every evaluation.
- **Tables** in `artifacts/evaluations/attack_cwe_ent/20261027T191929Z-006fe93d/`: `sources_summary_encoders_full.md`, `sources_summary_encoders_pool.md`, `sources_summary_encoders_set_full.md`, `sources_summary_encoders_set_pool.md`.

**Ranking (per gold pair), MRR, bge-base / mpnet / SecRoBERTa:**

| View | stix | capec | intersection | union |
|---|---|---|---|---|
| full | 0.0420 / 0.0615 / 0.0299 | 0.0520 / 0.0596 / 0.0265 | 0.0453 / 0.0662 / 0.0372 | 0.0501 / 0.0585 / 0.0244 |
| pool | 0.0859 / 0.1222 / 0.0588 | 0.1092 / 0.1253 / 0.0608 | 0.0911 / 0.1337 / 0.0697 | 0.1056 / 0.1216 / 0.0571 |

Random MRR is 0.0063 to 0.0069 (full) and 0.0375 to 0.0390 (pool). Union Hits@1 / Hits@10 for bge-base: full 0.0207 / 0.0908, pool 0.0414 / 0.2118 (mpnet 0.0239 / 0.1178 and 0.0462 / 0.2611; SecRoBERTa 0.0096 / 0.0366 and 0.0159 / 0.1067). MRR lift over random in the full view: bge 6.7x / 7.6x / 7.1x / 7.3x, mpnet 9.8x / 8.7x / 10.4x / 8.6x, SecRoBERTa 4.8x / 3.9x / 5.9x / 3.6x. bge is above SecRoBERTa on MRR and Hits@10 in all four sources and both views; on Hits@1 it is above in 5 of 8 cells, tied in intersection (full, 0.0181 each) and below in stix pool (0.0209 against 0.0244) and intersection pool (0.0226 against 0.0317). bge is below mpnet on MRR, Hits@1 and Hits@10 in every cell.

**Assignment (top-3, greedy one-to-one), union:**

| View | Method | P | R | F1 | #Pred | #Correct | #Gold |
|---|---|---|---|---|---|---|---|
| full | bge-base | 0.0862 | 0.0159 | 0.0269 | 116 | 10 | 628 |
| pool | bge-base | 0.1194 | 0.0127 | 0.0230 | 67 | 8 | 628 |

(mpnet has 12 and 14 correct pairs, SecRoBERTa 9 and 7, in the full and pool views.) F1, bge against SecRoBERTa, full view: stix 0.0226 against 0.0341, capec 0.0265 against 0.0270, intersection 0.0214 against 0.0436, union 0.0269 against 0.0246; pool view: 0.0303 against 0.0365, 0.0221 against 0.0227, 0.0382 against 0.0467, 0.0230 against 0.0206. So bge is above SecRoBERTa on F1 only in union; each cell rests on 3 to 10 correct pairs and I do not read a direction from it.

**Set-based, union, bge-base (mpnet and SecRoBERTa in section 3):**

| View | Metric | @5 | @10 | @15 | @20 |
|---|---|---|---|---|---|
| full | Hits | 0.1613 | 0.2419 | 0.3065 | 0.3548 |
| full | Recall | 0.0838 | 0.1196 | 0.1521 | 0.1784 |
| full | Precision | 0.0366 | 0.0296 | 0.0254 | 0.0228 |
| pool | Hits | 0.3333 | 0.4301 | 0.5000 | 0.5699 |
| pool | Recall | 0.1739 | 0.2253 | 0.2933 | 0.3600 |
| pool | Precision | 0.0817 | 0.0672 | 0.0570 | 0.0522 |

Recall@10 lift over random (stix / capec / intersection / union): full view bge 10.5x / 9.8x / 10.5x / 9.8x (mpnet 18.1x / 11.0x / 18.8x / 11.0x, SecRoBERTa 12.0x / 5.5x / 13.4x / 5.3x); pool view bge 2.4x / 3.3x / 2.5x / 3.2x (mpnet 3.4x / 4.5x / 3.5x / 4.3x, SecRoBERTa 2.0x / 2.5x / 2.2x / 2.4x).

Of the 12 cells per source and view (Hits, Recall and Precision at k = 5, 10, 15, 20), bge is above SecRoBERTa in: full view 9 (stix), 12 (capec), 8 (intersection), 12 (union); pool view 11, 12, 11, 12. The cells where it is not above are all in stix and intersection: Hits@5 (a tie in stix full, 0.0938 each; lower in intersection full), Recall@5 and Recall@10 in the full view, Recall@15 in intersection full, and Recall@15 in the pool view. bge is above mpnet in 0 of 12 cells in every source and view.

### 9.3 Hubs (read-only probe, union, 186 techniques, 969 CWEs, top-10 lists)

| Store | Recall@10 | Distinct CWEs | Hardware-range share of slots | Gold CWEs reaching a top-10 | Five most frequent CWEs (lists containing them) |
|---|---|---|---|---|---|
| bge-base | 0.1196 | 281 | 21.9% | 31 | CWE-497 (83), CWE-1249 (77), CWE-520 (71), CWE-1422 (55), CWE-639 (52) |
| mpnet | 0.1344 | 317 | 18.7% | 37 | CWE-1249 (71), CWE-214 (58), CWE-520 (54), CWE-509 (51), CWE-379 (43) |
| SecRoBERTa | 0.0652 | 263 | 59.9% | 15 | CWE-1311 (108), CWE-1302 (96), CWE-1267 (94), CWE-1290 (71), CWE-1427 (59) |

The probe's bge Recall@10 equals the evaluator value. Both sentence encoders lose the hardware / SoC hubs seen with SecRoBERTa and keep smaller ones (CWE-497, CWE-1249, CWE-520); bge's most frequent CWE is in 83 of 186 lists, mpnet's in 71. "Hardware range" is CWE ids 1200 to 1399, as in section 4.

### 9.4 Report (no tuning)

- **Order of the frozen encoders:** mpnet is above bge-base, and bge-base above SecRoBERTa, on MRR, Hits@10, Recall@k and Precision@k in both views (union MRR 0.0585 / 0.0501 / 0.0244; Recall@10 0.1344 / 0.1196 / 0.0652). The gap between the two sentence encoders is smaller than their gap to SecRoBERTa. The gain over SecRoBERTa is not uniform: in stix and intersection bge is below SecRoBERTa on a few low-k Recall cells and on Hits@1 in the pool view, and its one-to-one F1 is below SecRoBERTa in 3 of 4 sources (full view).
- **Both sentence encoders** show the effect that matters most here: far fewer hardware / SoC CWEs in the top-10 lists (22% and 19% of slots against 60%) and more distinct gold CWEs reached (31 and 37 against 15). This points at the encoder as the source of the SecRoBERTa hubs; I did not test the cause.
- **Absolute level is still low** for bge as well: union Precision@10 is 0.030 (ceiling 0.319); Recall@10 is 0.120, and 0.225 in the optimistic pool view.
- **Limits:** one run per encoder; the encoders differ in pretraining, objective, pooling and normalisation (bge's vectors are unit-norm, mpnet's are not), so the order is an order of whole models; no query instruction for bge; truncation at 512 tokens affects 105 technique descriptions for both sentence encoders; the other limits of section 6 apply.
