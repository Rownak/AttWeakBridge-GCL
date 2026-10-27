# Sprint 11.7: Set-based gold evaluation (Hits@k, Recall@k, Precision@k)

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [11.6](sprint_11_6.md). Why this protocol: [data_analysis](../data_analysis/attack_capec_cwe_gold_alignment_analysis.md) section 4.

## 0. Outcome in brief

- New evaluator `gold_set_retrieval`: one query per ATT&CK technique, all of its gold CWEs are relevant together. Reports Hits@k, Recall@k, Precision@k and the precision ceiling `min(|R|, k) / k`, for k = 1, 5, 10, 15, 20, 50.
- **Checks passed:** repeating the union run gives 420 of 420 identical rows (max difference 0.0). Scoring the pre-fix SecRoBERTa store against the 11.6 ingests gives the same rows as scoring it against the old ingests, for all four sources (max difference 0.0).
- **The text fix did not clearly improve frozen SecRoBERTa.** At small k it helps (Recall@5 on union 0.031 → 0.043); at larger k it is lower or equal (Hits@20 on union 0.246 → 0.180; Recall@10 is mixed). It removed the CWE-120 / CWE-732 hubs (11.6) without a gain in retrieval.
- **Neither dual GCN beats frozen SecRoBERTa** on Recall@10 or Hits@10 in either view. InfoNCE is level with frozen on stix and intersection (full view) and clearly lower on capec and union. Triplet is near random on the full view.
- **InfoNCE is higher than triplet** in every source on the full view; on the pool view the two are closer.
- Scores stay low: with all 944 CWEs, frozen SecRoBERTa finds about 7% of a technique's gold CWEs in the top 10 (union Recall@10 0.066), about 5x random.

## 1. What was implemented

| File | Change |
|---|---|
| `src/cga/pipelines/evaluate/evaluators/gold_set_retrieval.py` | New evaluator, registered as `gold_set_retrieval` |
| `src/cga/pipelines/evaluate/evaluators/__init__.py` | Imports the new module |
| `src/cga/pipelines/evaluate/metrics.py` | Added `precision_at_k` and `precision_ceiling_at_k` |
| `scripts/unsup_summary.py` | Row set `attack_ent_set`, `build_set_report`, CLI option `--set-eval` |
| `configs/experiments/attack_cwe_ent_set_eval.yaml` | Methods and evaluations of this phase |
| `tests/unit/pipelines/evaluate/test_gold_set_retrieval.py` | 9 tests (new) |
| `tests/unit/test_unsup_summary.py` | 1 test for `build_set_report` |

**Evaluator rules** (all verified by the unit tests):
- **Queries:** one per source node with a gold pair in the split, present in every method, with at least one gold target among the candidates. Targets outside the candidate pool are dropped from that source's relevant set. A source left with none is not a query (counted in `n_dropped_missing_in_a_method`).
- **Candidates:** every `graph_2` node shared by all methods (944 CWEs), or the 151-CWE `target_pool`. There is no sibling filter.
- **Ranking:** cosine, ties broken by node id (the same `top_k_by_similarity` as the other evaluators).
- **Averaging:** macro over techniques. `Precision@k = |top-k ∩ R| / k` with a fixed `k` even when the pool is smaller. The ceiling is averaged over the same techniques.
- **Provenance** (in `metrics.json`): query-set sha256, number of candidates, `|R|` mean / median / max (union: 3.42 / 2 / 23).
- `gold_alignment` and `gold_alignment_prf` are unchanged.

**Unit tests** (587 passed in total): hand-built 3-technique case (`|R| = 1`, `|R| > k`, a relevant CWE with no embedding); `k` larger than the pool; tie-break by id; ceiling values; `target_pool` restriction and the error for a gold target outside the pool; invalid `ks`; provenance; random vectors give Recall@10 ≈ 10 / 200 within 0.03; the precision helpers.

## 2. Evaluate runs (11.6 ingests; `attack_cwe_ent_set_eval`)

| Gold source | Ingest run | Evaluate run | Techniques scored (`n_queries`) |
|---|---|---|---|
| stix | `20261027T134239Z-8101c66c` | `20261027T143203Z-359b85eb` | 93 of 96 |
| capec | `20261027T134252Z-cbfb9a99` | `20261027T143214Z-460087ac` | 174 of 174 |
| intersection | `20261027T134304Z-d983eac7` | `20261027T143225Z-d9a8711e` | 78 of 78 |
| union | `20261027T134320Z-32a16662` | `20261027T143236Z-91f56f41` | 183 of 186 |

- Repeat of union: `20261027T143249Z-91f56f41`.
- The 3 stix / union techniques outside the dual-GCN training universe are dropped for all methods, so every row shares one query set.
- Methods: `ent-s-r1` (frozen SecRoBERTa, new texts, store `20261027T134403Z-8bc08adc`), `ent-s-r1-old` (pre-fix store `20261027T010723Z-8bc08adc`), `ent-d-trip` (`ent-dual-gcn-triplet-s42b`, run `20261027T141524Z-5ac0fba7`), `ent-d-nce` (`ent-dual-gcn-infonce-s42b`, run `20261027T141732Z-3f70b1a2`), `ent-rand-s42/43/44`.
- Evaluations: `set_full` and `set_pool` (new evaluator), plus `gold_full` and `prf_full` kept as a bridge to the Sprint 11.4 numbers.
- Reports: `artifacts/evaluations/attack_cwe_ent/20261027T143236Z-91f56f41/sources_summary_set_full.md` and `sources_summary_set_pool.md` (all four sources, k = 5, 10, 15, 20, precision ceiling, Recall@10 lift).

## 3. Checks

- **Repeat:** union, 420 rows, 420 matched, max difference 0.0.
- **Old vs new texts, method scored on the old ingests:** the config scores `ent-s-r1-old` against the new ingests (same node ids and gold pairs, verified in 11.6). I also ran the config against the four old ingests (runs `20261027T143306Z-48f8fc47`, `…143315Z-5cf806c3`, `…143326Z-f818674b`, `…143336Z-d5a25f9d`, in source order). The 60 rows of `ent-s-r1-old` per source are identical, with equal `n_queries`.

## 4. Results

Union gold (183 techniques, mean 3.4 and median 2 gold CWEs per technique). The other sources are in the report files. Random = mean of 3 seeds.

**All 944 CWEs (`set_full`)**

| Method | Hits@5 | Hits@10 | Hits@20 | Recall@5 | Recall@10 | Recall@20 | Precision@10 |
|---|---|---|---|---|---|---|---|
| Dual GCN, triplet | 0.044 | 0.066 | 0.109 | 0.007 | 0.016 | 0.035 | 0.009 |
| Dual GCN, InfoNCE | 0.049 | 0.077 | 0.159 | 0.020 | 0.036 | 0.071 | 0.008 |
| SecRoBERTa (frozen) | 0.087 | 0.120 | 0.180 | 0.043 | 0.066 | 0.100 | 0.013 |
| SecRoBERTa (frozen, pre-fix) | 0.077 | 0.148 | 0.246 | 0.031 | 0.063 | 0.120 | 0.016 |
| Random | 0.013 | 0.035 | 0.073 | 0.004 | 0.013 | 0.026 | 0.004 |
| *Precision ceiling* | | | | | | | 0.323 |

**151-CWE pool (`set_pool`)**

| Method | Hits@5 | Hits@10 | Hits@20 | Recall@5 | Recall@10 | Recall@20 | Precision@10 |
|---|---|---|---|---|---|---|---|
| Dual GCN, triplet | 0.115 | 0.240 | 0.421 | 0.029 | 0.105 | 0.225 | 0.035 |
| Dual GCN, InfoNCE | 0.153 | 0.257 | 0.426 | 0.072 | 0.127 | 0.230 | 0.033 |
| SecRoBERTa (frozen) | 0.186 | 0.284 | 0.454 | 0.101 | 0.167 | 0.288 | 0.036 |
| SecRoBERTa (frozen, pre-fix) | 0.219 | 0.339 | 0.492 | 0.100 | 0.184 | 0.289 | 0.043 |
| Random | 0.118 | 0.197 | 0.326 | 0.040 | 0.071 | 0.128 | 0.023 |
| *Precision ceiling* | | | | | | | 0.323 |

**Recall@10 and lift over random (`set_full`)**

| Method | stix | capec | intersection | union |
|---|---|---|---|---|
| Dual GCN, triplet | 0.005 (0.7x) | 0.017 (1.3x) | 0.006 (0.8x) | 0.016 (1.2x) |
| Dual GCN, InfoNCE | 0.070 (9.8x) | 0.038 (2.9x) | 0.083 (10.8x) | 0.036 (2.8x) |
| SecRoBERTa (frozen) | 0.076 (10.8x) | 0.070 (5.3x) | 0.091 (11.9x) | 0.066 (5.2x) |
| SecRoBERTa (frozen, pre-fix) | 0.062 (8.7x) | 0.071 (5.4x) | 0.076 (9.9x) | 0.063 (4.9x) |
| Random | 0.007 | 0.013 | 0.008 | 0.013 |

**Recall@10 and lift over random (`set_pool`)**

| Method | stix | capec | intersection | union |
|---|---|---|---|---|
| Dual GCN, triplet | 0.107 (1.3x) | 0.104 (1.5x) | 0.094 (1.1x) | 0.105 (1.5x) |
| Dual GCN, InfoNCE | 0.148 (1.8x) | 0.131 (1.9x) | 0.170 (2.1x) | 0.127 (1.8x) |
| SecRoBERTa (frozen) | 0.163 (2.0x) | 0.170 (2.5x) | 0.182 (2.2x) | 0.167 (2.4x) |
| SecRoBERTa (frozen, pre-fix) | 0.187 (2.3x) | 0.193 (2.8x) | 0.212 (2.6x) | 0.184 (2.6x) |
| Random | 0.081 | 0.069 | 0.083 | 0.071 |

## 5. Findings

1. **Effect of the text fix on frozen SecRoBERTa.**
   - Hubness: CWE-120 and CWE-732 disappear from the top-3 lists (11.6).
   - Retrieval is mixed. Full view: Recall@5 rises in all four sources (union 0.031 → 0.043), but Hits@15 and Hits@20 are lower than before in all four sources, and Recall@10 changes by -0.001 to +0.015. Pool view: the old store is higher at most k (union Recall@10 0.184 → 0.167).
   - So the fix corrects the data, but it does not raise the scores; it moves relevant CWEs up at small k and loses some at large k. I did not investigate why. One untested guess is that mitigation and detection wording is closer to ATT&CK's own detection and mitigation prose.
2. **Dual GCN versus frozen SecRoBERTa** (Recall@10, Hits@10). Neither dual row is higher in either view.
   - Full view: InfoNCE equals frozen on Hits@10 for stix and intersection (0.129 and 0.154) and is lower on Recall@10 in all four sources; for capec and union it is lower on both (Recall@10 0.038 vs 0.070; 0.036 vs 0.066).
   - Pool view: both losses are lower on Recall@10 in all four sources. Hits@10 is lower too, except InfoNCE on stix, which ties (0.247).
3. **Which loss is higher.** InfoNCE, in every source on the full view (Recall@10 2.3x to 14x triplet's). Triplet is at random level there (0.7x to 1.3x). In the pool view the gap is smaller (Recall@10 0.127 vs 0.105 on union).
4. **Precision is far below its ceiling.** Union Precision@10 is 0.013 for frozen SecRoBERTa against a ceiling of 0.323, about 4% of it. The ceiling is below 1 because most techniques have only 1-2 gold CWEs.
5. **Recall is the main signal.** With 944 candidates and 3.4 gold CWEs per technique, random Recall@10 is 0.013 (union); frozen SecRoBERTa is about 5x that, but still finds only about 7% of the gold CWEs of a technique in its top 10.

## 6. Limits

- The dual models are single-seed (seed 42), bit-reproducible only on CPU with one thread (11.6). Differences of a few points on one source should not be read.
- Only 78 to 183 techniques per source, and 1-2 gold CWEs for most.
- The pool is built from the union gold CWEs, so the pool view is optimistic; read it next to the random rows (random Hits@20 is already 0.33 there).
- The gold relation is transitive through CAPEC ("a technique can exploit this weakness"), not equivalence; many gold CWEs are generic (Pillar or Class level), so text similarity is a weak signal for part of the pairs.
- 3 stix and union techniques are missing from the dual-GCN universe and are not scored for any method.
- Precision@k is bounded by the technique's number of gold CWEs; read it only next to the ceiling.
- The Sprint 11.4 numbers (old texts, per-pair protocol) are not comparable with these.

## 7. Commands

```
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_set_eval \
  --set evaluate.inputs.ingest_run=<11.6 ingest run of that source>
U=20261027T143236Z-91f56f41
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run $U --rows attack_ent_set \
  --set-eval set_<full|pool> --out-name sources_summary_set_<full|pool>.md \
  --source-run stix=20261027T143203Z-359b85eb --source-run capec=20261027T143214Z-460087ac \
  --source-run intersection=20261027T143225Z-d9a8711e --source-run union=$U
uv run pytest tests/unit -q
```
