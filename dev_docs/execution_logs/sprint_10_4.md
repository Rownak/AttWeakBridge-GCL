# Sprint 10.4: Evaluate on held-out gold and report

Plan: [sprint_10.md](../sprints/sprint_10.md). Previous: [10.3](sprint_10_3.md). Decision: report only.

## 0. Outcome in brief

- **ATT&CK → CWE is nearly unsolved by frozen SecureBERT2 text alone.** Frozen SecureBERT2 gets MRR 0.015–0.020 and Hits@10 0.028–0.037 on all four gold sets (random ranking over 969 CWEs gives MRR ≈ 0.007, Hits@10 ≈ 0.010). F1 is 0.003–0.007.
- **The trained GCN does worse than frozen** on every source (MRR 0.008–0.009 against 0.015–0.020).
- **Fusion (w 0.25) is about equal to frozen:** it adds +0.001 MRR on `capec` and `union` (seed std 0.0007) and nothing on `stix` and `intersection`; its F1 is below or equal to frozen everywhere. It does not beat the frozen encoder.
- These numbers are low enough that I checked them independently (section 5). The gold is a loose, CAPEC-derived link (for example `T1021.005 VNC → CWE-309 Use of Password System for Primary Authentication`), so a text model cannot be expected to rank it high.

## 1. Runs

Config: [attack_cwe_ent_test.yaml](../../configs/experiments/attack_cwe_ent_test.yaml). Rows: 1 frozen SecureBERT2 (`ent-r1`), 2 GCN(SecureBERT2) (`ent-r2`, models `ent-g-sb2-s42/43/44` from 10.3), 3 fusion (`ent-r3`: SecureBERT2 weight 1, row 2 weight 0.25, per seed).

| Gold source | Ingest run | Evaluate run |
|---|---|---|
| stix | `20261026T224906Z-a0725a37` | `20261027T005822Z-5bc724d1` |
| capec | `20261026T224923Z-1b4b5b3d` | `20261027T005840Z-8e0b54ac` |
| intersection | `20261026T224935Z-dee50c06` | `20261027T005904Z-7ba2ece6` |
| union | `20261026T224948Z-0f7e98dc` | `20261027T005920Z-ae06ec32` |

Tables: each run's `summary.md`; the combined report is `artifacts/evaluations/attack_cwe_ent/20261027T005920Z-ae06ec32/sources_summary.md`. Protocol: `gold_alignment` (global, `sibling_filter_splits: [test]`, Hits@1/10, MRR) and `gold_alignment_prf` (top-3, greedy one-to-one, threshold 0.0, `biogitom_filters: false`).

## 2. Results (every pair is `test`; mean ± sample std over 3 seeds, the frozen row has 1)

Columns: `#Gold` counts every gold pair (P/R/F1 view); `R ceiling` = unique test techniques / `#Gold`, the most recall one-to-one assignment can reach.

### stix (#Gold 287, ranking queries 284)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecureBERT2 (frozen) | 0.0035 | 0.0282 | 0.0153 | 0.0169 | 0.0035 | 0.0058 | 59.0 | 1.0 | 0.3345 |
| GCN(SecureBERT2) | 0.0012 ± 0.0020 | 0.0153 ± 0.0124 | 0.0088 ± 0.0035 | 0.0038 ± 0.0066 | 0.0012 ± 0.0020 | 0.0018 ± 0.0031 | 89.0 ± 1.7 | 0.3 ± 0.6 | 0.3345 |
| SecureBERT2 + 0.25 × GCN (fusion) | 0.0035 ± 0.0000 | 0.0246 ± 0.0035 | 0.0152 ± 0.0002 | 0.0171 ± 0.0002 | 0.0035 ± 0.0000 | 0.0058 ± 0.0000 | 58.3 ± 0.6 | 1.0 ± 0.0 | 0.3345 |

### capec (#Gold 566, ranking queries 566)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecureBERT2 (frozen) | 0.0035 | 0.0371 | 0.0196 | 0.0225 | 0.0035 | 0.0061 | 89.0 | 2.0 | 0.3074 |
| GCN(SecureBERT2) | 0.0018 ± 0.0000 | 0.0118 ± 0.0097 | 0.0083 ± 0.0017 | 0.0062 ± 0.0002 | 0.0018 ± 0.0000 | 0.0028 ± 0.0000 | 160.3 ± 5.5 | 1.0 ± 0.0 | 0.3074 |
| SecureBERT2 + 0.25 × GCN (fusion) | 0.0053 ± 0.0018 | 0.0342 ± 0.0010 | 0.0208 ± 0.0007 | 0.0143 ± 0.0064 | 0.0024 ± 0.0010 | 0.0040 ± 0.0018 | 93.7 ± 1.5 | 1.3 ± 0.6 | 0.3074 |

### intersection (#Gold 221, ranking queries 221)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecureBERT2 (frozen) | 0.0045 | 0.0317 | 0.0179 | 0.0189 | 0.0045 | 0.0073 | 53.0 | 1.0 | 0.3529 |
| GCN(SecureBERT2) | 0.0015 ± 0.0026 | 0.0121 ± 0.0114 | 0.0079 ± 0.0036 | 0.0046 ± 0.0079 | 0.0015 ± 0.0026 | 0.0023 ± 0.0039 | 75.7 ± 2.5 | 0.3 ± 0.6 | 0.3529 |
| SecureBERT2 + 0.25 × GCN (fusion) | 0.0045 ± 0.0000 | 0.0271 ± 0.0045 | 0.0179 ± 0.0002 | 0.0190 ± 0.0002 | 0.0045 ± 0.0000 | 0.0073 ± 0.0000 | 52.7 ± 0.6 | 1.0 ± 0.0 | 0.3529 |

### union (#Gold 628, ranking queries 625)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecureBERT2 (frozen) | 0.0032 | 0.0352 | 0.0184 | 0.0110 | 0.0016 | 0.0028 | 91.0 | 1.0 | 0.2962 |
| GCN(SecureBERT2) | 0.0016 ± 0.0000 | 0.0133 ± 0.0107 | 0.0087 ± 0.0019 | 0.0060 ± 0.0002 | 0.0016 ± 0.0000 | 0.0025 ± 0.0000 | 167.0 ± 6.6 | 1.0 ± 0.0 | 0.2962 |
| SecureBERT2 + 0.25 × GCN (fusion) | 0.0048 ± 0.0016 | 0.0325 ± 0.0009 | 0.0194 ± 0.0007 | 0.0139 ± 0.0062 | 0.0021 ± 0.0009 | 0.0037 ± 0.0016 | 96.0 ± 1.0 | 1.3 ± 0.6 | 0.2962 |

### MRR / F1 per row and gold source

| Row | stix | capec | intersection | union |
|---|---|---|---|---|
| SecureBERT2 (frozen) | 0.0153 / 0.0058 | 0.0196 / 0.0061 | 0.0179 / 0.0073 | 0.0184 / 0.0028 |
| GCN(SecureBERT2) | 0.0088 / 0.0018 | 0.0083 / 0.0028 | 0.0079 / 0.0023 | 0.0087 / 0.0025 |
| SecureBERT2 + 0.25 × GCN | 0.0152 / 0.0058 | 0.0208 / 0.0040 | 0.0179 / 0.0073 | 0.0194 / 0.0037 |

## 3. Reading

- Everything is within a few correct pairs of zero: `#Correct` is 0–2 of 221–628 gold pairs. A difference of one correct pair moves R by 0.002–0.005, so F1 differences are not interpretable. Only the GCN's lower MRR and Hits@10 against frozen are consistent across all four sources.
- The GCN is trained to reproduce SecureBERT2's own text neighbourhoods (InfoNCE on SecureBERT2 top-k pairs), so it cannot add information about CAPEC-derived links, and it loses some of the frozen geometry. This matches NCIT–DOID (9.3, 9.5), where trained rows were also at or below frozen.
- The fusion weight keeps the encoder dominant, so fusion tracks frozen (std ≤ 0.0007 MRR across seeds).
- The gold is noisy for a text matcher: each technique has about 3 CWEs reached through CAPEC, often by a loose link, and ATT&CK descriptions share little vocabulary with CWE descriptions.

## 4. Disclosures

- **Gold reads:** `test` was read by the four evaluate runs, one per source, plus **one repeat** of `union` (`20261027T010010Z-ae06ec32`) for the determinism check. Gold was read by nothing else; training used `held_out_policy: none`.
- **Fixed fusion weight:** w = 0.25 comes from NCIT–DOID (9.5) and was not tuned here.
- **Recall ceiling:** one-to-one assignment gives each source at most one prediction; with about 3 gold CWEs per technique, R is capped at 0.30–0.35 (column `R ceiling`).
- **Gold-set sizes:** 287 (`stix`), 566 (`capec`), 221 (`intersection`), 628 (`union`) pairs after id resolution; the four sets are separate and not pooled.
- **Pseudo-pairs:** they come from SecureBERT2.
- **Common query set:** the ranking metrics use only pairs every method can score. Rows 2 and 3 were trained on the 2,927 nodes that have both a SecureBERT2 and a DeepWalk vector (10.3); three techniques missing from that set remove 3 pairs from `stix` and `union`, so those rankings use 284 and 625 queries for all three rows. P/R/F1 keep all pairs in `#Gold` (the missing ones count as unpredicted).
- **Frozen row seeds:** frozen SecureBERT2 has no seed, so it has no std.

## 5. Checks (gate 10.4)

- **Determinism:** the `union` evaluation repeated (`20261027T010010Z-ae06ec32`) gives 84 metric rows identical to the first (max absolute difference 0.0), the frozen rows included.
- **Fusion:** `test_fusion_matches_its_explicit_weighted_mean_scores` (existing) confirms fusion reproduces the weighted member cosines.
- **Independent ranking check:** I ranked each of the 628 `union` pairs in numpy straight from the SecureBERT2 store (cosine over all 969 CWEs, no sibling filter): MRR 0.0181, Hits@10 0.032, median rank 313. The evaluator gives MRR 0.0184, Hits@10 0.035 (it also filters sibling gold targets). The gap is the sibling filter. A sample of eight gold pairs matches the CAPEC links.
- **Metric-correctness test:** `tests/unit/pipelines/evaluate/test_one_to_many_gold.py` (3 tests) uses a hand-computed 1:n fixture: Hits@1 2/3, Hits@2 1, MRR 5/6 under the test-only sibling filter, MRR 1 under the all-splits filter, and one-to-one P 1, R 2/3, F1 4/5, `n_gold` 3, recall at the sources/pairs ceiling.
- All 565 unit tests pass.

## 6. Files

New: `configs/experiments/attack_cwe_ent_test.yaml`, `tests/unit/pipelines/evaluate/test_one_to_many_gold.py`. Changed: `scripts/unsup_summary.py` (`--rows attack_ent`, `--source-run NAME=RUN`, `build_source_report`), `tests/unit/test_unsup_summary.py` (+1 test), `dev_docs/sprints/sprint_10.md` (checkboxes).

## 7. Commands

```
for pair in stix=20261026T224906Z-a0725a37 capec=20261026T224923Z-1b4b5b3d \
            intersection=20261026T224935Z-dee50c06 union=20261026T224948Z-0f7e98dc; do
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_test \
    --set evaluate.inputs.ingest_run=${pair#*=}
done
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run <evaluate run> --rows attack_ent
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run 20261027T005920Z-ae06ec32 \
  --source-run stix=20261027T005822Z-5bc724d1 --source-run capec=20261027T005840Z-8e0b54ac \
  --source-run intersection=20261027T005904Z-7ba2ece6 --source-run union=20261027T005920Z-ae06ec32
uv run pytest tests/unit -q
```
