# Sprint 10.5: SecRoBERTa as the text encoder

Plan: [sprint_10.md](../sprints/sprint_10.md). Previous: [10.4](sprint_10_4.md). Decision: report only.

## 0. Outcome in brief

- **SecRoBERTa is a better frozen encoder than SecureBERT2 here, on all four gold sources:** MRR 0.023–0.029 against 0.015–0.020 (Δ +0.005 to +0.011); Hits@10 0.048–0.063 against 0.028–0.037. Both are still far from usable: random ranking gives MRR ≈ 0.007 and Hits@10 ≈ 0.010.
- **The 10.4 pattern repeats:** the trained GCN is worse than frozen (MRR 0.011–0.015 against 0.023–0.029), and fusion (w 0.25) tracks frozen and does not beat it on MRR (0.020–0.024 against 0.023–0.029).
- **F1 gaps are one or two correct pairs.** `#Correct` is 2–4 of 221–628 gold pairs for every row, so F1 and Hits@1 differences are not interpretable. The MRR and Hits@10 differences, which use every pair, are the stable signal.

## 1. What was done

- **Embed:** `hf_mean_pool`, `jackaduma/SecRoBERTa`, label `pt_SecRoBERTa`, text variant `description`, `embed.yaml` defaults (`max_length 512`, mean pooling, `batch_size 8`). Run `20261027T010723Z-8bc08adc` (embedder id `hf_mean_pool-pt_SecRoBERTa`) on the `attack_cwe_ent` lineage. Log `logs/sprint_10_5_embed.log`.
- **Train:** priority `8` of `scripts/run_unsup_matrix.sh` (the 10.3 commands with row id `g-srb`). Pseudo-pairs: SecRoBERTa text-cosine top-k, cross-graph (k 5), sampler `srb-rank_topk-k5`, `held_out_policy: none`. Model: `residual_adapter`, `adapter: gcn`, `residual_floor: false`, InfoNCE (temperature 0.1), Adam 1e-3, 100 epochs, last epoch. Log `logs/sprint_10_5_train.log`.
- **Evaluate:** `configs/experiments/attack_cwe_ent_srb_test.yaml`, one run per gold source, same prediction protocol as 10.4.
- **Tables:** `scripts/unsup_summary.py --rows attack_ent_srb`, plus an encoder comparison against the 10.4 runs.

## 2. Runs

| Item | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| `training_data` (`srb-rank_topk-k5`) | `20261027T011601Z-fd838625` | `20261027T011627Z-84135849` | `20261027T011655Z-e8a7400a` |
| `gcl_train` (`ent-g-srb-s<seed>`) | `20261027T011611Z-27f660cb` | `20261027T011636Z-34a3c601` | `20261027T011706Z-e523038c` |
| Final loss (epoch 100) | 2.7590 | 2.6096 | 2.6962 |

Manifests: `protocol: unsupervised`, `held_out_policy: none`, 14,635 pseudo-pairs per seed. The final loss is about ten times higher than SecureBERT2's (≈ 0.24): SecRoBERTa vectors are less separable under InfoNCE with temperature 0.1.

| Gold source | Ingest run | Evaluate run |
|---|---|---|
| stix | `20261026T224906Z-a0725a37` | `20261027T011727Z-4d2bfc86` |
| capec | `20261026T224923Z-1b4b5b3d` | `20261027T011738Z-f47b2218` |
| intersection | `20261026T224935Z-dee50c06` | `20261027T011750Z-27e45319` |
| union | `20261026T224948Z-0f7e98dc` | `20261027T011800Z-9426c1a6` |

Combined report: `artifacts/evaluations/attack_cwe_ent/20261027T011800Z-9426c1a6/sources_summary.md`.

## 3. Store and universe

- **Store:** 5,131 of 5,840 nodes embedded; 709 nodes have empty text (the same 709 as SecureBERT2: the `x-mitre-detection-strategy` objects and four other non-technique objects). Every gold technique and CWE has a vector.
- **Universe:** `intersection` of the SecRoBERTa and DeepWalk stores, **2,927 nodes**, the same size as in 10.3.

| Source | Pairs | Lost to the universe | Techniques / CWEs lost |
|---|---|---|---|
| stix | 287 | 3 | 3 / 2 |
| capec | 566 | 0 | 0 / 0 |
| intersection | 221 | 0 | 0 / 0 |
| union | 628 | 3 | 3 / 2 |

The ranking metrics use a common query set across the three rows: 284 queries for `stix`, 625 for `union`, all pairs for `capec` and `intersection`. P/R/F1 keep every pair in `#Gold` (missing ones count as unpredicted).

## 4. Results (every pair is `test`; mean ± sample std over 3 seeds, frozen row has 1)

`R ceiling` = unique test techniques / `#Gold`, the most recall one-to-one assignment can reach.

### stix (#Gold 287)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecRoBERTa (frozen) | 0.0070 | 0.0528 | 0.0251 | 0.0328 | 0.0070 | 0.0115 | 61.0 | 2.0 | 0.3345 |
| GCN(SecRoBERTa) | 0.0023 ± 0.0041 | 0.0188 ± 0.0073 | 0.0109 ± 0.0041 | 0.0109 ± 0.0109 | 0.0035 ± 0.0035 | 0.0053 ± 0.0053 | 92.0 ± 0.0 | 1.0 ± 1.0 | 0.3345 |
| SecRoBERTa + 0.25 × GCN (fusion) | 0.0070 ± 0.0000 | 0.0329 ± 0.0020 | 0.0199 ± 0.0006 | 0.0314 ± 0.0003 | 0.0070 ± 0.0000 | 0.0114 ± 0.0000 | 63.7 ± 0.6 | 2.0 ± 0.0 | 0.3345 |

### capec (#Gold 566)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecRoBERTa (frozen) | 0.0053 | 0.0530 | 0.0253 | 0.0345 | 0.0053 | 0.0092 | 87.0 | 3.0 | 0.3074 |
| GCN(SecRoBERTa) | 0.0029 ± 0.0020 | 0.0253 ± 0.0020 | 0.0148 ± 0.0018 | 0.0103 ± 0.0095 | 0.0029 ± 0.0027 | 0.0046 ± 0.0042 | 161.3 ± 2.3 | 1.7 ± 1.5 | 0.3074 |
| SecRoBERTa + 0.25 × GCN (fusion) | 0.0071 ± 0.0000 | 0.0436 ± 0.0027 | 0.0241 ± 0.0006 | 0.0433 ± 0.0109 | 0.0071 ± 0.0018 | 0.0122 ± 0.0030 | 92.3 ± 0.6 | 4.0 ± 1.0 | 0.3074 |

### intersection (#Gold 221)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecRoBERTa (frozen) | 0.0090 | 0.0633 | 0.0286 | 0.0385 | 0.0090 | 0.0147 | 52.0 | 2.0 | 0.3529 |
| GCN(SecRoBERTa) | 0.0030 ± 0.0052 | 0.0226 ± 0.0090 | 0.0127 ± 0.0053 | 0.0130 ± 0.0130 | 0.0045 ± 0.0045 | 0.0067 ± 0.0067 | 77.3 ± 0.6 | 1.0 ± 1.0 | 0.3529 |
| SecRoBERTa + 0.25 × GCN (fusion) | 0.0090 ± 0.0000 | 0.0377 ± 0.0026 | 0.0223 ± 0.0004 | 0.0361 ± 0.0004 | 0.0090 ± 0.0000 | 0.0145 ± 0.0000 | 55.3 ± 0.6 | 2.0 ± 0.0 | 0.3529 |

### union (#Gold 628)

| Variant | Hits@1 | Hits@10 | MRR | P | R | F1 | #Pred | #Correct | R ceiling |
|---|---|---|---|---|---|---|---|---|---|
| SecRoBERTa (frozen) | 0.0048 | 0.0480 | 0.0234 | 0.0326 | 0.0048 | 0.0083 | 92.0 | 3.0 | 0.2962 |
| GCN(SecRoBERTa) | 0.0027 ± 0.0018 | 0.0235 ± 0.0024 | 0.0138 ± 0.0017 | 0.0099 ± 0.0091 | 0.0027 ± 0.0024 | 0.0042 ± 0.0038 | 168.7 ± 2.1 | 1.7 ± 1.5 | 0.2962 |
| SecRoBERTa + 0.25 × GCN (fusion) | 0.0064 ± 0.0000 | 0.0395 ± 0.0024 | 0.0223 ± 0.0005 | 0.0420 ± 0.0105 | 0.0064 ± 0.0016 | 0.0111 ± 0.0028 | 95.3 ± 0.6 | 4.0 ± 1.0 | 0.2962 |

### MRR / F1 per row and gold source

| Row | stix | capec | intersection | union |
|---|---|---|---|---|
| SecRoBERTa (frozen) | 0.0251 / 0.0115 | 0.0253 / 0.0092 | 0.0286 / 0.0147 | 0.0234 / 0.0083 |
| GCN(SecRoBERTa) | 0.0109 / 0.0053 | 0.0148 / 0.0046 | 0.0127 / 0.0067 | 0.0138 / 0.0042 |
| SecRoBERTa + 0.25 × GCN | 0.0199 / 0.0114 | 0.0241 / 0.0122 | 0.0223 / 0.0145 | 0.0223 / 0.0111 |

### SecureBERT2 (10.4) against SecRoBERTa: MRR and F1 of the seed means

| Row | Source | SecureBERT2 MRR | SecRoBERTa MRR | Δ MRR | SecureBERT2 F1 | SecRoBERTa F1 | Δ F1 |
|---|---|---|---|---|---|---|---|
| frozen | stix | 0.0153 | 0.0251 | +0.0099 | 0.0058 | 0.0115 | +0.0057 |
| frozen | capec | 0.0196 | 0.0253 | +0.0057 | 0.0061 | 0.0092 | +0.0031 |
| frozen | intersection | 0.0179 | 0.0286 | +0.0107 | 0.0073 | 0.0147 | +0.0074 |
| frozen | union | 0.0184 | 0.0234 | +0.0050 | 0.0028 | 0.0083 | +0.0056 |
| GCN | stix | 0.0088 | 0.0109 | +0.0021 | 0.0018 | 0.0053 | +0.0035 |
| GCN | capec | 0.0083 | 0.0148 | +0.0065 | 0.0028 | 0.0046 | +0.0018 |
| GCN | intersection | 0.0079 | 0.0127 | +0.0048 | 0.0023 | 0.0067 | +0.0044 |
| GCN | union | 0.0087 | 0.0138 | +0.0051 | 0.0025 | 0.0042 | +0.0017 |
| fusion | stix | 0.0152 | 0.0199 | +0.0047 | 0.0058 | 0.0114 | +0.0056 |
| fusion | capec | 0.0208 | 0.0241 | +0.0033 | 0.0040 | 0.0122 | +0.0081 |
| fusion | intersection | 0.0179 | 0.0223 | +0.0044 | 0.0073 | 0.0145 | +0.0072 |
| fusion | union | 0.0194 | 0.0223 | +0.0029 | 0.0037 | 0.0111 | +0.0074 |

## 5. Reading

- SecRoBERTa is ahead of SecureBERT2 on every row and source in MRR and F1, with the same sign everywhere. The gap is small in absolute terms (about 0.005–0.011 MRR for frozen) and rests on a few pairs, so it supports "SecRoBERTa is somewhat better on this gold" and not more. I did not pick an encoder on this: both are reported.
- The GCN loses to frozen under both encoders. It is trained to reproduce the encoder's own neighbourhoods on pseudo-pairs, so it adds no link to the CAPEC-derived gold. The ordering frozen > fusion > GCN is the same for both encoders.
- Fusion with SecRoBERTa raises precision and F1 slightly on `capec` and `union` (F1 0.012 and 0.011 against frozen 0.009 and 0.008) at the cost of MRR; with four correct pairs against three, this is within noise.
- The higher final training loss for SecRoBERTa (≈ 2.7 against ≈ 0.24) shows the pseudo-pair task is harder for this encoder; the GCN does not do better because of it.

## 6. Disclosures

- **Gold reads:** `test` was read by four more evaluate runs, one per source, plus one repeat of `union` (`20261027T011830Z-9426c1a6`) for the determinism check: eight source reads in total with 10.4, plus two repeats. Neither encoder was selected using gold; both are reported.
- **Fixed settings:** fusion weight 0.25 (from NCIT–DOID 9.5) and all training settings are those of earlier sprints, not tuned for SecRoBERTa.
- **Recall ceiling, gold-set sizes, common query set:** as in 10.4 (columns `R ceiling`, `#Gold`; section 3).
- **Pseudo-pairs:** they come from SecRoBERTa.
- **Encoder differences:** SecRoBERTa (`jackaduma/SecRoBERTa`) and SecureBERT2 (`cisco-ai/SecureBERT2.0-base`) differ in pre-training corpus; both store 5,131 vectors, 768 d.
- **Embedding config:** the SecRoBERTa store uses the same `embed.yaml` defaults as the SecureBERT2 store of 10.2. Loading `jackaduma/SecRoBERTa` printed the standard Hugging Face "UNEXPECTED keys" note about unused task-head weights (ignored when loading from a different architecture).

## 7. Checks (gate 10.4 for the SecRoBERTa runs)

- **Determinism:** the repeated `union` evaluation gives 84 metric rows identical to the first (max absolute difference 0.0), frozen rows included.
- **Fusion:** the existing fusion unit test (`test_fusion_matches_its_explicit_weighted_mean_scores`) covers the fusion method; no code changed.
- **Metric correctness:** the 10.4 fixture test still passes. 566 unit tests pass.

## 8. Files

New: `configs/experiments/attack_cwe_ent_srb_test.yaml`. Changed: `scripts/run_unsup_matrix.sh` (priority 8; priority 7 reformatted onto continuation lines), `scripts/unsup_summary.py` (`--rows attack_ent_srb`, `--compare-source-run`, `--compare-rows`, `build_encoder_comparison`, `rows` argument of `build_source_report`), `tests/unit/test_unsup_summary.py` (+1 test), `dev_docs/sprints/sprint_10.md` (checkboxes).

## 9. Commands

```
uv run cga run embed --dataset attack_cwe_ent --set embed.embedder.kind=hf_mean_pool \
  --set embed.embedder.model_ref=jackaduma/SecRoBERTa --set embed.embedder.label=pt_SecRoBERTa \
  --set embed.inputs.text_variant=description
TEXT_EMBEDDER=hf_mean_pool-pt_SecRoBERTa PREFIX=ent- SAMPLER_PREFIX=srb- \
  bash scripts/run_unsup_matrix.sh attack_cwe_ent 8
for pair in stix=20261026T224906Z-a0725a37 capec=20261026T224923Z-1b4b5b3d \
            intersection=20261026T224935Z-dee50c06 union=20261026T224948Z-0f7e98dc; do
  uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_srb_test \
    --set evaluate.inputs.ingest_run=${pair#*=}
done
uv run python scripts/unsup_summary.py --dataset attack_cwe_ent --run 20261027T011800Z-9426c1a6 \
  --rows attack_ent_srb --text-label SecRoBERTa --compare-label SecureBERT2 --compare-rows attack_ent \
  --source-run stix=20261027T011727Z-4d2bfc86 --source-run capec=20261027T011738Z-f47b2218 \
  --source-run intersection=20261027T011750Z-27e45319 --source-run union=20261027T011800Z-9426c1a6 \
  --compare-source-run stix=20261027T005822Z-5bc724d1 --compare-source-run capec=20261027T005840Z-8e0b54ac \
  --compare-source-run intersection=20261027T005904Z-7ba2ece6 --compare-source-run union=20261027T005920Z-ae06ec32
uv run pytest tests/unit -q
```
