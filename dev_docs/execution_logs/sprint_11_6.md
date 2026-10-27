# Sprint 11.6: Fix the CWE description bug, re-ingest, re-embed, retrain

Plan: [sprint_11.md](../sprints/sprint_11.md). Previous: [Sprint 11 log](sprint_11.md). Bug analysis: [data_analysis](../data_analysis/attack_capec_cwe_gold_alignment_analysis.md) section 6.

## 0. Outcome in brief

- **Bug fixed:** `cwe_parser._flatten_block` now reads only the weakness's direct-child `Description` / `Extended_Description`. Audit: 0 of 969 CWEs differ from the direct-child `<Description>` (was 241).
- **Ingest:** four new ingests; node, edge, text and gold-pair counts are unchanged and the gold pairs are identical. Only CWE texts changed (337 CWEs).
- **Embeddings:** the 5,131 SecRoBERTa vectors were recomputed. The ATT&CK vectors are bit-identical to the old store; 337 CWE vectors changed.
- **Hubness:** with frozen SecRoBERTa (union, top-3 lists), CWE-120 went from 77 to 0 appearances and CWE-732 from 34 to 1. The top-3 lists now hold 145 distinct CWEs instead of 124, but new hubs appear (CWE-1311: 59, CWE-1302: 51).
- **Seeding:** `gcl_train` now seeds the RNGs. Initialisation is identical across reruns. Bit-identical embeddings need CPU with one thread; on the GPU or with several threads they differ at the 1e-2 level (section 5).
- **Not evaluated yet:** this phase does not score anything. Phase 11.7 does.

## 1. Fix and tests

- `src/cga/adapters/attack_cwe/cwe_parser.py`: `_flatten_block` finds `cwe:<tag>` as a direct child of the `Weakness`. `xhtml:p` children are joined as before; a plain-text element is whitespace-normalised through `itertext()`; a missing element gives `N/A`. The detection, mitigation, consequence and example parsers are unchanged.
- `tests/unit/adapters/test_cwe_parser.py`: three new tests on an inline catalog: (a) a plain-text `<Description>` is kept when `Mitigation` / `Detection_Method` descriptions have `xhtml:p` children (and those fields still parse); (b) an `xhtml:p` description is still joined; (c) a weakness with no `<Description>` gives `N/A`, not nested text.
- Unit tests: 577 passed.
- **Audit** on `cwec_v4.20.xml`: 0 differences. Spot check: CWE-284 "The product does not restrict or incorrectly restricts access…", CWE-120 "The product copies an input buffer to an output buffer without verifying…", CWE-732 and CWE-200 are the weakness descriptions.

## 2. Re-ingest

| Gold source | Old ingest | New ingest | Gold pairs |
|---|---|---|---|
| stix | `20261026T224906Z-a0725a37` | `20261027T134239Z-8101c66c` | 287 |
| capec | `20261026T224923Z-1b4b5b3d` | `20261027T134252Z-cbfb9a99` | 566 |
| intersection | `20261026T224935Z-dee50c06` | `20261027T134304Z-d983eac7` | 221 |
| union (`LATEST`) | `20261026T224948Z-0f7e98dc` | `20261027T134320Z-32a16662` | 628 |

- Node ids, node/edge/text counts and gold pairs are identical to the old ingests for every source.
- Changed texts: 674 rows per ingest (2 variants x 337 CWEs); 0 ATT&CK texts.
- The 337 break down as 241 CWEs whose description was the wrong text, 81 whose description only changed in whitespace (the new code collapses the newlines and tabs of plain-text descriptions), and 15 more whose only change is in the extended description (19 extended descriptions changed in total, whitespace only). The whitespace-only changes are a side effect of the normalisation the plan asked for; those CWE vectors differ slightly.

## 3. Re-embed

| Store | Run |
|---|---|
| `hf_mean_pool-pt_SecRoBERTa` | `20261027T134403Z-8bc08adc` (old `…010723Z-8bc08adc`) |
| `deepwalk-deepwalk` | `20261027T135210Z-3b18ba0c` (old `…225954Z-655383ae`) |

- SecRoBERTa: 5,131 x 768, same index as the old store. ATT&CK rows identical (max difference 0.0); 337 of 969 CWE rows changed.
- DeepWalk re-run only to point at the new ingest; it does not read text.
- **Random stores and pool file kept.** `random-rand42/43/44` use the same node set. `target_pool_union.json`: regenerating it from the new ingests gives the same 151 node ids (sha256 `d66518c5ff42…`), and `--check` passed for the other three sources. The file still names the old union ingest as `source_run`; the gold pairs are identical, so nothing else needed to change.

## 4. Seeding

- `src/cga/pipelines/gcl_train/pipeline.py`: `seed_everything(ctx.seed)` before the model is built in the train step. `RunContext` only recorded the seed before; `seed_everything` was never called anywhere in `src/`.

## 5. Retrain and determinism

- `training_data`: `artifacts/training_data/attack_cwe_ent/s11b-rank_topk-k5/20261027T141039Z-90607333` (2,927 nodes, 0 held out, 14,635 examples, seed 42).
- `gcl_train` (`unsup_u2`, seed 42, 100 epochs):

| Model id | Run id | Loss epoch 1 | Loss epoch 100 | Device |
|---|---|---|---|---|
| `ent-dual-gcn-triplet-s42b` | `20261027T141524Z-5ac0fba7` | 0.7968 | 0.1241 | CPU, 1 thread |
| `ent-dual-gcn-infonce-s42b` | `20261027T141732Z-3f70b1a2` | 8.7938 | 5.1592 | CPU, 1 thread |

  Output: `models/gcl/attack_cwe_ent/<model id>/<run id>/`. `LATEST` of each model id points at these runs.
- **Determinism check** (triplet, same seed, two runs each):

  | Setting | Embeddings identical? | Max difference |
  |---|---|---|
  | Before the fix (11.2, GPU) | no | 4.01 |
  | After the fix, GPU (default `device: auto`) | no | 0.021 |
  | After the fix, GPU, 1 thread | no | 0.028 |
  | After the fix, CPU, default threads | no | 0.109 |
  | **After the fix, CPU, 1 thread** | **yes** | 0.0 |

  Epoch-1 losses are identical in all runs after the fix, so initialisation is now fixed. The remaining difference starts at the 1e-7 level (floating-point reduction order in the GPU / multi-threaded kernels) and grows over 100 epochs. The InfoNCE model was also checked on CPU with one thread: two runs identical.
- The earlier GPU runs of each model (`…141048Z-b26d4ee5`, `…141110Z-b26d4ee5` for triplet; `…141057Z-bb6dc693` for infonce) remain in the model directories but are not `LATEST`. Use the CPU runs above.
- Probe model ids created only for the determinism check (`det-probe-*`) were deleted.

## 6. Hubness check (frozen SecRoBERTa, union gold, 186 techniques, top-3 lists, all 969 CWEs)

| Store | CWE-120 | CWE-732 | Distinct CWEs | Most frequent |
|---|---|---|---|---|
| old | 77 | 34 | 124 | CWE-120 (77), CWE-805 (35), CWE-732 (34), CWE-807 (31) |
| new | 0 | 1 | 145 | CWE-1311 (59), CWE-1302 (51), CWE-1267 (35), CWE-1290 (20) |

The old hubs came from mitigation / detection text. New hubs are CWE-1311, CWE-1302 and CWE-1267; I did not check why (untested guess: CWEs with generic descriptions).

## 7. Commands

```
for k in stix capec intersection union; do
  uv run cga run ingest --dataset attack_cwe_ent --set ingest.split.fractions=[1.0,0.0,0.0] \
    --set dataset.alignments.equivalence.source=$k
done
uv run cga run embed --dataset attack_cwe_ent --set embed.embedder.kind=hf_mean_pool \
  --set embed.embedder.model_ref=jackaduma/SecRoBERTa --set embed.embedder.label=pt_SecRoBERTa \
  --set embed.inputs.text_variant=description
uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_deepwalk.yaml
uv run cga run training_data --dataset attack_cwe_ent --experiment unsup_u2 --set seed=42 \
  --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-pt_SecRoBERTa \
  --set training_data.sampler.sampler_id=s11b-rank_topk-k5
RUN=20261027T141039Z-90607333
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run cga run gcl_train --dataset attack_cwe_ent --experiment unsup_u2 \
  --set seed=42 --set device=cpu --set gcl_train.model.model_id=ent-dual-gcn-<loss>-s42b \
  --set gcl_train.inputs.training_data_sampler_id=s11b-rank_topk-k5 \
  --set gcl_train.inputs.training_data_run=$RUN --set gcl_train.loss.name=<triplet|infonce>
uv run pytest tests/unit -q
```

## 8. Notes and limits

- **The gate is met with a condition:** retrained models are bit-identical only on CPU with one thread. On the GPU, the same seed gives embeddings that differ by up to about 0.02 per component. Phase 11.7 should use the CPU runs.
- No scoring was done here. The Sprint 10/11 results on the old texts are not comparable to anything produced from the new stores; Phase 11.7 includes an old-versus-new comparison.
- Old stores, ingests and models were left in place and not deleted.
- Files changed: `src/cga/adapters/attack_cwe/cwe_parser.py`, `src/cga/pipelines/gcl_train/pipeline.py`, `tests/unit/adapters/test_cwe_parser.py`, `dev_docs/sprints/sprint_11.md`. New: this log.
