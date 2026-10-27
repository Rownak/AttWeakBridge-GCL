# Sprint 10.2: Ingest and embeddings

## What was done

- **Dataset config** `configs/datasets/attack_cwe_ent.yaml` (created in 10.1): `adapter: attack_cwe`, ATT&CK `enterprise_cwe/enterprise-attack-20.0.json`, CWE `enterprise_cwe/cwec_v4.20.xml` (`cwec_v4.20`), the two text variants of `attack_cwe.yaml`, `protocol: unsupervised`, and the `alignments` block (default split `[0, 0, 1]`).
- **Four ingests into one lineage** `data/processed/attack_cwe_ent/`, one per gold source, each with `ingest.split.fractions=[1.0,0.0,0.0]`; `union` last, so `LATEST` = `20261026T224948Z-0f7e98dc`.
- **Two embedding stores** in `artifacts/embeddings/attack_cwe_ent/`, both built from the `union` ingest.
- **One bug fixed on the way** (see below); the first set of ingests and stores was deleted and rebuilt.

## Ingest runs

| Source | Ingest run id |
|---|---|
| stix | `20261026T224906Z-a0725a37` |
| capec | `20261026T224923Z-1b4b5b3d` |
| intersection | `20261026T224935Z-dee50c06` |
| union (`LATEST`) | `20261026T224948Z-0f7e98dc` |

## Gate 10.2

`nodes`, `edges` and `texts` parquet hashes are equal across the four runs (sha256 prefixes `9983df13daaf`, `dec3d851e5f7`, `72645370ccad`; 5,840 nodes, 22,979 edges). Only `alignments` and `alignment_provenance` differ. Embeddings and models are therefore built once and scored against every source. Resolved and dropped counts are logged per source below.

## Embeddings

| Store | Run id | Rows | Dim | Notes |
|---|---|---|---|---|
| `hf_mean_pool-SecureBERT2` | `20261026T225000Z-4f1f6686` | 5,131 of 5,840 | 768 | `cisco-ai/SecureBERT2.0-base`, text variant `description`; 709 nodes have empty text (705 `x-mitre-detection-strategy`, 4 other non-technique objects) |
| `deepwalk-deepwalk` | `20261026T225954Z-655383ae` | 3,632 of 5,840 | 128 | `configs/pipelines/embed_deepwalk.yaml` (the Sprint 9 settings: 128 d, walk 30, 200 walks, window 5, 10 epochs, 1 worker); 2,208 nodes skipped (no edges to walk) |

All vectors are finite float32. The earlier Sprint 3 SecureBERT2 run did not record its config, so I used the `embed.yaml` defaults (`max_length 512`, `chunking fixed`, `mean_last_hidden`, `add_special_tokens false`, `batch_size 8`) plus the model, label and variant above. If the Sprint 3 store used other settings, these two stores differ in that respect.

## Coverage table

CSV rows → pairs kept; drops by reason; gold endpoints missing from each store. All kept pairs are `test` (default split).

| Source | CSV rows | Pairs kept | Techniques | CWEs | Dup rows | Dropped: source / target | Pairs lost to SecureBERT2 | Pairs lost to DeepWalk |
|---|---|---|---|---|---|---|---|---|
| stix | 376 | 287 | 96 | 77 | 69 | 0 / 20 | 0 | 3 (3 techniques) |
| capec | 657 | 566 | 174 | 149 | 49 | 1 / 41 | 0 | 0 |
| intersection | 233 | 221 | 78 | 68 | 3 | 0 / 9 | 0 | 0 |
| union | 800 | 628 | 186 | 151 | 119 | 1 / 52 | 0 | 3 (3 techniques) |

- Source drops: the one mobile technique (`capec`, `union`). Target drops: CWEs absent from v4.20.
- Dropped counts are per CSV row; "Dup rows" repeat an already-kept pair (another CAPEC path) and only add to its provenance.
- **Consequence for 10.4:** for `stix` and `union`, DeepWalk-based rows (4, 5 and fusions that use row 5) have no vector for 3 gold techniques, so those 3 pairs cannot be ranked there. Report `n_gold` per row, or say so in the table.

## Bug found and fixed

The gold reader initially mapped each CSV id to the first node with that `external_id`. 224 ATT&CK ids are shared by old `course-of-action` mitigations and techniques, so 147 of 628 `union` pairs pointed at a mitigation node. I only noticed it because the first DeepWalk store seemed to lack 36–51 gold techniques per source. Fix in `AttackCweAdapter.iter_alignments`: only `attack-pattern` nodes are gold sources, and ids still shared by several candidates are dropped and counted, never guessed. A test was added (561 unit tests pass). I deleted the first four ingests and both stores (all created in this session) and rebuilt them; pair counts did not change. Details are in `sprint_10_1.md`.

## Files

New or changed in this phase: `src/cga/adapters/attack_cwe/adapter.py` (fix), `tests/unit/pipelines/ingest/test_gold_alignments.py` (new test), `dev_docs/execution_logs/sprint_10_1.md` (correction note), `dev_docs/sprints/sprint_10.md` (checkboxes). Data outputs: `data/processed/attack_cwe_ent/`, `artifacts/embeddings/attack_cwe_ent/`.

## Commands

```
for k in stix capec intersection union; do
  uv run cga run ingest --dataset attack_cwe_ent \
    --set ingest.split.fractions=[1.0,0.0,0.0] \
    --set dataset.alignments.equivalence.source=$k
done
uv run cga run embed --dataset attack_cwe_ent \
  --set embed.embedder.kind=hf_mean_pool --set embed.embedder.model_ref=cisco-ai/SecureBERT2.0-base \
  --set embed.embedder.label=SecureBERT2 --set embed.inputs.text_variant=description
uv run cga run embed --dataset attack_cwe_ent --config configs/pipelines/embed_deepwalk.yaml
uv run pytest tests/unit -q
```
