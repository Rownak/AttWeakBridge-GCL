# Sprint 3 — Execution Log

## Phase 3.0 — Ingest without alignments, and data sanity check

### What changed
- [configs/datasets/ncit_doid.yaml](../../configs/datasets/ncit_doid.yaml): the `alignments:` block is
  commented out (with a note explaining why and that the gold-set sprint should restore it). Without it,
  `training_data` cannot mark `test`-split alignment endpoints as `held_out`, so the only held-out set
  is ingest's random 80/10/10 node split (same as `attack_cwe`).
- No code changes. Adapter already supports `alignments: None`.

### Ingest run
Run `20261009T164439Z-2d34c6da` (`data/processed/ncit_doid/`), ~15 s.

| Stat | NCIT (graph_1) | DOID (graph_2) | Doc expectation |
|---|---|---|---|
| nodes | 15,762 | 8,465 | 15,762 / 8,465 (nothing dropped) |
| `subClassOf` edges | 9,364 | 9,388 | 16,030 / 9,388 |
| dangling edges dropped | 0 | 0 | |
| split train/val/test | 12,610 / 1,576 / 1,576 | 6,772 / 846 / 847 | |

Totals: 24,227 nodes, 18,752 edges, 72,681 text rows (3 variants x 24,227), `n_alignments == 0`,
mean degree 2.29, max degree 504.

**NCIT edge count differs from the doc, and is correct.** Direct parent pointers
(`<rdfs:subClassOf rdf:resource=.../>`) in `ncit.owl` number 9,364, which matches ingest exactly.
A further 3,333 are `<rdfs:subClassOf>` blocks wrapping `owl:Restriction` logical axioms (out of
scope; "NCIT logical axioms as edges" is deferred). Those two sum to 12,697, the total opening-tag
count, so the doc's 16,030 could not be reproduced by any grep; its origin is unknown (possibly
counted differently, e.g. including closing tags). No `owl:Thing` parents are present, 0 edges were
dropped. Treat 9,364 as the true direct is-a count.

### Text coverage
Every node has all three variants (it falls back to the label), so coverage is measured from the text
content. Approximate: derived by comparing variant string lengths, not by re-parsing the OWL.

| Side | has definition | has synonym/definition beyond label |
|---|---|---|
| NCIT | 85.3% | ~100% (synonyms include the label itself, so this overstates true extra synonyms) |
| DOID | **1.6%** (~135 of 8,465) | ~37% (about 36% have an exact synonym) |

DOID definitions are rare in the file itself (138 `IAO_0000115` occurrences, 2 are property
declarations), so this is a data property, not a parsing bug. Consequence: for DOID,
`label_synonyms_definition` is effectively `label + synonyms`; the NCIT side is much richer. This
asymmetry may affect text-based cross-graph similarity and is worth remembering when reading results.
No node has an empty label.

### Spot check: NCIT `C3224` (Melanoma)
Label `Melanoma`; `label_synonyms_definition` = label + synonyms (`MELANOMA, MALIGNANT`,
`Malignant Melanoma`, `Melanoma`, `melanoma`) + four definitions (P97/P325), joined with `; `. Correct.
DOID sample (`angiosarcoma`) -> `angiosarcoma; hemangiosarcoma`, no definition.

### Scale check
Deferred to Phase 3.2 as specified (record peak RAM of `training_data`).

### Tests / commands
```
uv run cga run ingest --dataset ncit_doid --dry-run
uv run cga run ingest --dataset ncit_doid
uv run pytest tests/unit -q        # 372 passed
```

### Notes
- Windows hardlink warning from `uv` is harmless; `UV_LINK_MODE=copy` silences it.

## Phase 3.1 — Embeddings

### What was run
Stores under `artifacts/embeddings/ncit_doid/` (`text_variant=label_synonyms_definition` for text):

| Store (`embedder_id`) | Run | Vectors | `ok` | `skipped` | `failed` | Runtime |
|---|---|---|---|---|---|---|
| `hf_mean_pool-sapbert` | `20261009T182919Z-95011340` | 24,227 x 768 | 24,227 | 0 | 0 | ~23 min (batch 8, killed at 20,009/24,227 nodes) + 958 s resume (batch 512) |
| `hf_mean_pool-bert_base` | `20261009T191700Z-1278c23a` | 24,227 x 768 | 24,227 | 0 | 0 | 3,249 s (batch 512) |
| `deepwalk-deepwalk` | `20261009T201117Z-a544b033` | 16,386 x 128 | 16,386 | **7,841** | 0 | 4,013 s (`workers: 1`) |

node2vec was not run (optional; required again in Phase 3.5).

### Findings
- **No text failures.** Every node has non-empty text in both text stores, so there are no empty-text
  classes (this also confirms every DOID node has at least a label).
- **DeepWalk skipped 7,841 nodes, all in NCIT (49.7% of NCIT; 0 of DOID).** `skipped` equals exactly the set of
  nodes with no `subClassOf` edge (degree 0), so DeepWalk has no walk for them. These nodes' parents
  are expressed only via restriction/logical axioms (out of scope per the sprint), or they are roots.
  **Impact on Phase 3.2:** the graph-embedding store has no vector for half of NCIT; how `training_data`
  handles missing graph vectors (graph similarity, universe size) must be checked before relying on it.
  This was not changed here; raise it if `training_data` drops or errors on these nodes.
- **Speed.** Text embedding was CPU-bound (tokenizing/chunking and one shard per node), not GPU-bound:
  GPU memory stayed ~1 GB with batch 8, and batch 512 helped only partly (bert_base still took ~54 min).
  Throughput improvements would need a code change; none were made.

### Challenges / deviations
- The first SapBERT run (batch 8) was stopped after ~23 min at 20,009 of 24,227 shards. It was **resumed**
  with `--set embed.resume=20261009T182919Z-95011340 --set embed.embedder.batch_size=512`; completed
  shards were kept. Resume wrote a second, empty run dir (`...185331Z-390d8bc3`, manifest and log only)
  and pointed `LATEST` at it, which has no `vectors.npy`. **`LATEST` for `hf_mean_pool-sapbert` was
  manually set to `20261009T182919Z-95011340`** so `latest` resolves to the real store. Consider fixing
  `--resume` to update `LATEST` to the resumed store (code change, not done).
- An earlier interrupted DeepWalk attempt left `deepwalk-deepwalk/20261009T185242Z-a544b033/` (log only);
  it is ignored by `LATEST` and can be deleted.
- `batch_size=512` was passed via `--set`; `configs/pipelines/embed.yaml` is unchanged (still 8).

### Commands
```
uv run cga run embed --dataset ncit_doid --set embed.inputs.text_variant=label_synonyms_definition --set embed.embedder.model_ref=cambridgeltl/SapBERT-from-PubMedBERT-fulltext --set embed.embedder.label=sapbert --set embed.embedder.batch_size=512
uv run cga run embed --dataset ncit_doid --set embed.inputs.text_variant=label_synonyms_definition --set embed.embedder.model_ref=bert-base-uncased --set embed.embedder.label=bert_base --set embed.embedder.batch_size=512
uv run cga run embed --dataset ncit_doid --config configs/pipelines/embed_deepwalk.yaml
```
