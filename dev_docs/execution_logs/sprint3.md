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
