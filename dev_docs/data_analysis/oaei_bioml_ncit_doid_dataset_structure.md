# OAEI Bio-ML `ncit-doid` Dataset Structure & Suitability Analysis

Source: `benchmark/OAEI-Bio-ML` (Bio-ML track of the Ontology Alignment Evaluation Initiative, He et al., ISWC 2022). This document describes the `ncit-doid` task's current on-disk structure and evaluates its conceptual fit against a two-disconnected-text-attributed-graphs cross-graph-alignment setting (e.g. ATT&CK–CWE). It does not propose implementation or refactoring changes.

## Directory Layout (as present on disk)

```
benchmark/OAEI-Bio-ML/
├── README.md
├── documentation.md
└── ncit-doid/
    ├── ncit.owl                  # source ontology: NCI Thesaurus (55 MB)
    ├── doid.owl                  # target ontology: Disease Ontology (6.4 MB)
    ├── refs_equiv/
    │   ├── full.tsv              # all gold equivalence mappings (4,686 rows incl. header)
    │   ├── train.tsv             # equivalence mapping training split (1,406 rows)
    │   ├── test.tsv              # equivalence mapping test split (3,280 rows)
    │   └── test.cands.tsv        # ranking candidates for each test mapping
    └── refs_subs/
        ├── train.tsv             # gold subsumption mapping training split
        └── test.cands.tsv        # ranking candidates for subsumption test mappings
```

Unlike `MovieGraphBenchmark`, this data is present directly in the repo (not fetched/generated on demand), so both ontologies and all reference mapping files are immediately inspectable.

## Graph/Ontology Structure

Each `.owl` file is an OWL/XML ontology: a flat list of `owl:Class` elements (plus `owl:AnnotationProperty`, `owl:ObjectProperty`, and `owl:Axiom` metadata blocks), not a triple-store dump. Each `owl:Class` block is keyed by `rdf:about="<class IRI>"` and contains:

- **Structural (edge) axioms** inside the class block: `rdfs:subClassOf rdf:resource="<parent IRI>"` (direct is-a edges) and, in NCIT, more complex `owl:equivalentClass` / `owl:intersectionOf` / `owl:Restriction` blocks expressing logical-definition axioms (e.g. "Melanoma ≡ Neoplasm ⊓ hasSite.SomeTissue") rather than a plain parent pointer.
- **Text/annotation properties** inside the same class block: human-readable labels, synonyms, and definitions as annotation-property literals.

### DOID (target ontology) — representative class (`DOID_4362`)
```xml
<owl:Class rdf:about="http://purl.obolibrary.org/obo/DOID_4362">
    <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/DOID_120"/>
    <oboInOwl:hasExactSynonym ...>cervix cancer</oboInOwl:hasExactSynonym>
    <oboInOwl:hasExactSynonym ...>uterine cervical neoplasm</oboInOwl:hasExactSynonym>
    <rdfs:label ...>cervical cancer</rdfs:label>
</owl:Class>
```
DOID classes are simpler: one primary `rdfs:subClassOf` parent edge, plus `rdfs:label` and `oboInOwl:hasExactSynonym` for text. No free-text definition field was observed in the sampled class (DOID does carry `IAO_0000115` definitions on many classes elsewhere in the file, but not universally).

### NCIT (source ontology) — representative class (`C3224`, "Melanoma")
```xml
<owl:Class rdf:about="http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3224">
    <owl:equivalentClass> ... intersectionOf / Restriction ... </owl:equivalentClass>
    <P90>Melanoma</P90>                 <!-- synonym -->
    <P90>Malignant Melanoma</P90>
    <P97>A malignant, usually aggressive tumor composed of atypical, neoplastic melanocytes...</P97>  <!-- definition -->
    <P325>A malignant neoplasm composed of melanocytes.</P325>  <!-- alternate definition -->
    <rdfs:label>Melanoma</rdfs:label>   <!-- preferred name -->
</owl:Class>
```
NCIT classes use custom NCIT-specific annotation property IDs (`P90` = synonym, `P97`/`P325` = definition, `P106`/`P108` = semantic-type-like fields, `A8`/`A11`/`A14` = cross-references to other classes) rather than the OBO-standard `oboInOwl:*` predicates used by DOID. `rdfs:label` is present uniformly as the canonical name. Direct `rdfs:subClassOf` edges also exist in NCIT (16,030 occurrences across the file) alongside the logical-definition axioms — so simple parent/child structure is still extractable even though richer OWL semantics exist.

### Edge and class counts (from local grep, this file)
- NCIT: 15,762 `owl:Class` declarations; 16,030 `rdfs:subClassOf` edges.
- DOID: 8,465 `owl:Class` declarations; 9,388 `rdfs:subClassOf` edges.
- (Note: the OAEI Bio-ML 2022 paper's published statistics table lists 6,835 NCIT / 8,448 DOID classes for the *pruned* ncit-doid task; the raw counts above are from the full `.owl` files as stored in this repo and may include classes outside the pruned/aligned subset.)

## Text Attributes Available

Both ontologies expose per-class textual fields usable as "node text," though under different predicate names:

| Text role | NCIT predicate | DOID predicate |
|---|---|---|
| Canonical name/label | `rdfs:label` | `rdfs:label` |
| Synonyms | `P90` | `oboInOwl:hasExactSynonym` |
| Definition/description | `P97`, `P325` | (`IAO_0000115` where present; not confirmed universal) |

A dataset-agnostic loader would need a small per-ontology predicate mapping (label/synonym/definition → NCIT vs. DOID vs. OBO-standard vocab) since the two ontologies do not share one uniform annotation vocabulary, unlike a case where both sides use identical predicate URIs.

## Ground-Truth Alignment Labels

- `refs_equiv/full.tsv`, `train.tsv`, `test.tsv`: tab-separated, header `SrcEntity  TgtEntity  Score`. Each row is a gold **equivalence** mapping between one NCIT class IRI and one DOID class IRI, `Score` fixed at `1.0`. These are exactly analogous to `ent_links`/fold files in MovieGraphBenchmark — a global mapping file plus pre-defined train/test partitions (1,406 train / 3,280 test rows, i.e. roughly 30/70, not a 721 split).
- `refs_equiv/test.cands.tsv`: header `SrcEntity  TgtEntity  TgtCandidates`. For each test gold mapping, a pre-sampled pool of negative target-class IRIs (textually and structurally confusable candidates) is stored as a Python-literal tuple string, for Hits@K/MRR-style ranking evaluation.
- `refs_subs/train.tsv`, `refs_subs/test.cands.tsv`: same shape as above but for **subsumption** (is-a, not equivalence) gold mappings — a different, harder alignment relation not needed for a straightforward node-alignment setting.
- No file establishes direct graph edges between NCIT and DOID classes; the only cross-ontology information is these gold mapping/candidate files, matching the "no assumed cross-graph edges beyond alignment ground truth" requirement.

## Suitability Evaluation Against the Two-Disconnected-Text-Attributed-Graphs Setting

| Requirement | `ncit-doid` support |
|---|---|
| Two separate graphs/ontologies with meaningful intra-graph relationships | **Yes.** NCIT and DOID are structurally independent OWL files; each has thousands of `rdfs:subClassOf` is-a edges forming a real hierarchy (plus richer logical axioms in NCIT). |
| Textual node attributes (names, definitions, descriptions, synonyms) | **Yes.** `rdfs:label` on both sides; synonyms (`P90` / `hasExactSynonym`); definitions on NCIT (`P97`/`P325`) and at least partially on DOID. |
| No assumed cross-graph edges beyond alignment ground truth | **Yes.** The only cross-ontology artifacts are the `refs_equiv`/`refs_subs` gold mapping files; no relation triples link NCIT classes to DOID classes directly. |
| Feasible size for a short experiment | **Partially.** Full files are large (NCIT 55 MB / ~15.8k classes, DOID 6.4 MB / ~8.5k classes), and NCIT's OWL is non-trivial to parse (nested `equivalentClass`/`intersectionOf`/`Restriction` blocks, custom `P*`/`A*` predicate IDs). The task is explicitly designed to be *prunable* (per `documentation.md`'s "Ontology Pruning" section) to a smaller aligned subset, but that pruning is itself a step not present as a ready-made small file in this repo — the on-disk data is the full-size ontology, not a pre-shrunk sample. |
| Clear node identifiers, edges, text fields, alignment labels | **Yes, with caveats.** Node IDs are full class IRIs (clear and stable). Edges are `rdfs:subClassOf` (clear, though NCIT layers on non-trivial OWL logical axioms). Text fields exist but live under different, ontology-specific predicate names on each side (NCIT's proprietary `P*` codes vs. DOID's OBO-standard `oboInOwl:*`), so extracting them cleanly needs a small predicate-mapping/lookup step per ontology, unlike a single shared vocabulary. |

## Conclusion

`ncit-doid` is a **better conceptual fit** for validating a two-disconnected-text-attributed-graphs, embedding-based cross-graph alignment setting:

- It is explicitly built for **cross-ontology alignment** on two different, non-overlapping controlled vocabularies (a source ontology of clinical/biomedical concepts vs. a target ontology of diseases), which is much closer in spirit to an ATT&CK–CWE pairing (two different classification systems related but not identical in scope).
- Its gold mappings (`refs_equiv/*`) are equivalence alignments between conceptually distinct ontologies, and it additionally ships subsumption-mapping ground truth (`refs_subs/*`), which is a strictly richer evaluation surface than MovieGraphBenchmark provides.
- The intra-graph edges (`rdfs:subClassOf` hierarchies) and text attributes (`rdfs:label`, synonyms, definitions) are both present on both sides, satisfying the two structural requirements MovieGraphBenchmark's `attr_triples_*` files did not (recall: MovieGraphBenchmark's attribute files are not edges at all).

**size and format complexity**: `ncit-doid` requires OWL/XML parsing (e.g. via `rdflib`/`owlready2`/DeepOnto, the two ontologies use different, non-uniform annotation-property vocabularies for text extraction, and the raw files here are full-size (tens of thousands of classes) rather than pre-pruned to a small experiment-ready subset — some ontology pruning or subsetting would likely be needed to keep a short experiment tractable, though the necessary gold-standard files (`refs_equiv/train.tsv`, `test.tsv`, `test.cands.tsv`) are already split and ready to consume as-is regardless of subsetting.

## Update (Sprint 3.0.0): how the adapter reads NCIT's logical axioms

The NCIT logical-definition axioms above are **not** a side note: a class has either direct
`rdfs:subClassOf` parents or an `owl:equivalentClass` definition, never both, so for ~7,864 classes
(~half of NCIT) the `equivalentClass` block is the only place their parents are recorded. Only 9,364
of the raw "16,030 `rdfs:subClassOf`" occurrences are direct parent pointers; the rest are nested in
restrictions.

The adapter (`edge_sources` in `configs/datasets/ncit_doid.yaml`) now turns the **named members of an
`owl:equivalentClass` -> `owl:intersectionOf`** into `subClassOf` edges (entailed direct superclasses).
`owl:Restriction` members are still ignored. Result for NCIT: 23,486 edges, 6 isolated nodes (top-level
categories). DOID is unchanged (9,388 edges). See [ncit_edge_extraction_analysis.md](ncit_edge_extraction_analysis.md).
