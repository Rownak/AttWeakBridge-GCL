# NCIT edge extraction: why half of NCIT has no edges, and how to add them

Analysis only; no code was changed. Numbers below were computed directly from
`data/raw/ncit_doid/ncit.owl` and `doid.owl` (full parse of every `owl:Class`, scripts in the
session scratchpad, not committed). Related: [oaei_bioml_ncit_doid_dataset_structure.md](oaei_bioml_ncit_doid_dataset_structure.md),
[sprints/sprint_3.md](sprints/sprint_3.md).

## 1. Finding

The observation is correct. [owl_parser.py](../src/cga/adapters/ncit_doid/owl_parser.py) keeps only
`rdfs:subClassOf rdf:resource="…"` children, so any NCIT class whose hierarchy is written as a logical
definition gets no edge.

| NCIT, 15,762 classes | Count |
|---|---|
| Classes with ≥1 direct `rdfs:subClassOf rdf:resource` | 7,880 (50.0%) |
| Classes with **none** (so no edge, no DeepWalk walk, isolated GCN node) | **7,882 (50.0%)** |
| Direct `subClassOf` edges the adapter produces | 9,364 |
| Connected components of the resulting graph | 7,873 (largest: 4,726 nodes) |

DOID has no such problem: 8,464 of 8,465 classes have a direct parent (the one without is the root),
the graph is a single connected component, and the file has no `equivalentClass` or restriction
axioms at all.

The two ways of stating the hierarchy are **mutually exclusive per class**: every NCIT class has
either direct `subClassOf` parents or an `equivalentClass` definition, never both (7,880 + 7,864 +
18 roots = 15,762). So the logical definitions are not extra structure on top of the hierarchy. For
the 7,864 classes defined this way, they are the only place their parents are recorded.

(The `16,030 rdfs:subClassOf` count in the dataset doc comes from a raw grep. It also counts
`subClassOf` lines nested inside restrictions. Only 9,364 of them are direct parent pointers.)

## 2. What the three OWL constructs mean

Taking `C100051` from the file as the running example:

```xml
<owl:Class rdf:about="…#C100051">                     <!-- Renal Cell Carcinoma Associated with Neuroblastoma -->
  <owl:equivalentClass>
    <owl:Class>
      <owl:intersectionOf rdf:parseType="Collection">
        <rdf:Description rdf:about="…#C9385"/>        <!-- Renal Cell Carcinoma (a named class) -->
        <owl:Restriction>
          <owl:onProperty    rdf:resource="…#R126"/>  <!-- Disease_Has_Associated_Disease -->
          <owl:someValuesFrom rdf:resource="…#C3270"/><!-- Neuroblastoma -->
        </owl:Restriction>
      </owl:intersectionOf>
    </owl:Class>
  </owl:equivalentClass>
```

| Construct | Meaning | Logic | Edge-like? |
|---|---|---|---|
| `owl:equivalentClass` | The class is **defined as** (necessary *and* sufficient) the anonymous class expression inside it | `C ≡ X` | Not an edge itself. It is a wrapper that says "C equals this expression". |
| `owl:intersectionOf` | The expression is the AND of its members. A thing is in it only if it is in **every** member | `X = A ⊓ B ⊓ …` | Not an edge, but it **entails** `C ⊑ A` for each member. |
| `owl:Restriction` + `onProperty` + `someValuesFrom` | An anonymous class: "things related by property R to at least one instance of filler F" | `∃R.F` | Entails `C ⊑ ∃R.F`: a typed, non-hierarchical relation `C —R→ F`. |

Putting these together, `C100051 ≡ RenalCellCarcinoma ⊓ ∃Disease_Has_Associated_Disease.Neuroblastoma`,
which entails two different kinds of statement:

1. `C100051 ⊑ C9385`: **a true is-a edge**. `C100051` is a kind of Renal Cell Carcinoma. This is
   the same information a plain `rdfs:subClassOf C9385` line would carry.
2. `C100051 ⊑ ∃R126.C3270`: a **semantic relation**, not is-a. It says the disease is associated
   with Neuroblastoma. Neuroblastoma is not a parent of it.

This distinction matters because the two should not be given the same treatment as edges.

## 3. Can these be used as direct edges?

**Named members of the intersection: yes, as `subClassOf` edges.** An intersection member that is a
named class is, by the semantics, an entailed direct superclass, so it is equivalent to a
`subClassOf` line. Adding them is information-preserving, not a heuristic. Facts from the file:

- All 7,864 equivalence-defined classes have ≥1 named member (14,122 edges in total; 2,628 classes
  have 1, 4,338 have 2, 888 have 3–6). Multiple members mean multiple parents, as with ordinary
  multiple inheritance.
- The combined hierarchy (direct + intersection parents) is a **DAG** with **18 roots**. The 18
  classes with no edge at all after this step are NCIT's top-level categories (`Organism`, `Gene`,
  `Disease, Disorder or Finding`, …), which is the expected result.
- No named member points at a class missing from the file.

**Restriction fillers: usable, but as a separate edge type, and only deliberately.**
`someValuesFrom` is not is-a, and the properties are heterogeneous. In the equivalence axioms
the most common ones are `Disease_May_Have_Finding` (R115, 2,577 uses), `Disease_Has_Finding`
(R108, 2,555), `Disease_Has_Primary_Anatomic_Site` (R101, 1,245), `Disease_May_Have_Cytogenetic_Abnormality`
(R114, 1,086), and `Disease_Has_Abnormal_Cell` (R105, 966). Some are weak ("May_Have") or negated
by name (`Disease_Excludes_Finding`, R142, 649 uses; an edge to a finding the disease *excludes*
would be misleading). Fillers also point to very different kinds of entity, namely findings,
anatomic sites, genes, and cell types, whereas the hierarchy edges stay inside "disease"
concepts. 4,995 classes carry 14,222 such restrictions, in 846 cases two or more per class.

A further 1,835 classes have `subClassOf` **restrictions** (3,345 edges; e.g.
`Gene_Plays_Role_In_Process`, `Anatomic_Structure_Is_Physical_Part_Of`). These are the same kind of
typed relation as above, in the `subClassOf` position. All 1,835 of these classes already have direct
parents, so they do not matter for the isolation problem.

## 4. Effect on the graph (measured)

Undirected connectivity of the NCIT graph under each choice:

| Edge set | Edges | Isolated nodes | Components | Largest component |
|---|---|---|---|---|
| A. Direct `subClassOf` (current) | 9,364 | **7,841 (49.7%)** | 7,873 | 4,726 |
| B. A + intersection named members | 23,486 | 6 (0.0%) | 18 | 12,597 |
| C. B + equivalence restriction fillers | 37,691 | 6 | 8 | 15,735 |
| D. C + `subClassOf` restrictions | 40,532 | 6 | 7 | 15,756 |

Option B alone repairs the problem: isolated nodes go from 7,841 to 6. Options C and D add 14k
and 3k typed edges, which mostly join already-connected components (largest component 12,597 →
15,735) rather than rescuing nodes.

Relevance to the planned use: roughly half of the NCIT endpoints in the gold alignments are among the
currently isolated classes (734 of 1,406 train and 1,680 of 3,280 test pairs, 52% and 51%), so
leaving them unconnected would hurt both training and any later gold-set evaluation.

## 5. Recommendation

**Do B now, make C opt-in, defer D.**

1. **Add intersection named members as `subClassOf` edges** for NCIT. It is semantically exact,
   fixes the isolation, and keeps the graph a hierarchy (DAG, 18 roots), matching DOID's structure,
   which makes the two graphs comparable. Tag the edge `relation = "subClassOf"` as today.
2. **Store restriction edges with their own `relation`** (e.g. `restriction:R101`), controlled by an
   adapter config flag, default off (so the first run is B only). Reasons to keep it separate:
   the semantics differ, DOID has no counterpart (adding these only to NCIT makes the two graphs
   asymmetric in structure), and the "excludes"/"may have" properties are questionable
   as similarity signal. If switched on, restrict to an allow-list of properties rather than
   taking all 97 `ObjectProperty`s, and run it as an ablation (B vs. C) using the Sprint 2 metrics.
3. **Parser change (when approved):**
   - `OwlClassRecord` gains `intersection_parent_iris` (and optionally `restrictions: list[(property,
     filler)]`); `parse_owl_ontology` descends into `owl:equivalentClass → owl:Class →
     owl:intersectionOf` and takes children that are `rdf:Description`/`owl:Class` with `rdf:about`,
     ignoring `owl:Restriction` members unless the flag is on.
   - `iter_owl_edges` emits them with `relation="subClassOf"` (deduplicate on the existing `edge_id` hash,
     which already includes the relation), and drops any whose target is not a known class, as it
     does today.
   - `OntologyConfig` gets e.g. `edge_sources: [subClassOf, equivalent_intersection]` (default
     both for NCIT; DOID unaffected) and `restriction_properties: []`.
   - `raw_sha256` should include the new fields so change detection stays correct.
   - `edge_predicates` in `ncit_doid.yaml` is currently `[rdfs:subClassOf]`; the new setting belongs
     next to it.
4. **Check downstream handling before enabling typed restriction edges.** `EdgeRecord` already has a
   `relation` field, but a grep of `graph_embedders.py` and `gcl_train/*.py` shows no code reads it:
   DeepWalk and the GCN/GAT encoders treat every edge in `edges.parquet` alike. So option B needs no
   downstream change, but option C would silently mix restriction edges into walks and message
   passing unless a relation filter is added.
5. **Update the docs and tests.** `owl_parser.py`'s module docstring and
   `oaei_bioml_ncit_doid_dataset_structure.md` currently describe the logical axioms as not
   interpreted / a side note. Unit tests: a fixture class modeled on `C100051` (one named + one
   restriction member) yields exactly one edge with B, and two with C; a class with 3 named members
   yields 3 edges.

## 6. Open points

- One class, `C27691`, has a second `equivalentClass` that is neither an intersection nor a named
  class (it also has an intersection, which is handled). Inspect it before the parser change; the
  plan above ignores the extra axiom.
- The 6 classes still isolated under B were listed: `C12218` Chemotherapy Regimen or Agent
  Combination, `C20181` Conceptual Entity, `C22187`, `C22188`, `C43431` Activity, `C97325`
  Manufactured Object. All are top-level categories with no children in the file, so they are not parse
  misses. They are small enough to leave as isolated nodes (or drop if `universe_policy` removes them).
- Is the sprint-3 data-sanity phase (3.0) the right place for this? It changes the ingest output, so
  it should happen **before** the first `embed` run, and the node/edge counts quoted in
  [sprints/sprint_3.md](sprints/sprint_3.md) (~16,030 edges, 9,364 after this analysis) need correcting.
