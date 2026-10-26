# Sprint 1.3 Execution Log — Adapter layer and `ingest` pipeline

## Scope of this entry
Covers all of Phase 1.3. Part 1 (below) built `adapters/base.py` and the `attack_cwe` adapter package.
Part 2 (bottom of this file) completed the rest: the `ncit_doid` adapter, `pipelines/ingest/pipeline.py`,
the three config YAMLs, and `cli.py` — Phase 1.3 is now fully complete.

## Summary
Built `src/cga/adapters/`: the `DatasetAdapter` ABC + `@register_adapter` decorator (§4.1), and a full
`attack_cwe` adapter (§4.2) that reads the ATT&CK STIX bundle and CWE XML catalog into the four
canonical record types, ported from the legacy stage-1 scripts.

## Implementation

**`adapters/base.py`**:
- `GraphSpec` — frozen dataclass (`key`, `role`, `display_name`).
- `DatasetAdapter` ABC — `validate_sources()` (default no-op), abstract `iter_nodes`/`iter_edges`/
  `iter_texts`, `iter_alignments()` (default empty), `source_fingerprint()` (default `{}`).
- `register_adapter(name)` — sets `cls.name = name` and registers into `cga.core.registry.adapters`,
  reusing the generic `Registry` (duplicate-name raises `ConfigError`, same as every other registry).
- `adapters/__init__.py` imports `cga.adapters.attack_cwe` so registration happens on package import,
  per §4.1 ("adding a dataset = one adapter package + one config file").

**`adapters/attack_cwe/stix_parser.py`** (ports `2_Mitre_attack_to_json.py` +
`4_Attack_graph_create.py`, merged into one pass over the bundle):
- `parse_stix_bundle` indexes every object by its STIX `id` and counts relationship objects whose
  `source_ref`/`target_ref` don't resolve (`n_dangling_relationships`) — reported, not silently dropped.
- `iter_stix_nodes` yields every non-`relationship` object with an id; `external_id` comes from
  `external_references[0].external_id` when present.
- `iter_stix_edges` yields one edge per relationship with both endpoints present; drops relationships
  with a dangling endpoint (matches legacy `4_Attack_graph_create.py`'s `count` skip).
- `iter_stix_texts` takes a `{variant: [field_names]}` map (`description`, `detection` ->
  `x_mitre_detection`) and joins present fields with `". "`.

**`adapters/attack_cwe/cwe_parser.py`** (ports `1_CWE_xml_to_json.py` + `3_CWE_grpah_create.py`):
- `parse_cwe_catalog` walks `Weakness` elements under the CWE-7/XHTML namespaces, flattening each of
  the eight text fields exactly as the legacy script does (join `xhtml:p` children if present, else
  the element's own stripped text, else `"N/A"`).
- `iter_cwe_edges` emits an edge per `Related_Weaknesses` entry whose `Nature` is in the configured
  `edge_relations` list (default `[ChildOf]`), skipping entries whose `CWE_ID` isn't a known weakness
  (dangling endpoint) — same policy as the STIX side.
- `iter_cwe_texts` takes the same `{variant: [field_names]}` shape as the STIX side, using
  `getattr(info, field)` against `WeaknessInfo`, skipping `"N/A"`/empty fields.

**`adapters/attack_cwe/adapter.py`**:
- `AttackCweAdapterConfig` (pydantic, `extra="forbid"`) mirrors `configs/datasets/attack_cwe.yaml` §7.3:
  `attack: {file, edge_from_relationships, include_types}`, `cwe: {file, version, edge_relations}`.
- `AttackCweAdapter(DatasetAdapter)` — `graphs = (GraphSpec("attack", "graph_1", ...), GraphSpec("cwe",
  "graph_2", ...))`; `validate_sources()` raises `MissingInputError` with the actual missing path;
  `iter_nodes`/`iter_edges`/`iter_texts` chain the two parsers' outputs; `iter_alignments()` returns
  empty (no ground truth for this dataset, per §4.2); `source_fingerprint()` hashes both raw files via
  `core.io.sha256_file`.
- Constructor takes `root` (dataset's raw dir) and `text_variants` (the dataset YAML's `text_variants`
  block) alongside the validated config, so the adapter has no path/text-composition knowledge baked in.

## Tests
`tests/unit/adapters/` (22 new tests), fixtures committed under `tests/unit/adapters/fixtures/`
(`stix_sample.json` — 3 nodes incl. one dangling-relationship target and one empty-description node;
`cwe_sample.xml` — 3 weaknesses incl. one `ChildOf` chain and one dangling `PeerOf` reference):
- `test_stix_parser.py` (4) — dangling-relationship count, node/edge extraction incl. `has_text`
  and `external_id`, text-variant assembly, empty-text node yields `""`.
- `test_cwe_parser.py` (6) — version + weakness-id extraction, `xhtml:p`-paragraph joining vs. plain
  text, node extraction incl. `has_text`, `ChildOf`-only edge filtering, dangling-endpoint drop,
  text-variant assembly skipping `"N/A"`.
- `test_base.py` (3) — `@register_adapter` populates the registry and sets `.name`, duplicate-name
  registration raises `ConfigError`, default `iter_alignments`/`source_fingerprint`/`validate_sources`.
- `test_attack_cwe_adapter.py` (7) — `validate_sources` pass/raise, combined node iteration assigns the
  correct `graph_role` per source, edges never cross `graph_role`, empty alignments, fingerprint covers
  both files, and **cross-graph edge / duplicate node_id are rejected end-to-end through
  `graph_store.write_nodes`/`write_edges`** (per the Phase 1.3 checklist's explicit ask), not just at
  the parser level.

Combined with Phases 1.0–1.2: `uv run pytest tests/unit -q` → **79 passed**.

**`tests/regression/`** (new top-level dir, separate from `tests/unit`) — added after the raw
`ics-attack.json` in `data/raw/attack_cwe/` was swapped to byte-identical with
`legacy/datasets/ics_attack/attack.json` (confirmed via sha256), making a direct comparison against
the legacy stage-1 graph JSONs possible. `test_attack_cwe_legacy_graph.py` compares
`iter_stix_edges`/`iter_cwe_edges` output against `legacy/datasets/{ics_attack/attack_graph.json,
cwe/cwec_v4.12_graph.json}` `links`, as unordered `(source_id, target_id)` sets (the legacy CWE graph
has ~116 duplicate links from the original script, so set comparison — not list equality — is the
correct check). Both match exactly: 781/781 ATT&CK links, 1,141/1,141 unique CWE `ChildOf` links.
Deliberately kept out of `tests/unit` (per user request) since it depends on `legacy/` and the full raw
source files rather than committed fixtures, and is not meant to run on every `pytest tests/unit`
invocation; each test `skipif`s when those paths are absent. Run explicitly:
`uv run pytest tests/regression -q`.

## Files Created
- `src/cga/adapters/__init__.py`, `src/cga/adapters/base.py`
- `src/cga/adapters/attack_cwe/__init__.py`, `{stix_parser,cwe_parser,adapter}.py`
- `tests/unit/adapters/__init__.py`, `test_{base,stix_parser,cwe_parser,attack_cwe_adapter}.py`
- `tests/unit/adapters/fixtures/{stix_sample.json,cwe_sample.xml}`
- `tests/regression/__init__.py`, `tests/regression/test_attack_cwe_legacy_graph.py`

## Files Modified
- `dev_docs/sprints/sprint_1.md` (first two Phase 1.3 items marked complete)

## Notes / assumptions
- **Update:** the raw ATT&CK file under `data/raw/attack_cwe/` was originally `ics-attack-15.0.json`
  (ICS ATT&CK v15.0), which produced 1,273 nodes/2,846 edges — not the spec's reference 1,136/2,078.
  It has since been replaced with `ics-attack.json`, byte-identical (sha256-verified) to
  `legacy/datasets/ics_attack/attack.json`, the exact bundle the reference counts were computed from.
  The `tests/regression/` suite added below confirms the new parsers reproduce the legacy edge lists
  exactly against this file, so the discrepancy is resolved — no further reconciliation needed.
- `edge_from_relationships` and `include_types` are accepted in `AttackConfig` (matching §7.3's YAML
  shape) but not yet read by `stix_parser` — every object with an id becomes a node and every
  relationship becomes an edge, matching the legacy behavior (`include_types: null` in the spec's
  default config means "every object with an id"). Wiring `include_types` filtering is left for when
  the `ingest` pipeline config is built, since no test in this phase's checklist exercises it.
- `source_version` is left as `""` for ATT&CK nodes (the STIX bundle's `x_mitre_attack_spec_version`/
  collection version isn't a per-object field); CWE nodes get the catalog's `Version` attribute
  (`"4.12"`). Revisit if `ingest`'s `stats.json` needs a non-empty ATT&CK version string.

## CLI to verify (Part 1)
```
uv run pytest tests/unit/adapters -q
uv run pytest tests/unit -q
uv run pytest tests/regression -q
```

---

# Part 2 — `ncit_doid` adapter, `ingest` pipeline, configs, `cli.py`

## Summary
Completed Phase 1.3: the `ncit_doid` adapter (shared streaming OWL parser + refs TSV parser), the
`ingest` pipeline's six steps (§5.1), the three config YAMLs (§7.3–7.4), and the full `cga` CLI (§8.1).
Both datasets now run end-to-end: `cga run ingest --dataset attack_cwe` and
`cga run ingest --dataset ncit_doid` both succeed, validate, and inspect cleanly.

## Implementation

**`adapters/ncit_doid/owl_parser.py`** — one streaming parser (`xml.etree.ElementTree.iterparse`,
never holds the full 55MB NCIT file as one DOM tree) shared by both ontologies, per §4.3's explicit
design: NCIT and DOID have the *same structural shape* (`owl:Class rdf:about=IRI` +
`rdfs:subClassOf rdf:resource=parentIRI`) and differ only in which annotation-property vocabulary
carries text (NCIT's ontology-local `P90`/`P97`/`P325` vs. DOID's standard `oboInOwl:hasExactSynonym`
and `obo:IAO_0000115`) — so one parser takes a **predicate map** (`{role: [predicate names]}`) per
call rather than being split per ontology.
- `parse_owl_ontology` walks every `owl:Class`, reading only `rdf:about` and direct
  `rdfs:subClassOf/@rdf:resource` children — `owl:equivalentClass`/`intersectionOf`/`Restriction`
  logical-axiom blocks are structurally skipped (never matched as a `subClassOf` child), per §4.3 and
  the §12 backlog. Verified against real data: 15,762 NCIT classes / 9,364 `subClassOf` edges, 8,465
  DOID classes / 9,388 edges — both counts match a raw `grep -c "rdfs:subClassOf rdf:resource"`
  exactly, and match `dev_docs/oaei_bioml_ncit_doid_dataset_structure.md`'s reported class counts.
- `_resolve_predicate_tag` resolves a `prefix:localname` predicate (`rdfs:label`,
  `oboInOwl:hasExactSynonym`, `obo:IAO_0000115`) against a small fixed-prefix table, and a bare name
  (`P90`, `P97`) against the ontology's own declared default namespace — matching how NCIT declares
  its custom annotation properties in its own `Thesaurus.owl#` vocabulary. (`obo:` was added to the
  fixed-prefix table after discovering DOID's real `IAO_0000115` predicate lives in the shared
  `http://purl.obolibrary.org/obo/` namespace, not DOID's own `doid.owl#` — target_architecture.md
  §7.4's example config writes it bare; `configs/datasets/ncit_doid.yaml` here uses the explicit
  `obo:IAO_0000115` form instead, confirmed correct against the real file: 138 DOID classes carry a
  definition.)
- `iri_fragment(iri)` — fragment after `#`, else the final `/` segment (id_style "fragment" for both
  sides per §7.4, since DOID IRIs have no `#`).
- `iter_owl_nodes`/`iter_owl_edges`/`iter_owl_texts` mirror the `attack_cwe` parsers' shape; edges to
  an IRI outside the ontology's own class set are dropped (dangling-parent policy, same as the STIX/
  CWE side); `iter_owl_texts` takes the same `{variant: [role, ...]}` shape as the other adapter, over
  generic roles (`label`, `synonym`, `definition`) rather than raw predicate names.

**`adapters/ncit_doid/refs_parser.py`**:
- `parse_candidate_pools` parses `test.cands.tsv`'s `TgtCandidates` column with `ast.literal_eval`
  **not** `eval` (explicit requirement in the Phase 1.3 checklist), and raises `ValueError` if the
  literal isn't a tuple/list of strings — closes the arbitrary-code-execution surface a raw `eval`
  on file content would have.
- `iter_refs_alignments` reads `train.tsv`/`test.tsv`, assigns `split` from which file a row came
  from, attaches the matching candidate pool (by `(SrcEntity, TgtEntity)` IRI pair) only to its own
  test row. Verified against real data: 1,406 train + 3,280 test = 4,686 total, matching
  `dev_docs/oaei_bioml_ncit_doid_dataset_structure.md` exactly.

**`adapters/ncit_doid/adapter.py`** — `NcitDoidAdapterConfig` (`ncit`/`doid`/`pruning` blocks, pydantic
`extra="forbid"`) mirrors §7.4; `NcitDoidAdapter` chains the OWL parser (once per ontology, each with
its own predicate map) and the refs parser; `iter_alignments()` returns real data (unlike
`attack_cwe`), reading the dataset YAML's `alignments.equivalence` block.

**`pipelines/ingest/pipeline.py`** — `IngestPipeline` (`@pipelines.register("ingest")`), six steps per
§5.1:
1. `ResolveAdapterStep` — looks up `dataset.adapter` in the adapter registry, builds it via
   `inspect.signature` (so datasets without an `alignments` kwarg, like `attack_cwe`, don't need a
   dummy parameter), calls `validate_sources()`, records every `source_fingerprint()` path as a
   manifest input via `ctx.manifest.record_input` (hash computed fresh from disk, not trusted from the
   adapter).
2. `IngestNodesStep` — drains `iter_nodes()`, writes via `graph_store.write_nodes` (pk-uniqueness
   enforced there).
3. `IngestEdgesStep` — drains `iter_edges()`, drops any edge whose endpoint isn't in the just-written
   node set (dangling, counted into `stats.json`), **dedupes by `edge_id`** before writing (the CWE
   XML source itself has ~157 duplicate `ChildOf` declarations — same duplication the
   `tests/regression/` suite found in the legacy graph JSON), then `graph_store.write_edges`
   (cross-`graph_role` check enforced there).
4. `IngestTextsStep` — drains `iter_texts()`, `graph_store.write_texts` (composite pk + FK enforced
   there).
5. `IngestAlignmentsStep` — drains `iter_alignments()`, `graph_store.write_alignments` (empty-table-OK
   + opposite-role FK enforced there).
6. `ValidateAndManifestStep` — writes `stats.json` (per-`graph_role` node/edge counts, dropped-edge
   count, text coverage, mean/max degree, per-split alignment counts), records all five outputs into
   the manifest.
`IngestPipeline.run()` calls the base class's `run()` (writes `manifest.json`, marks success/failed)
then, only on success, updates the dataset's `LATEST` file — a failed run must never become `latest`.

**`configs/pipelines/ingest.yaml`, `configs/datasets/{attack_cwe,ncit_doid}.yaml`** — follow §7.3–7.4
exactly, with two adjustments for what's actually on disk: `attack.file: ics-attack.json` (not the
spec's `attack.json`, matching the committed raw file — see Part 1's notes on this), and
`doid.predicate_map.definition: ["obo:IAO_0000115"]` (explicit prefix, see above).

**`cli.py`** — `argparse`-based, per §8.1:
- `cga list [pipelines|adapters|embedders|samplers]` — prints each registry's `.names()`.
- `cga run <pipeline> --dataset <id> [--experiment] [--config ...] [--set ...] [--run-name]
  [--resume] [--dry-run]` — layers config via `core.config.build_config`, resolves `--set
  key.path=value` through the existing `parse_set_value`, builds the pipeline from the registry, runs
  it under a fresh `RunContext`. `--dry-run` resolves config and prints the plan without creating a
  run directory. Catches `CgaError` and exits 1 without a traceback.
- `cga validate <artifact_dir>` — checks `manifest.json` exists and `status == "success"`, then
  re-reads every present canonical table through `graph_store`'s schema-validating readers (this is
  the read-side check §8.3 describes as `cga validate` running standalone).
- `cga inspect <artifact_dir>` — prints the manifest and recursively walks `inputs[].run_id` links to
  upstream manifests, stopping at raw-file inputs (`run_id` empty) — the traceability chain §3.6
  describes.
- `cga config show <pipeline> [...layering flags]` — prints the fully-resolved config as JSON.
- `_PIPELINE_OUTPUT_ROOT` maps a pipeline name to its `paths.*` output subdirectory key (`ingest` ->
  `processed`); later pipelines (`embed`, `training_data`, `gcl_train`) will extend this dict rather
  than needing new CLI logic.

## Tests
`tests/unit/adapters/` (+22 tests: `test_owl_parser.py` 8, `test_refs_parser.py` 5,
`test_ncit_doid_adapter.py` 9), fixtures under `tests/unit/adapters/fixtures/`
(`ncit_sample.owl`/`doid_sample.owl` — 3 classes each incl. one `owl:equivalentClass` logical-axiom
block to prove it's ignored, and one dangling `subClassOf` parent; three small `refs_equiv_*.tsv`
files including a candidate-pool row):
- `test_owl_parser.py` — IRI fragment extraction (both `#` and `/`-terminated forms),
  `owl:equivalentClass`/`intersectionOf` correctly ignored as a `subClassOf` source, node/edge
  extraction, dangling-parent drop, NCIT vs. DOID predicate maps both resolve correctly against the
  *same* parser code.
- `test_refs_parser.py` — `ast.literal_eval`-based candidate parsing, rejection of a non-string-tuple
  literal, split assignment, candidate-pool attached only to its matching test row, empty when no
  paths given.
- `test_ncit_doid_adapter.py` — same shape as `test_attack_cwe_adapter.py`: validate_sources
  pass/raise, combined-graph node/edge iteration, alignments populated with train/test splits, empty
  without an alignments config, fingerprint covers ontologies + alignment files, and
  cross-graph-edge/duplicate-node_id rejected end-to-end through `graph_store`.

`tests/unit/pipelines/ingest/test_pipeline.py` (5 tests, against the small `attack_cwe` fixtures via a
hand-built config dict, not the full raw dataset): writes all four tables + `stats.json`; writes
`LATEST` pointing at the just-completed run; edge-id dedup collapses a source XML's duplicate
`Related_Weaknesses` declarations; **a monkeypatched adapter that yields a cross-graph edge causes the
whole run to raise** (`ArtifactValidationError` propagates out of `Pipeline.run`, matching the
checklist's "cross-graph edge raises" ask at the pipeline level, not just inside `graph_store`
directly); every text row's `node_id` resolves in the written `nodes.parquet`.

`tests/unit/cli/test_cli.py` (8 tests, against a temporary `configs/`+`data/raw/` tree built per test):
`list` shows registered pipelines/adapters; `--dry-run` resolves config and writes nothing; an unknown
`--dataset` exits 1 with a clear stderr message; a real `run` writes `manifest.json` + `LATEST`;
`validate` returns 0 on a successful run; `inspect` prints the manifest chain; `config show` prints
resolved JSON; the parser requires a subcommand.

Combined with Phases 1.0–1.2 and Part 1: `uv run pytest tests/unit -q` → **115 passed**.
`uv run pytest tests/regression -q` → **2 passed** (unchanged from Part 1).

## End-to-end verification against real data
```
cga run ingest --dataset attack_cwe   → 1,217 nodes / 1,922 edges / 2,434 texts / 0 alignments
cga run ingest --dataset ncit_doid    → 24,227 nodes / 18,752 edges / 72,681 texts /
                                         4,686 alignments (1,406 train / 3,280 test)
cga validate <each run's dir>         → OK (4 tables validated, manifest status=success)
cga inspect  <each run's dir>         → prints pipeline/run/status plus each raw-file input's hash
```
`attack_cwe`'s node/edge counts (1,217/1,922) do **not** match the Phase 1.3 checklist's original
"1,136 nodes / 2,078 edges" reference — see Notes below; the checklist text has been corrected rather
than the pipeline changed, since the new count is the architecturally-intended one.

## Files Created
- `src/cga/adapters/ncit_doid/__init__.py`, `{owl_parser,refs_parser,adapter}.py`
- `src/cga/pipelines/__init__.py`, `src/cga/pipelines/ingest/{__init__.py,pipeline.py}`
- `src/cga/cli.py`
- `configs/base.yaml` companions: `configs/pipelines/ingest.yaml`,
  `configs/datasets/{attack_cwe,ncit_doid}.yaml`
- `tests/unit/adapters/test_{owl_parser,refs_parser,ncit_doid_adapter}.py`
- `tests/unit/adapters/fixtures/{ncit_sample.owl,doid_sample.owl,refs_equiv_train.tsv,
  refs_equiv_test.tsv,refs_equiv_test_cands.tsv}`
- `tests/unit/pipelines/__init__.py`, `tests/unit/pipelines/ingest/{__init__.py,test_pipeline.py}`
- `tests/unit/cli/{__init__.py,test_cli.py}`

## Files Modified
- `src/cga/adapters/__init__.py` (also imports `ncit_doid` now)
- `dev_docs/sprints/sprint_1.md` (all remaining Phase 1.3 items marked complete; Verify line updated
  with the real node/edge/alignment counts and a note on why they differ from the legacy reference)

## Notes / assumptions
- **`attack_cwe` node/edge count vs. the original checklist reference (1,136/2,078):** that number was
  the *legacy* pipeline's count, which only kept nodes reachable by at least one edge (933 CWE + 203
  ATT&CK node-set-filtered nodes). target_architecture.md §5.1 explicitly removes that filter
  ("Node-set policy: **all** parsed entities are written... exclusions are recorded with a reason
  instead of silently shifting indices") and defers it to the embedding/status layer. The new
  pipeline's 1,217/1,922 is fully explained: 781 STIX edges (no dedup needed, STIX ids are already
  unique) + 1,141 *unique* `(src, dst)` CWE `ChildOf` pairs (the source XML declares 1,298 raw
  `Related_Weaknesses` entries but 157 are exact duplicates — the same duplication
  `tests/regression/` found in the legacy graph JSON) = 1,922. Confirmed correct, not a bug — see the
  AskUserQuestion exchange this session that walked through the arithmetic before updating the
  checklist.
- `IngestEdgesStep`'s edge_id-based dedup is pipeline-level, not adapter-level — an adapter's
  `iter_edges()` may legitimately yield the same logical edge twice if its source data does (as CWE's
  XML does); deduping at ingest-write time keeps the parsers a faithful, dedup-free translation of
  their source format, and keeps the "no duplicate edge_id" invariant enforced in exactly one place
  (`graph_store.write_edges`'s pk-uniqueness check, which would otherwise raise on that duplication).
- `cli.py`'s `_resolve_config` hardcodes the `configs/` layout (`configs/datasets/{id}.yaml`,
  `configs/pipelines/{name}.yaml`, `configs/experiments/{name}.yaml`) as relative paths from the
  current working directory — matches how `cga` is meant to be invoked (from the repo root) and how
  every other `configs/*.yaml` in this repo is referenced; no CLI flag to relocate `configs/` itself
  was in the Phase 1.3 scope.
- `--resume RUN_ID` is accepted by the CLI parser but not yet interpreted by `IngestPipeline` (ingest
  has no resumable per-node state, unlike the future `embed` pipeline's `--resume`, which is where
  §5.2 actually specifies resume semantics). Left as a no-op flag for now rather than raising, so the
  same `cga run` invocation shape works across pipelines once `embed` implements it.
- `n_nodes_with_text` in `stats.json` is `has_text` boolean coverage (set by each adapter's node
  parser), not zero-text-string detection — matches how `NodeRecord.has_text` is already defined in
  `data/schemas.py` from Phase 1.2.

## CLI to verify (Part 2)
```
uv run pytest tests/unit -q
uv run pytest tests/regression -q
uv run python -m cga.cli run ingest --dataset attack_cwe
uv run python -m cga.cli run ingest --dataset ncit_doid
uv run python -m cga.cli validate data/processed/attack_cwe/<run_id>
uv run python -m cga.cli inspect  data/processed/attack_cwe/<run_id>
```

## Next Phase
Phase 1.4: `embed` pipeline — `pipelines/embed/text_embedders.py` (`hf_mean_pool`),
`pipelines/embed/graph_embedders.py` (`deepwalk`/`node2vec`), `pipelines/embed/pipeline.py`
(`generate`→`materialize`→`validate`, `--resume`), `configs/pipelines/embed.yaml`.
