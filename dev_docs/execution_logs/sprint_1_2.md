# Sprint 1.2 Execution Log — Canonical data layer

## Summary
Built `src/cga/data/`: node identity, pyarrow/pydantic schemas for every canonical table (§6.1–6.6), FK/pk-enforcing stores for the four ingest tables, `AlignedMatrix`/`NodeUniverse` for dense arrays, a retryable embedding store, and an example store validated against a universe.

## Implementation

**`data/ids.py`** — `make_node_id`/`parse_node_id` splitting on the *first* colon only (source ids like STIX ids and IRIs routinely contain colons), rejecting empty parts or a colon in `graph_key`.

**`data/schemas.py`** — pyarrow `pa.schema` + pydantic `BaseModel` (`extra="forbid"`) pairs for `nodes`, `edges`, `texts`, `alignments`, `examples`, `status`, `index`, `shards`, `universe`, matching field types/nullability from §6.1–6.6 exactly (e.g. `texts.text` non-nullable but empty-allowed, `alignments.candidates` nullable list).

**`data/graph_store.py`** — write/read for the four ingest tables:
- `write_nodes`: schema + `node_id` pk uniqueness.
- `write_edges`: schema + `edge_id` pk uniqueness + FK (`src_id`/`dst_id` ⊂ nodes) + same-`graph_role` check on both endpoints and the edge's own `graph_role` column.
- `write_texts`: schema + composite pk (`node_id`, `text_variant`) uniqueness + FK ⊂ nodes.
- `write_alignments`: schema + `alignment_id` pk uniqueness + FK ⊂ nodes with `node_1_id` required `graph_1` and `node_2_id` required `graph_2`; an empty table (no-ground-truth datasets) is always valid and skips the row-level checks.

**`data/aligned.py`**:
- `AlignedMatrix` — `load()` validates row-count-vs-index match, `row` values are exactly `0..n-1`, and (when a `manifest.json` sidecar is present) sha256 hashes of `vectors.npy`/`index.parquet`; `row_of`, `select` (reorder/subset, raises listing missing ids), `save` (writes `vectors.npy` + `index.parquet` + a hash manifest).
- `NodeUniverse` — ordered `node_id` + `graph_role` lists, `from_parquet` (row-ordered), `save`; rejects duplicate ids or mismatched list lengths in `__post_init__`.

**`data/embedding_store.py`** — `EmbeddingStore(directory)`:
- `upsert_status` replaces any existing row for a `node_id` (never appends a second row) and increments `attempts`.
- `append_shard` writes one more `shards/part-NNNNN.parquet` file per call — append-only, crash-safe.
- `pending_nodes(all_ids)` returns every id whose current status isn't `ok` (including never-attempted ids).
- `materialize()` reads all shards, keeps only `ok`-status ids in their status-table order, and — when a node has multiple shard rows (e.g. failed-then-retried-ok) — the *last* shard file wins, so a stale failed vector never survives into the final matrix; writes through `AlignedMatrix.save`.

**`data/example_store.py`** — `write_examples` checks all five id columns (`anchor_id`, `text_pos_id`, `text_neg_id`, `graph_pos_id`, `graph_neg_id`) against `NodeUniverse.node_ids` before writing, plus `example_id` pk uniqueness after write.

## Bug fixed along the way
`core/io.py::write_parquet` only caught `pa.ArrowInvalid`/`pa.ArrowTypeError` around `table.cast(schema)`, but pyarrow raises a bare `ValueError` when the table's field *names* don't match the target schema (as opposed to a type-only mismatch). Added `ValueError` to the caught exception tuple so a genuinely malformed table still surfaces as `ArtifactValidationError` instead of leaking a pyarrow-specific exception type — caught by `test_write_nodes_schema_violation_raises`.

## Tests
`tests/unit/data/` (35 new tests):
- `test_ids.py` (6) — roundtrip, colon-containing source_id roundtrip, rejection of bad graph_key/empty parts/missing separator.
- `test_graph_store.py` (12) — nodes/edges/texts/alignments roundtrip, pk-duplicate rejection, schema-violation rejection, edge FK/cross-role rejection, texts FK rejection, alignments wrong-role-order rejection, empty-alignments-table validity.
- `test_aligned.py` (10) — AlignedMatrix save/load roundtrip, `row_of` hit/miss, `select` reorder and missing-id rejection, duplicate-id and row-count-mismatch rejection, NodeUniverse save/load roundtrip and validation.
- `test_embedding_store.py` (5) — all-pending with no status, `ok` rows excluded from resume, `failed`→`ok` re-run increments attempts, materialize excludes non-`ok` rows, materialize keeps index aligned with vectors.
- `test_example_store.py` (3) — write success, outside-universe rejection, duplicate `example_id` rejection.

Combined with Phase 1.1's suite: `uv run pytest tests/unit -q` → **58 passed**.

## Files Created
- `src/cga/data/__init__.py`
- `src/cga/data/{ids,schemas,graph_store,aligned,embedding_store,example_store}.py`
- `tests/unit/data/{test_ids,test_graph_store,test_aligned,test_embedding_store,test_example_store}.py`

## Files Modified
- `src/cga/core/io.py` (bug fix above)
- `dev_docs/sprints/sprint_1.md` (Phase 1.2 marked complete)

## Notes / assumptions
- `EmbeddingStore.upsert_status` and `append_shard` do full-table read/rewrite per call rather than a true append-in-place update for status (Parquet has no in-place row update); this is fine at the per-node granularity described in the spec but would need batching for very large node counts — left as-is since nothing in Phase 1.2's scope calls for it yet.
- Graph embedders' "isolated nodes get `status = skipped`" (§3.4) is a Phase 1.4 (`embed` pipeline) concern — `embedding_store.py` supports any status string, so no change needed here.
- `AlignedMatrix.save` always casts vectors to `float32` before writing, matching the spec's "vectors stay `.npy` float32" rule.

## CLI to verify
```
uv run pytest tests/unit -q
```

## Next Phase
Phase 1.3: Adapter layer and `ingest` pipeline — `adapters/base.py`, `attack_cwe`/`ncit_doid` adapters, `pipelines/ingest/pipeline.py`, dataset configs, and `cli.py`.
