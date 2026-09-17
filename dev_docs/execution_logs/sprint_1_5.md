# Sprint 1.5 Execution Log — `training_data` pipeline

## Pre-implementation check: label-leakage architectural gap

Before writing code, reviewed `dev_docs/target_architecture.md` §5.3, §6.5, §7.6 and the legacy
`GCL/data_preprocessing/3_pos_neg_sample.py` this phase ports from. Found a real statistical
problem in the spec as written and stopped to raise it before implementing (per `CLAUDE.md` "When
stuck"):

- §5.3 ran `sample` (step 3) **before** `split` (step 4). `rank_topk`/`percentile` build cosine
  similarity over **every** node in the universe, with no notion of held-out test-alignment nodes
  yet. `split` only assigned a `split` label onto already-generated examples "grouped by anchor
  node" — it never filtered which nodes could act as anchor/positive/negative.
- For `ncit_doid`, which has a real `test` alignment split reserved for `evaluate`'s Hits@k/MRR,
  this meant a `test`-split node could be selected as an unsupervised anchor/positive/negative by
  `rank_topk`/`percentile`, contaminating the very embeddings that evaluation later scores against.
  This directly conflicts with the rest of the architecture's emphasis on FK checks, manifest
  chains, and stable ids preventing exactly this kind of silent contamination.

Raised this via `AskUserQuestion`; user chose "apply the smallest correction, then implement."
Amended `target_architecture.md` (§5.3, §6.6 `universe.parquet` schema, §7.6 config) and
`dev_docs/sprints/sprint_1.md`'s Phase 1.5 checklist before writing any pipeline code:

- `universe.parquet` gains a `held_out: bool` column, computed in `build_universe` (step 1) from
  `alignments.parquet` rows whose `split` is in `training_data.held_out_splits` (default `[test]`).
- `similarity` (step 2) and `rank_topk`/`percentile` sampling (step 3) now operate **only** over
  `held_out == False` rows — held-out nodes are structurally unreachable as example endpoints,
  not filtered out after the fact. They stay in `universe.parquet` (still needed by `gcl_train`/
  `evaluate`), just never become a similarity-matrix row/column for unsupervised sampling.
- `supervised` sampling may draw on non-held-out alignment splits (e.g. `train`) — that's the
  intended supervised signal — but never on `held_out_splits`.
- `validate_and_manifest` (step 5) asserts no `held_out` node id appears in any of
  `examples.{anchor_id, text_pos_id, text_neg_id, graph_pos_id, graph_neg_id}` for non-supervised
  examples, raising `ArtifactValidationError` if it does — this is what makes the exclusion
  enforced rather than merely intended.

`attack_cwe` has no alignments, so `held_out` is `False` for every node and behavior there is
unchanged from the original spec.

## Implementation

**`pipelines/training_data/similarity.py`**:
- `SimilarityMatrix` — a small dataclass pairing a dense cosine matrix with its own `row_ids`/
  `col_ids`, so every downstream consumer indexes by id, never by raw array position.
- `compute_text_similarity` — cross-graph cosine in text-embedding space, `graph_1` rows x
  `graph_2` columns, restricted to non-`held_out` ids present in the text `AlignedMatrix`.
- `compute_graph_similarity` — within-graph cosine in structural-embedding space for one
  `graph_role`, same restriction.
- A node absent from a store (e.g. excluded upstream by the universe's `intersection` policy) is
  silently skipped here rather than raising — `build_universe` is the single place eligibility is
  decided; similarity just respects whatever set it was handed.

**`pipelines/training_data/samplers.py`** — three registered samplers, each `(anchor, text_pos,
text_neg, graph_pos, graph_neg)` tuples in, ready for the pipeline to attach `example_id`/`split`:
- `rank_topk` — top-k/bottom-k by argsort, pairs zipped by rank (legacy `sample_10`). Verified
  against real `attack_cwe` data: **34,080 examples**, matching the legacy script's reported count
  exactly.
- `percentile` — percentile thresholds, one view rank-zipped and the other paired by a uniformly
  random index within its own pool (legacy `sample_11`). Unlike the legacy script's unseeded
  `random`, this uses `random.Random(seed)` throughout, so output is reproducible under a fixed
  seed (a Phase 1.5 test requirement the legacy script never met).
- `supervised` — positives are `alignments.parquet` gold pairs (post-`held_out`-split filtering,
  done by the pipeline before calling the sampler); negatives are drawn uniformly from the
  non-positive pool. No-ops (`[]`) when `alignments` is empty or `None`.
- Both `rank_topk`/`percentile`'s per-direction helpers look up each anchor's row **independently
  by id** in the text view and the graph view rather than assuming matching row order/length —
  caught by a unit test where the graph view had a node (`b:3`, isolated in text space) absent from
  the text view; naive positional zipping raised `IndexError`, fixed by id-keyed lookup.

**`pipelines/training_data/pipeline.py`** — `TrainingDataPipeline`, five steps:
1. `build_universe` — intersects `status == ok` node ids across both embedding stores (policy
   `intersection`/`text_only`), preserves `nodes.parquet`'s node order for determinism, and flags
   `held_out` from `alignments.parquet`.
2. `similarity` — three `SimilarityMatrix`es (`text_g1_g2`, `graph_g1`, `graph_g2`), all
   `held_out`-restricted.
3. `sample` — resolves the configured sampler from the registry; for `supervised`, pre-filters
   `alignments` to exclude `held_out_splits` before the sampler ever sees them.
4. `split` — unsupervised examples get `train`/`val` by shuffling **anchors** (not individual
   examples) with `random.Random(seed)` and taking `val_fraction` of them, per `split.group_by:
   anchor_id`; supervised examples take their split directly from the source alignment row.
5. `validate_and_manifest` — the leakage assertion described above, then `write_examples` (which
   also enforces the universe FK from `cga.data.example_store`), `stats.json`, manifest outputs,
   `LATEST`.

**`configs/pipelines/training_data.yaml`** — per target_architecture.md §7.6, plus the new
`held_out_splits: [test]` key.

**`src/cga/cli.py`** — `_run_root` needed a `training_data`-specific case (writes to
`artifacts/training_data/{dataset_id}/`, one level under `paths.artifacts`, mirroring how `embed`
already special-cases its own nesting) since the generic `{paths.<subdir>}/{dataset_id}` lookup has
no `paths.training_data` key.

**`data/schemas.py`, `data/aligned.py`** — `UNIVERSE_SCHEMA`/`UniverseRecord` gained `held_out:
bool`; `NodeUniverse` gained a `held_out: list[bool] | None` field (defaults to all-`False` when
omitted, so every existing keyword-based call site in `tests/unit/data/test_aligned.py` and
`test_example_store.py` kept working unmodified).

## Files modified or created
- `dev_docs/target_architecture.md` — §5.3, §6.6, §7.6 amended for the leakage fix.
- `dev_docs/sprints/sprint_1.md` — Phase 1.5 checklist updated to match.
- `src/cga/data/schemas.py`, `src/cga/data/aligned.py` — `held_out` column/field.
- `src/cga/pipelines/training_data/__init__.py`, `similarity.py`, `samplers.py`, `pipeline.py` — new.
- `src/cga/pipelines/__init__.py` — registers `training_data`.
- `src/cga/cli.py` — `_run_root` case for `training_data`.
- `configs/pipelines/training_data.yaml` — new.
- `tests/unit/pipelines/training_data/test_similarity.py`, `test_samplers.py`, `test_pipeline.py` — new
  (`test_samplers.py` later gained the two self-pairing regression tests described below).

## Verification
- `uv run pytest tests/unit -q` → **146 passed** (0 failures, no regressions; +2 from the
  self-pairing regression tests added with the diagonal-exclusion fix above).
- `cga run training_data --dataset attack_cwe` against the Phase 1.3/1.4 `ingest`/`embed` runs
  already on disk → succeeded, both before and after the diagonal-exclusion fix. `stats.json`
  (unchanged by the fix — excluding the diagonal shifts which node fills the k-th rank slot, not
  how many slots exist):
  ```json
  {"n_examples": 34080, "n_examples_by_split": {"train": 30690, "val": 3390},
   "n_held_out_nodes": 0, "n_universe_nodes": 1136, "sampler": "rank_topk"}
  ```
  `universe.parquet` has exactly 1,136 rows; `examples.parquet` has 34,080 rows, all five id columns
  resolving inside the universe (enforced by `write_examples`'s FK check). `held_out_nodes == 0` is
  correct — `attack_cwe` has no alignments. Post-fix, directly checked `graph_pos_id`/`graph_neg_id`
  against `anchor_id` across all 34,080 rows: **0 self-pairs**.
- `ncit_doid` has no `embed` runs on disk yet (large dataset, intentionally out of the integration
  test suite per target_architecture.md §2 "Testing data"), so the `held_out`/leakage path was only
  exercised against real data indirectly; it is covered directly by
  `tests/unit/pipelines/training_data/test_pipeline.py::test_rank_topk_examples_never_reference_held_out_test_nodes`
  and `test_supervised_split_keeps_test_alignment_out_of_train_examples`, which hand-build an ingest
  run with a `test`-split alignment pair and assert the produced `examples.parquet` never references
  either endpoint.

## Post-implementation fix: self-similarity diagonal leaking into graph pos/neg

User caught a bug in the first cut of `rank_topk`/`percentile`: the within-graph similarity views
(`graph_g1`, `graph_g2`) are square, self-similarity-on-diagonal matrices — an anchor's cosine
similarity with itself is always exactly 1.0, the maximum possible value. `_topk_bottomk_indices`
and `_percentile_pairs_row` ranked the **full** row including that diagonal entry, so an anchor's
own id was guaranteed to land in its own top-k/top-percentile pool and could be emitted as
`graph_pos_id == anchor_id` (or, less commonly, `graph_neg_id == anchor_id` for a near-empty
bottom pool). That is a degenerate training example — the contrastive loss pulls the anchor's
graph embedding toward itself, contributing no learning signal.

This was flagged in the original execution log as an "intentional faithful port" of the legacy
`get_sample_10`/`get_sample_11` scripts, which have the identical unguarded-diagonal pattern. On
review, "faithful to a known-flawed legacy script" was not a good enough reason to leave in a
one-line-fixable degenerate-example bug, so it has been corrected:

- `_topk_bottomk_indices` (`samplers.py`) now takes an optional `exclude_index`: the anchor's own
  column is dropped from the row **before** `argsort`, so it can never appear in `top`/`bottom`.
- `_percentile_pairs_row` takes the same `exclude_index`, dropped before percentile thresholds are
  computed (so the anchor's 1.0 no longer skews `top_threshold` either) and before the candidate
  pool is built.
- Both `_rank_topk_one_direction` and `_percentile_one_direction` now look up the anchor's own
  index in `graph_sim.col_ids` (`graph_col_of.get(anchor_id)`) and pass it through. The **text**
  view (`text_g1_g2`) is cross-graph and has no diagonal, so it is untouched — only the
  within-graph `graph_g1`/`graph_g2` views needed the exclusion.
- The `supervised` sampler already excluded the anchor from its graph positive/negative candidate
  pools (`[nid for nid in graph_g1_ids if nid != anchor_id]`) and needed no change.

Added two direct regression tests
(`test_rank_topk_never_pairs_anchor_with_itself_as_graph_positive`,
`test_percentile_sampler_never_pairs_anchor_with_itself_as_graph_positive`) against the existing
hand-built fixture in `test_samplers.py`, whose `graph_g1`/`graph_g2` matrices already have a 1.0
diagonal. Confirmed these tests would have caught the original bug by re-running the pre-fix
ranking logic standalone: for anchor `a:1` with `k=2` and no exclusion, the top-2 by rank was
`['a:1', 'a:2']` — the anchor ranked as its own top-1 neighbor, as expected from cosine similarity
being maximized at 1.0 on the diagonal.

Re-ran `cga run training_data --dataset attack_cwe` after the fix: same example count (34,080 —
excluding the diagonal only changes which node fills the k-th rank slot, not how many slots exist)
and directly verified `graph_pos_id`/`graph_neg_id` against `anchor_id` across all 34,080 rows —
**0 self-pairs** in the real output (previously nonzero, un-measured in the initial run since the
bug was found afterward).

## Assumptions / limitations
- `cga validate` was not extended to recognize `training_data` artifacts (it currently only checks
  the four ingest tables); running it against a `training_data` run dir reports "0 table(s)
  validated" rather than failing — noted but left out of scope for this phase.
- `supervised` sampler's graph positive/negative are random draws within the anchor's own graph
  (no structural ground truth exists in `alignments.parquet`), matching what the architecture doc
  specifies ("positives taken from alignments.parquet"); negatives (text and graph) are uniform
  random rather than hard-negative-mined, consistent with the legacy design's scope.

## CLI commands to test and run this phase
```
uv run pytest tests/unit/pipelines/training_data -q
uv run pytest tests/unit -q
uv run python -m cga.cli run training_data --dataset attack_cwe
uv run python -m cga.cli config show training_data --dataset attack_cwe
```
