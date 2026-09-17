# Sprint 1.6 Execution Log — Per-node `train`/`val`/`test` split

## Motivation

Phase 1.5 (`training_data`) shipped `held_out` derived *only* from `alignments.parquet`. That was
correct for the leakage bug it was fixing, but left a gap the developer flagged afterward:
`attack_cwe` has no alignment ground truth (`dataset.alignments: null`), so `held_out` was `False`
for every node — no held-out set, no generalization check at all for that dataset. Meanwhile
`ncit_doid`'s node-level eligibility was entirely derived from its own OAEI `refs_equiv/test.tsv`
alignment pairs, not from an independent node partition.

Discussed with the developer:

1. Where should the split live? — `ingest`, as a new step writing `splits.parquet` alongside the
   other canonical tables, not buried in `training_data`. Split membership is a property of the
   dataset, not of a training run, so it must be computed once and shared by every downstream
   `training_data`/`evaluate` run via the manifest chain.
2. Should a dataset's own split (`ncit_doid`'s OAEI train/test) be followed when present? — No.
   Every dataset gets the same per-node partition, drawn independently of any alignment split.
3. Stratification policy — **proportional per graph**: each `graph_role` independently shuffled and
   cut into `train`/`val`/`test` by the same fractions (default 80/10/10), so neither graph is
   over/under-represented in either split regardless of relative graph size.
4. Held-out scope — **both** `val` and `test` nodes excluded from similarity/sampling, not just
   `test`.

This raised a real consequence: a uniform node split ignores `ncit_doid`'s own OAEI
`refs_equiv/test.tsv`, so an OAEI test pair could land in our `train` node split and get trained on
— while `evaluate` still scores Hits@k/MRR against that same OAEI test pair, silently reintroducing
leakage through a different door than Phase 1.5 closed. Presented three options; developer chose
**"union both exclusions"**: the uniform per-node split is the single source of truth for node
partitioning, but `held_out` (the sampling-eligibility flag) is the union of *both*
`split in {val, test}` *and* "endpoint of a `test`-split alignment pair." Vacuous for `attack_cwe`
(no alignments); for `ncit_doid` it keeps OAEI's own test pairs genuinely unseen regardless of where
the uniform draw happened to place them, so reported Hits@k/MRR stay comparable to published
OAEI Bio-ML numbers.

Architecture amended first (§5.1.1 new, §5.3, §6.5 renumbered, §6.7, §7.5 new, §7.7, §8.4) and the
Phase 1.6 checklist written into `dev_docs/sprints/sprint_1.md` before any code changed, per the
project's established practice of fixing the spec before implementing against it.

## Implementation

**`data/schemas.py`**:
- `NodeSplit = Literal["train", "val", "test"]`.
- `SPLITS_SCHEMA` / `SplitRecord` (§6.5) — `node_id` pk, `graph_role`, `split`. One row per node,
  every dataset, including those with no alignments.
- `UNIVERSE_SCHEMA` / `UniverseRecord` gained `split: NodeSplit = "train"`, carried verbatim from
  `splits.parquet`; `held_out` docstring updated to describe the union rule.

**`data/graph_store.py`**: `write_splits`/`read_splits`. `write_splits` enforces schema, pk
uniqueness, an FK against `nodes.parquet`, **and** completeness — every node in `nodes.parquet` must
have exactly one `splits.parquet` row, raising `ArtifactValidationError` listing the first few
missing ids otherwise. This is stricter than the other four tables (which tolerate partial/empty
data) because a split's whole point is to cover every node.

**`data/aligned.py`**: `NodeUniverse` gained `split: list[str] | None = None` (defaults to
all-`"train"` for callers that don't track it, e.g. any pre-existing test construction), inserted
before the existing `held_out` field. Checked all call sites use keyword arguments before adding it
(`grep -rn "NodeUniverse("`) — every one did, so no positional-argument breakage.

**`pipelines/ingest/splitter.py`** (new) — `split_nodes(node_ids_by_role, *, fractions, seed)`:
for each `graph_role` independently, shuffles with a **fresh** `random.Random(f"{seed}:{role}")`
instance (first attempt used a `(seed, role)` tuple, which `random.Random` rejects — `TypeError:
The only supported seed types are: None, int, float, str, bytes, and bytearray`; fixed to an
f-string) and cuts into contiguous `train`/`val`/`test` blocks. `min()` guards on the train/val
block sizes prevent rounding from over-allocating on small graphs (verified with a 3-node /
1-node fixture). `validate_fractions` enforces exactly 3 non-negative values summing to 1.0.

**`pipelines/ingest/pipeline.py`**: new `SplitNodesStep` (step 6, before the renumbered
`validate_and_manifest` step 7) reads `ingest.split.fractions`/`ingest.split.seed` (seed defaults to
`ctx.seed` when unset), groups nodes by `graph_role`, calls `split_nodes`, writes `splits.parquet`.
`ValidateAndManifestStep` gained `n_nodes_by_split_by_graph_role` in `stats.json` and records
`splits` as a fifth manifest output.

**`configs/pipelines/ingest.yaml`**: `split.fractions: [0.8, 0.1, 0.1]`, `split.seed: null`.

**`pipelines/training_data/pipeline.py`**:
- `BuildUniverseStep` now reads `splits.parquet`, carries `split` onto `universe.parquet`, and
  computes `held_out` as `split in {val, test} OR node in a test-split alignments.parquet row` —
  the union rule. Two module-level frozensets (`_HELD_OUT_NODE_SPLITS`, `_HELD_OUT_ALIGNMENT_SPLITS`)
  replace the removed `training_data.held_out_splits` config key, since which splits are held out is
  now a fixed architectural rule, not a per-run tuning knob.
- `SampleStep`'s `supervised` branch now filters alignment pairs by `held_out` on **both** endpoints
  (looked up from `state.universe.held_out`) rather than the alignment's own `split` column — a
  `train`-split alignment pair whose node landed in our `val` split is now correctly excluded, which
  the old alignment-split-only filter would have missed.
- `SplitStep` renamed to `AssignExampleSplitStep`: every example now inherits its anchor's
  `universe.split` directly. The old independent random anchor-shuffle (`val_fraction`,
  `random.Random(ctx.seed).shuffle(anchors)`) is gone — `training_data` no longer draws any split of
  its own, node partitioning is `ingest`'s job alone.
- `ValidateAndManifestStep`'s leakage assertion now applies to **every** sampler including
  `supervised` (previously exempted) — now that `SampleStep` pre-filters held-out endpoints out of
  supervised's alignment pool too, a supervised example referencing a held-out node is an invariant
  violation everywhere, not a case to special-case around.

**`configs/pipelines/training_data.yaml`**: dropped `held_out_splits` and the `split` block entirely;
added a comment pointing at `ingest`'s `split.fractions`/`seed` as the actual source.

## Files modified or created
- `dev_docs/target_architecture.md` — §5.1.1 (new), §5.3, §6.5 (new `splits.parquet`, renumbers
  `examples.parquet` to §6.6, embedding-store tables to §6.7), §7.5 (new `ingest.yaml`, renumbers
  `embed.yaml`→§7.6, `training_data.yaml`→§7.7, `gcl_train.yaml`→§7.8, `evaluate.yaml`→§7.9,
  experiment overlay→§7.10), §8.4, data-flow diagram (§9), all fixed cross-references.
- `dev_docs/sprints/sprint_1.md` — new Phase 1.6, checklist completed; downstream phases (`gcl_train`,
  `evaluate`, finalize) renumbered by the developer directly; two `§7.x` references in the pending
  `gcl_train`/`evaluate` checklist items updated to match.
- `src/cga/data/schemas.py`, `src/cga/data/graph_store.py`, `src/cga/data/aligned.py` — split schema/
  store/`NodeUniverse` changes above.
- `src/cga/pipelines/ingest/splitter.py` — new.
- `src/cga/pipelines/ingest/pipeline.py` — `SplitNodesStep`, updated `ValidateAndManifestStep`.
- `src/cga/pipelines/training_data/pipeline.py` — `BuildUniverseStep`, `SampleStep`,
  `AssignExampleSplitStep` (renamed from `SplitStep`), `ValidateAndManifestStep`.
- `configs/pipelines/ingest.yaml`, `configs/pipelines/training_data.yaml`.
- `tests/unit/pipelines/ingest/test_splitter.py` — new (9 tests).
- `tests/unit/pipelines/ingest/test_pipeline.py` — 3 new tests (`splits.parquet` completeness,
  reproducibility under a fixed seed, configurable fractions in `stats.json`); renamed
  `test_ingest_writes_four_tables_and_stats` → `test_ingest_writes_five_tables_and_stats`.
- `tests/unit/pipelines/training_data/test_pipeline.py` — `_write_ingest_run` gained a `node_splits`
  override param and now writes `splits.parquet`; `_build_config` dropped `held_out_splits`/`split`;
  2 new tests for the node-split half of `held_out` and the union rule
  (`test_build_universe_flags_held_out_from_node_split`,
  `test_build_universe_unions_node_split_and_alignment_held_out`); rewrote
  `test_supervised_split_keeps_test_alignment_out_of_train_examples` to assert the exact inherited
  split rather than a now-trivial `in ("train", "val")` check; added
  `test_supervised_sampler_excludes_pairs_with_a_val_split_endpoint` covering the case the old
  alignment-split-only filter would have missed.

## Verification
- `uv run pytest tests/unit -q` → **161 passed** (0 failures; up from 146 before this phase: +9
  splitter, +3 ingest, +3 net training_data changes).
- `cga run ingest --dataset attack_cwe` → 1,217 nodes / 1,922 edges (unchanged from Phase 1.3) +
  **1,217 `splits.parquet` rows**, one per node. `stats.json.n_nodes_by_split_by_graph_role`:
  ```json
  {"graph_1": {"train": 207, "val": 26, "test": 26},
   "graph_2": {"train": 766, "val": 96, "test": 96}}
  ```
  207+26+26 = 259 (`attack`), 766+96+96 = 958 (`cwe`) — both ≈80/10/10, confirming the per-graph
  proportional policy independent of the two graphs' very different sizes.
- Re-ran `embed` (`hf_mean_pool` text, `deepwalk` structural) against the new ingest run, then
  `cga run training_data --dataset attack_cwe`:
  ```json
  {"n_examples": 27270, "n_examples_by_split": {"train": 27270},
   "n_held_out_nodes": 227, "n_universe_nodes": 1136, "sampler": "rank_topk"}
  ```
  Universe split breakdown (1,136 intersection-eligible nodes): `{"train": 909, "val": 111,
  "test": 116}` — ≈80/10/10 as expected of the ~1,217-node partition restricted to the
  embedding-store intersection. `n_examples` dropped from Phase 1.5's 34,080 to **27,270**
  (≈20% fewer, matching the ≈20% of nodes now held out) — the predicted consequence, confirmed.
  Directly checked all five id columns of `examples.parquet` (34,080... now 27,270 rows) against the
  227 held-out node ids: **zero** held-out nodes referenced anywhere. Every example's `split` is
  `"train"`, matching the architecture doc's expectation that unsupervised anchors can never be
  `val`/`test` since `build_universe`/`similarity`/`sample` already excluded them.

## Assumptions / limitations
- `evaluate` (Phase 1.8, not yet implemented) is specified to report diagnostics both over all nodes
  and restricted to the `test` split (§8.4), so `attack_cwe` gets a genuine held-out number despite
  having no alignment ground truth — this phase only lays the `splits.parquet` groundwork; the
  diagnostics-restriction logic itself is Phase 1.8's responsibility.
- `cga validate` still only recognizes the four original ingest tables (noted as a limitation in the
  Phase 1.5 log) and now also does not know about `splits.parquet`; extending it is out of scope here.
- The splitter's per-graph seed derivation (`f"{seed}:{role}"`) means changing `graph_1`'s
  `display_name`/`key` in a dataset config would not change the split (the string uses the fixed
  `graph_1`/`graph_2` role, not the dataset's `graph_key`), which is intentional — the partition
  should be stable across cosmetic config renames, not just across seeds.

## CLI commands to test and run this phase
```
uv run pytest tests/unit/pipelines/ingest/test_splitter.py -q
uv run pytest tests/unit/pipelines/ingest -q
uv run pytest tests/unit/pipelines/training_data -q
uv run pytest tests/unit -q
uv run python -m cga.cli run ingest --dataset attack_cwe
uv run python -m cga.cli run embed --dataset attack_cwe
uv run python -m cga.cli run embed --dataset attack_cwe --config configs/pipelines/embed_deepwalk.yaml
uv run python -m cga.cli run training_data --dataset attack_cwe
```
