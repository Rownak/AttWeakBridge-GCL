# Sprint 1, Phase 1.7 — `gcl_train` pipeline

## What was implemented

The fourth pipeline in the training chain: `pipelines/gcl_train/` (`models.py`, `losses.py`,
`trainer.py`, `pipeline.py`), `configs/pipelines/gcl_train.yaml`,
`configs/experiments/attack_cwe_secroberta_gcn.yaml`, CLI wiring (`_run_root` for `gcl_train` in
`cli.py`), and `tests/unit/pipelines/gcl_train/`.

## Validation before implementation

Per the task instructions, the proposed training workflow was independently checked against standard
graph contrastive learning practice before writing any code:

- **Architecture**: `DualEncoder` fuses two GNN towers (one over text-embedding features, one over
  structural-embedding features) via full-batch, transductive message passing over the single
  `edge_index` built from `edges.parquet`. Since edges are intra-graph only, `graph_1`/`graph_2` form
  disjoint connected components and a GCN/GAT layer never mixes information across the pair — sound
  and dataset-agnostic.
- **Loss**: `alpha·L_text + (1−alpha)·L_graph`, both terms computed on the *final fused* embeddings
  (not raw input features), using `examples.parquet`'s cross-graph text pair and within-graph
  structural pair columns exactly as produced by `training_data` — no mismatch between what the
  sampler produced and what the loss consumes.
- **Leakage check**: `examples.parquet` already excludes held-out (`val`/`test`-split) nodes from
  anchor/positive/negative (enforced by `training_data`'s `ValidateAndManifestStep`). `gcl_train.load`
  re-verifies this independently (raises `ArtifactValidationError` if any example id falls outside the
  universe) rather than trusting the upstream guarantee blindly.
- **Transductive scope** (confirmed with the developer): the forward pass covers every node in the
  universe, including `val`/`test`, because message passing needs their structure/text features and
  `evaluate` (Phase 1.8) needs their embeddings — this is standard practice for transductive GNNs (cf.
  citation-network benchmarks), not label leakage, since held-out nodes never appear as loss-supervised
  anchor/positive/negative rows.
- **`jsd` loss dropped** (confirmed with the developer): the legacy `JSDContrastiveLoss` treats a
  softmax over an L2-normalized embedding's dimensions as a probability distribution before computing
  Jensen-Shannon divergence — this has no principled interpretation (softmax over embedding
  coordinates isn't a natural probability), so it was not ported. Only `triplet` and `infonce` are
  registered losses; `configs/pipelines/gcl_train.yaml` documents the omission.

No leakage, mismatched positive/negative usage, or other invalidating issue was found, so implementation
proceeded per the architecture doc.

## Approach and key decisions

- **`models.py`** — `GCN`, `GAT`, `DualEncoder` ported from `legacy/GCL/models/GNN_dual_loss2.py`,
  generalized: no dataset-specific dimensions or hardcoded feature names; `build_dual_encoder(...)`
  constructs matching towers from `gcl_train.model` config (`encoder: gcn|gat`, `hidden_dim`,
  `out_dim`, `gat_heads`). `GATConv`'s second layer now sets `concat=False` (the legacy code relied on
  `heads=1`'s implicit averaging=identity behavior; being explicit makes the shape contract clear for
  any `gat_heads` value).
- **`losses.py`** — `triplet` (cosine-distance margin loss, ported faithfully) and `infonce`
  (in-batch negatives: every row's negative also serves as a negative for every other row in the
  batch, standard InfoNCE practice, strictly more informative than the legacy one-negative-per-anchor
  version without introducing any new positive/negative source). Both registered via
  `cga.core.registry.losses`.
- **`trainer.py`** — `train_model(...)` runs the full-batch loop, writes one JSON line per epoch to
  `metrics.jsonl`, and checkpoints every `checkpoint_every` epochs plus always at the final epoch.
  Checkpoints embed `universe_sha256` (`sha256_of_universe`, hash of the ordered node-id list);
  `load_checkpoint(...)` raises `ValueError` if the recorded hash doesn't match the caller's universe.
- **`pipeline.py`** — `LoadStep` resolves the `training_data` run via its own manifest (not a
  hand-passed embedding-store ref), so `gcl_train` cannot read a text/graph store mismatched from what
  `training_data` actually used; `ctx.resolve(...)` is called again for each upstream artifact so this
  run's own manifest records them (per `RunContext.resolve`'s "one call" contract). `TrainStep` builds
  the model/optimizer from config and calls `trainer.train_model`. `ExportStep` writes final embeddings
  as an `AlignedMatrix` under `node_embeddings/`. `ManifestStep` writes `config.resolved.yaml` and a
  summary (`training_data_run_id`, `text_embedding_run_id`, `graph_embedding_run_id`,
  `universe_sha256`, `n_nodes`, `n_examples`, `n_examples_train`) per §6.9.

## Issues found and fixed during verification (not caught by unit tests alone)

Running the pipeline against the real `attack_cwe` artifacts surfaced two integration bugs invisible to
hand-built fixtures:

1. **Manifest input-name mismatch**: `training_data`'s `BuildUniverseStep` records its ingest input as
   `"ingest"` (via `ctx.resolve("ingest", ...)`), not `"ingest_run"`. `gcl_train.load` was looking for
   the wrong name. Fixed to look up `"ingest"`.
2. **Out-of-universe edges are expected, not an error**: `edges.parquet` covers every ingested node,
   but `universe.parquet` (under `universe_policy: intersection`) only keeps nodes present in *both*
   the text and graph embedding stores — a node with missing/empty text legitimately has no text
   embedding and so is absent from the universe. The real `attack_cwe` run had 1 such edge out of 1922.
   **Asked the developer**: should such edges be dropped (logged) or treated as fatal, same as an
   out-of-universe example id? Decided: drop and log the count — this is a routine, expected
   consequence of the intersection policy, unlike an out-of-universe *example* id, which would indicate
   the `training_data` leakage check itself had failed. Implemented in `LoadStep` with an info-level log
   line (`"load: dropped %d/%d edges with an endpoint outside the universe"`).
3. **Loss registry not populated under the real CLI**: `pipeline.py` only imported `models` and
   `trainer`, not `losses`, so `cga.core.registry.losses` was empty when the CLI actually ran the
   pipeline (a fixture-order accident in the test suite masked this — `test_losses.py` happened to run
   before `test_pipeline.py` in the same pytest session, registering the losses as a side effect).
   Fixed by importing `cga.pipelines.gcl_train.losses` from `pipeline.py` so it always registers.

## Files modified or created

- `src/cga/pipelines/gcl_train/__init__.py`, `models.py`, `losses.py`, `trainer.py`, `pipeline.py`
  (new)
- `src/cga/pipelines/__init__.py` — import `gcl_train` for registration
- `src/cga/cli.py` — `_run_root` entry for `gcl_train` → `models/gcl/{dataset_id}`
- `configs/pipelines/gcl_train.yaml`, `configs/experiments/attack_cwe_secroberta_gcn.yaml` (new)
- `tests/unit/pipelines/gcl_train/__init__.py`, `test_models.py`, `test_losses.py`, `test_trainer.py`,
  `test_pipeline.py` (new)
- `dev_docs/sprints/sprint_1.md` — Phase 1.7 checklist marked complete

## Assumptions / limitations

- `evaluate` (Phase 1.8) is out of scope here — `gcl_train.monitor` config exists but is not wired to
  any in-loop ranking code yet (that wiring is a Phase 1.8 task per the architecture doc).
- No pydantic validation schema for `gcl_train`'s config block exists yet — config validation across
  all pipelines is still informal (dict-based), consistent with every other pipeline shipped so far in
  this sprint; not something Phase 1.7 was scoped to add.
- GAT's second layer uses `concat=False` (single-head passthrough), a minor explicit-vs-implicit
  deviation from the legacy code's reliance on `heads=1` behaving the same way — behaviorally
  equivalent for the documented default (`gat_heads: [8, 1]`).

## CLI commands to test and run this phase

```bash
uv run pytest tests/unit/pipelines/gcl_train -q
uv run pytest tests/unit -q

# Requires prior ingest/embed/training_data runs for attack_cwe (already present in this repo):
cga run gcl_train --dataset attack_cwe --set gcl_train.train.epochs=5
cga inspect models/gcl/attack_cwe/<run_id>
```

Verified: 16/16 new unit tests pass, 177/177 full unit suite passes, and a real 5-epoch run against
the existing `attack_cwe` `training_data` artifacts produced `node_embeddings/` of shape `(1136, 128)`,
a 5-line `metrics.jsonl` with monotonically decreasing loss (0.85 → 0.46), and a `manifest.json` with
`status: success` and full upstream lineage (`training_data_run_id`, `text_embedding_run_id`,
`graph_embedding_run_id`, `universe_sha256`).
