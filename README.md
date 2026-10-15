# AttWeakBridge-GCL

Aligning two text-attributed graphs with graph contrastive learning.

The system learns **one embedding space in which nodes from two separate graphs are directly
comparable**, combining text embeddings, structural embeddings, and a contrastive objective. The
original motivation is bridging MITRE ATT&CK attack patterns with MITRE CWE weaknesses, but the
pipeline is **dataset-agnostic**: ATT&CK–CWE is one dataset among several.

## Problem setting

The input is always a **graph pair** — two independent graphs with no edges between them:

| Concept | Meaning |
|---|---|
| `graph_1` / `graph_2` | The two sides. Roles are fixed per dataset and carry no semantics beyond left/right. |
| Node | An entity in one graph, identified by a stable `node_id` = `"{graph_key}:{source_id}"`. |
| Edge | A typed relation between two nodes **of the same graph**. Cross-graph edges are never ingested. |
| Node text | One or more text variants per node (name, description, synonyms, …). |
| Alignment pair | Optional ground-truth correspondence between a `graph_1` and a `graph_2` node, used for evaluation and optionally supervision. |

Two datasets ship today:

| `dataset_id` | `graph_1` | `graph_2` | Source format | Alignments |
|---|---|---|---|---|
| `attack_cwe` | MITRE ATT&CK | MITRE CWE | STIX JSON, CWE XML | none |
| `ncit_doid` | NCI Thesaurus | Disease Ontology | OWL/XML | `refs_equiv/*.tsv` |

## Architecture in one picture

Everything dataset-specific lives in a **dataset adapter**. Only the first pipeline is dataset-aware;
every stage after it sees one canonical schema and is identical for all datasets. Each stage is a
registered pipeline driven by layered YAML and invoked through a single `cga` CLI.

```
data/raw/{dataset_id}/              raw source files (immutable)
        │
        ▼  cga run ingest           << the only dataset-aware step (adapter registry)
data/processed/{dataset_id}/{run_id}/
        nodes / edges / texts / alignments / splits .parquet
        │
        ├──── texts ────► cga run embed  (hf_mean_pool)  ─┐   text vectors
        └──── edges ────► cga run embed  (deepwalk)      ─┤   structural vectors
                                                          │   artifacts/embeddings/…
                                                          ▼
                                       cga run training_data
                          artifacts/training_data/{dataset_id}/{run_id}/
                            universe.parquet   (node_id ↔ row, split, held_out)
                            examples.parquet   (anchor / positive / negative)
                                                          │
                                                          ▼
                                       cga run gcl_train
                          models/gcl/{dataset_id}/{model_id}/{run_id}/
                            node_embeddings/   ← the research output
                                                          │
                                                          ▼
                     cga run evaluate   (learned embeddings + baselines, side by side)
                          artifacts/evaluations/{dataset_id}/{run_id}/metrics.parquet
                                                          │
                                                          ▼
                                       cga run plot
                          artifacts/plots/{dataset_id}/{run_id}/figures/*.png
```

Three properties are worth knowing up front:

- **Stable identity.** Every vector, edge, and training example carries a `node_id`. Dense arrays
  (`.npy`) are never read without their `index.parquet` sidecar.
- **Immutable runs.** Each run writes a new `{run_id}/` directory plus a `manifest.json` recording
  its config hash, upstream run ids, and input hashes. A `LATEST` file points at the newest run, so
  downstream configs can say `ref: latest`.
- **Config over code.** Adding a dataset means one adapter plus one YAML file; adding a baseline,
  metric, or figure usually means only YAML.

## Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- An NVIDIA GPU with CUDA 12.4 wheels is assumed by `pyproject.toml` (text embedding and training
  run on CPU too, just slowly)

## Installation

```bash
git clone <repo-url>
cd AttWeakBridge-GCL
uv sync
```

This installs the `cga` console entry point. Verify the install and see what is registered:

```bash
uv run cga list          # pipelines, adapters, embedders, samplers
uv run pytest tests/unit -q
```

All commands below are shown with `uv run`; inside an activated venv you can drop that prefix.

## Datasets

Raw source files are **not** committed. Place them under `data/raw/{dataset_id}/`, matching the
filenames declared in the dataset's config:

```
data/raw/attack_cwe/
  ics-attack.json          # ATT&CK STIX bundle
  cwec_v4.12.xml           # CWE catalog

data/raw/ncit_doid/
  ncit.owl
  doid.owl
  refs_equiv/{train,test,test.cands}.tsv
```

The filenames, graph keys, text variants, and alignment paths all come from
[configs/datasets/attack_cwe.yaml](configs/datasets/attack_cwe.yaml) and
[configs/datasets/ncit_doid.yaml](configs/datasets/ncit_doid.yaml) — change those files if your
copies are named differently.

To add a new dataset: write an adapter under [src/cga/adapters/](src/cga/adapters/), add a
`configs/datasets/<name>.yaml`, drop the sources in `data/raw/<name>/`, and run the same pipelines
unchanged.

## Running the pipeline

Each command is one stage. By default every stage picks up the `latest` run of its upstream stage,
so running them in order just works.

```bash
# 1. Ingest: raw sources -> canonical nodes/edges/texts/alignments/splits
uv run cga run ingest --dataset attack_cwe

# 2a. Text embeddings (SecRoBERTa mean-pooled)
uv run cga run embed --dataset attack_cwe

# 2b. Structural embeddings (DeepWalk over the graph)
uv run cga run embed --dataset attack_cwe --config configs/pipelines/embed_deepwalk.yaml

# 3. Contrastive training examples (similarity -> sampling -> split)
uv run cga run training_data --dataset attack_cwe

# 4. Train the dual encoder (GCN by default; --set for a quick smoke run)
uv run cga run gcl_train --dataset attack_cwe
uv run cga run gcl_train --dataset attack_cwe --set gcl_train.model.encoder=gat

# 5. Compare methods (baselines + trained models) under one evaluation
uv run cga run evaluate --dataset attack_cwe --experiment neighborhood_compare

# 6. Render figures from the evaluation's metrics.parquet
uv run cga run plot --dataset attack_cwe --config configs/plots/neighborhood_compare.yaml
```

Useful flags and helper commands:

```bash
uv run cga run ingest --dataset attack_cwe --dry-run       # resolve config, run nothing
uv run cga config show gcl_train --dataset attack_cwe      # print the fully-resolved config
uv run cga validate data/processed/attack_cwe/<run_id>     # re-check schemas + manifest status
uv run cga inspect  artifacts/evaluations/attack_cwe/<run_id>   # walk the upstream run chain
```

Note that step 5 compares several **methods** at once — each baseline embedding store and each
trained model — so it expects the embedding stores and both `gcn`/`gat` runs listed in
[configs/experiments/neighborhood_compare.yaml](configs/experiments/neighborhood_compare.yaml) to
exist. Trim that `methods:` list if you have only run a subset.

## Sweeping a hyperparameter

`cga sweep --spec <file>` expands a YAML spec's array-valued config paths into one `cga run` per
combination, tagging each run's manifest with a `sweep_point_id` so results stay traceable back to
the exact overrides that produced them. Keys are crossed (cartesian product) by default; `zip:`
pairs keys by index instead; `fixed:` pins values that every point needs but that don't vary. A
failed combination is logged and skipped rather than aborting the whole grid. See
[dev_docs/sweep_experiment_plan.md](dev_docs/sweep_experiment_plan.md) and
[dev_docs/sprints/sprint_2.md](dev_docs/sprints/sprint_2.md) for the design and per-phase results.

```bash
# training_data sampler sweep (Phase 2.1): rank_topk.k, and percentile.{top_pct,bottom_pct}
uv run cga sweep --spec configs/sweeps/rank_topk_k.yaml
uv run cga sweep --spec configs/sweeps/percentile_pct.yaml

# gcl_train loss-mix sweep (Phase 2.2): alpha in {0.0, 0.25, 0.5, 0.75, 1.0},
# sampler held fixed at Phase 2.1's winner (rank_topk, k=5)
uv run cga sweep --spec configs/sweeps/alpha.yaml

# Compare every sweep point's evaluate metrics side by side, then render figures
uv run cga run evaluate --dataset attack_cwe --experiment sampler_sweep_compare
uv run cga run plot --dataset attack_cwe --config configs/plots/sampler_sweep_compare.yaml

uv run cga run evaluate --dataset attack_cwe --experiment alpha_sweep_compare
uv run cga run plot --dataset attack_cwe --config configs/plots/alpha_sweep_compare.yaml
```

Each `configs/sweeps/*.yaml` spec's own header comment documents which config paths it sweeps and
why; each `configs/experiments/*_sweep_compare.yaml`'s `methods:` list must have a `model_id`/
`ref: latest` entry per sweep point, matching the ids that spec produces.

## Supervised run on gold alignments (`ncit_doid_supervised`)

[configs/datasets/ncit_doid_supervised.yaml](configs/datasets/ncit_doid_supervised.yaml) is
`ncit_doid` with the gold `refs_equiv` alignments switched on, under its own dataset id so its
artifacts never touch the unsupervised `ncit_doid` lineage. Gold **train** pairs are the positive
cross-graph examples (the `supervised` sampler); gold **test** pairs are held out of every example
and are what the `gold_alignment` evaluator retrieves. Plan and results:
[dev_docs/sprints/sprint_4.md](dev_docs/sprints/sprint_4.md),
[dev_docs/execution_logs/sprint_4.md](dev_docs/execution_logs/sprint_4.md).

```bash
# Ingest with an all-train node split, so held-out nodes come only from gold test endpoints
uv run cga run ingest --dataset ncit_doid_supervised --set "ingest.split.fractions=[1.0,0.0,0.0]"
# Embeddings are reused: copy artifacts/embeddings/ncit_doid/<embedder_id>/<run> (and its LATEST) to
# artifacts/embeddings/ncit_doid_supervised/<embedder_id>/ -- no re-embedding needed.

uv run cga run training_data --dataset ncit_doid_supervised   --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert   --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5
uv run cga run gcl_train --dataset ncit_doid_supervised   --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5   --set gcl_train.model.model_id=gcn-supervised --set gcl_train.train.epochs=100

uv run cga run evaluate --dataset ncit_doid_supervised --experiment ncit_doid_gold_compare
uv run cga run plot --dataset ncit_doid_supervised --config configs/plots/ncit_doid_gold_compare.yaml
```

Registered evaluators (`evaluate.evaluations[].kind`): `neighborhood`, `cluster_quality`, and
`gold_alignment` — Hits@K and MRR for held-out gold pairs, in `global` (all graph_2 nodes, other
gold targets of the source removed) and `candidates` (the pair's `test.cands` pool) modes.

## SapBERT-residual graph adapter (`ncit_doid_srga`)

[configs/datasets/ncit_doid_srga.yaml](configs/datasets/ncit_doid_srga.yaml) is `ncit_doid_supervised`
plus a `val` split: `alignments.val_split: {fraction: 0.15, seed: 42}` moves 15% of the gold train
sources (all their pairs) to `val`, whose endpoints are held out like test. Every choice is made on
`val`; `test` is read once per finished configuration. The SRGA encoder
(`gcl_train.model.encoder: residual_adapter`) exports `z = [x̂ ; sqrt(lambda) * normalize(h)]`: the
frozen SapBERT vector next to a graph/learned correction `h`, so `lambda = 0` ranks exactly like
SapBERT. Adapters: `neighbor_mean` (parameter-free, optional `directional`), `mlp` (no edges),
`relational` (parent/child relational layer) and `gcn` (the paper's text branch: `fc1`, the 2-layer `GCN` tower and `fc3`;
`model.residual_floor: false` drops the SapBERT block); `loss.name: infonce_full` trains with full-softmax
InfoNCE over every non-held-out candidate, and `model.edge_rewire` gives a degree-preserving rewired
control. Design, plan and results:
[dev_docs/design_docs/sapbert_residual_graph_adapter.md](dev_docs/design_docs/sapbert_residual_graph_adapter.md),
[dev_docs/sprints/sprint_5.md](dev_docs/sprints/sprint_5.md), `dev_docs/execution_logs/sprint_5_*.md`.

```bash
uv run cga run ingest --dataset ncit_doid_srga --set "ingest.split.fractions=[1.0,0.0,0.0]"
# Embeddings are reused: copy artifacts/embeddings/ncit_doid_supervised/<embedder_id>/ to
# artifacts/embeddings/ncit_doid_srga/<embedder_id>/ (same procedure as above).
uv run cga run training_data --dataset ncit_doid_srga   --set training_data.inputs.text_embeddings.embedder_id=hf_mean_pool-sapbert   --set training_data.sampler.name=supervised --set training_data.sampler.negatives_per_positive=5

# Training-free A1: neighbour mean, directional, lambda 0.25 (no optimizer loop)
uv run cga run gcl_train --dataset ncit_doid_srga   --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5   --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=neighbor_mean   --set gcl_train.model.lambda=0.25 --set gcl_train.model.directional=true --set gcl_train.model.model_id=srga-nb-l0.25-dir

# Trained A3 (mlp) / A4 (relational, --set gcl_train.model.directional=true); one run per seed
uv run cga run gcl_train --dataset ncit_doid_srga --set seed=42   --set gcl_train.inputs.training_data_sampler_id=supervised-negatives_per_positive5   --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=mlp --set gcl_train.model.lambda=0.25   --set gcl_train.loss.name=infonce_full --set gcl_train.loss.temperature=0.05   --set gcl_train.optim.weight_decay=0.0001 --set gcl_train.optim.lr=0.001 --set gcl_train.train.epochs=50   --set gcl_train.model.model_id=srga-mlp-l0.25-s42

uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_compare   # gold_val + gold_test
uv run cga run plot --dataset ncit_doid_srga --config configs/plots/ncit_doid_srga_compare.yaml
uv run python scripts/srga_paired_ranks.py --dataset ncit_doid_srga --split val   --group A3="srga-mlp-l0.25-s{seed}"   # paired wins/losses vs SapBERT
uv run python scripts/srga_stratified.py --dataset ncit_doid_srga --split test   --group A3="srga-mlp-l0.25-s{seed}"  # fixed/broken by stratum
uv run python scripts/srga_pool_diagnostic.py --dataset ncit_doid_srga --split val   --group A3="srga-mlp-l0.25-s{seed}"   # MRR on all / held-out-only / trainable-only candidate pools
```

In `--set`, write small floats as `0.0001`, not `1e-4` (YAML reads the latter as a string).

## Configuration

All configuration is YAML under [configs/](configs/); nothing else is configuration. Layers are deep
merged, later layers win, and lists replace wholesale:

```
configs/base.yaml                       paths, seed, device, logging
  + configs/datasets/<--dataset>.yaml   the graph pair: sources, graph keys, text variants
  + configs/pipelines/<pipeline>.yaml   defaults for the pipeline being run
  + configs/experiments/<--experiment>.yaml   named overlay spanning pipelines
  + each --config <file>                extra file, applied in order
  + each --set key.path=value           final CLI override
```

Only `${paths.<key>}` and `${env:VAR}` interpolation is supported. The resolved config is hashed into
`config_hash` and snapshotted as `config.resolved.yaml` inside the run directory.

Two overlay flavors are worth distinguishing:

- `configs/experiments/` — named overlays selected with `--experiment`, each top-level key overlaying
  the matching pipeline (e.g. a method/evaluation set, or an encoder + loss combination).
- `configs/plots/` — figure specs selected with `--config`, since a figure selects on
  `evaluation_id` / `metric_name` and is reusable across datasets.

## Artifacts

Outputs are split across three roots by lifecycle, each partitioned by `dataset_id` and then by an
immutable `run_id`:

```
data/processed/{dataset_id}/{run_id}/          ingest: the canonical tables
artifacts/embeddings/{dataset_id}/{embedder_id}/{run_id}/
                                               vectors.npy, index.parquet, status.parquet
artifacts/training_data/{dataset_id}/{run_id}/ universe.parquet, examples.parquet
models/gcl/{dataset_id}/{model_id}/{run_id}/   checkpoints/, metrics.jsonl, node_embeddings/
artifacts/evaluations/{dataset_id}/{run_id}/   metrics.parquet, metrics.json, plots/
artifacts/plots/{dataset_id}/{run_id}/         figures/*.png
```

Every run directory carries a `manifest.json` (config hash, upstream runs, input hashes, status) and
a `run.log`. `embedder_id` (e.g. `hf_mean_pool-pt_SecRoBERTa`, `deepwalk-deepwalk`) and `model_id`
(`gcn`, `gat`) keep parallel variants from overwriting each other's `LATEST` pointer, which is what
makes side-by-side comparison in `evaluate` possible.

## Repository structure

```
configs/            all YAML: base, datasets/, pipelines/, experiments/, plots/
src/cga/
  cli.py            the `cga` entry point
  core/             config layering, run context, manifests, registries, IO, seeds
  data/             canonical schemas and stores (ids, graph, embeddings, examples)
  adapters/         >>> the only dataset-specific code <<< attack_cwe/, ncit_doid/
  pipelines/        ingest/, embed/, training_data/, gcl_train/, evaluate/, plot/
tests/              unit/, integration/, regression/
data/               raw/ (source files) and processed/ (ingest output)
artifacts/ models/  derived outputs, partitioned by run_id
legacy/             frozen pre-refactor code, kept for reference only
dev_docs/           architecture and sprint documentation
```

## Documentation

- [dev_docs/target_architecture.md](dev_docs/target_architecture.md) — the full design: canonical
  schemas, adapter interface, per-pipeline contracts, config reference, data flow.
- [dev_docs/sprints/](dev_docs/sprints/) — sprint plans ([sprint_0](dev_docs/sprints/sprint_0.md):
  environment and legacy cleanup; [sprint_1](dev_docs/sprints/sprint_1.md): building the
  architecture above).
- [dev_docs/execution_logs/](dev_docs/execution_logs/) — per-phase implementation notes, including
  verified run counts and commands.
- [dev_docs/existing_architecture.md](dev_docs/existing_architecture.md) — the legacy pipeline that
  `legacy/` still contains.
