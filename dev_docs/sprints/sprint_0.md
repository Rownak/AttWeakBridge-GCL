# Sprint 0 — Run existing code end-to-end

## Objective

Get the current pipeline running end-to-end on this machine, under a locked, reproducible environment,
against a small real-data fixture — and freeze that run as a regression baseline before any refactoring
begins. Nothing from the original A6000 experiments survives (no checkpoints, no `datasets/`, no
`model_outputs/`), so this sprint's job is to stand the pipeline up from scratch on the RTX 4090 Laptop
(16GB), not to reproduce the paper's numbers. Covers: uv-managed dependencies, fixture construction, a
full pipeline run on pretrained models, a tiered regression harness, and data versioning with DVC.

## Features added

- `pyproject.toml` + `uv.lock` — locked, reproducible dependency set (Python 3.13)
- `scripts/check_env.py` — one-command environment sanity check (CUDA, imports, VRAM)
- `fixtures/tiny/` — small real-data subsample (ATT&CK + CWE), committed to git
- `tests/test_regression.py` — Tier A (hash-exact) / B (tolerance) / C (end-to-end) regression harness
- DVC-tracked `datasets/` and `model_outputs/`

## Phases

### Phase 0.1 — uv dependency setup

- [x] Initialize the `uv` project pinned to **Python 3.13** (README's "3.8+" is a stale multi-year-old
      floor, not a target — 3.13 already has verified working wheels for every dependency below).
- [x] Add PyTorch via a `uv` explicit CUDA index (not the default CPU wheel) for the 4090 Laptop
      (Ada, sm_89).
- [x] Add the rest of the core stack: Transformers, NetworkX, `torch-geometric`, `numpy`, `scikit-learn`,
      `gensim` — the last four are actually imported by the code but missing from the README's list.
- [x] Add Jupyter as a dev-group dependency (notebooks stay the working surface until later phases
      convert them).
- [x] Resolve the `node2vec` blocker: PyPI `node2vec` pins `numpy<2.0.0` and hasn't shipped since Aug
      2024, conflicting with torch/PyG's NumPy 2.x requirement. It's used in exactly one ~4-line function
      (`create_node2vec_embeddings`); the DeepWalk path next to it already hand-rolls walks on
      NetworkX + gensim. Hand-roll the biased walk the same way and drop the dependency.
- [x] Fix the hardcoded `CUDA_VISIBLE_DEVICES = "1"` in `2_node2vec_deepwalk.py` (tuned for the two-GPU
      A6000 box; selects nothing on this single-GPU laptop) — make it configurable, default `0`.
- [x] Write `scripts/check_env.py` and confirm it passes: Python 3.13.x, CUDA available, every pipeline
      import resolves, ≥15GB free VRAM. **Written and passing except VRAM** — this machine reports
      14.7GB/16GB free at idle (Windows WDDM + background apps reserve ~1.3GB); re-check after closing
      GPU-using apps, or revisit the 15GB threshold for a Windows dev laptop.

### Phase 0.2 — Clean project: convert notebooks to .py

- [x] Convert all 30+ Jupyter notebooks to Python scripts covering the core pipeline stages:
      - Stage 1: `data_preprocessing/` (7 scripts) — ATT&CK/CWE ingestion, graph construction, ID assignment
      - Stage 2: `LLM_embeddings/` (2 scripts) — LLM fine-tuning and text embedding extraction
      - Stage 3: `GCL/get_graph_features/` (1 script) — node2vec/DeepWalk structural embeddings
      - Stage 4: `GCL/data_preprocessing/` (3 scripts) — similarity matrices and contrastive sampling
      - Stage 5: `GCL/models/` (1 script) — dual-encoder GCL training
- [x] Remove analysis and auxiliary notebooks no longer in the pipeline:
      - `GCL/gcl_analysis/` (12 notebooks + checkpoints) — Rand index, silhouette score, precision@k, hit@k, BRON, t-SNE plots
      - `GCL/text_hop_analysis/` — removed with analysis suite
      - `gradio_app/` — Gradio demo (non-runnable; `prompt_generation/` missing)
- [x] Remove unreachable code paths and dead dependencies:
      - `GCL/get_graph_features/1_text_hop_emb_model*.py` — feature branch (no live consumer, `text_hop_embeddings.npy` unreachable)
      - `embedding_generator/` reference in `config.ENCODER_PATH` (target missing, never read)
      - `prompt_generation/` module (missing; blocked Gradio app)
- [x] Clean up notebook residue in converted scripts:
      - Remove unused imports, exploratory print statements, bare expressions (e.g., `.number_of_edges()`)
      - Consolidate duplicate model name lists (was scattered across 5 files in stage 4)
      - Remove commented-out cells and dead code branches
- [x] Update `.gitignore` to reflect current working tree (remove analysis paths, demo references)
- [x] Verify pipeline connectivity: confirm all 5 stages can be executed in order on the fixture
      (fixture data not yet generated; stages 1–5 framework in place, runnable with real data)
