# Sprint 0 — Execution Log

## Phase 0.1 — uv dependency setup

### What was implemented

- Initialized a `uv`-managed project (`pyproject.toml`, `uv.lock`, `.python-version`) pinned to
  Python 3.13, replacing the stale "3.8+" floor in the README.
- Added PyTorch via an explicit `uv` index (`https://download.pytorch.org/whl/cu124`) instead of
  the default CPU wheel, so `uv sync` resolves a CUDA build for the RTX 4090 Laptop (Ada, sm_89).
- Added the rest of the core stack as top-level dependencies: `transformers`, `networkx`,
  `torch-geometric`, `numpy`, `scikit-learn`, `gensim`.
- Added `jupyter` as a `dev` dependency group (notebooks remain the working surface for this sprint).
- Removed the `node2vec` PyPI dependency (pinned `numpy<2.0`, unmaintained since Aug 2024, conflicts
  with torch/PyG's numpy 2.x requirement) from
  [GCL/get_graph_features/2_node2vec_deepwalk.py](GCL/get_graph_features/2_node2vec_deepwalk.py) and
  replaced `create_node2vec_embeddings` with a hand-rolled 2nd-order biased random walk
  (Grover & Leskovec p/q biasing) + `gensim.Word2Vec`, following the same pattern already used by
  the neighboring DeepWalk implementation.
- Fixed the hardcoded `CUDA_VISIBLE_DEVICES = "1"` in the same file (tuned for a two-GPU A6000 box) —
  now defaults to `"0"` via `os.environ.setdefault`, and is overridable by setting the env var before
  running the script.
- Wrote [scripts/check_env.py](scripts/check_env.py): checks Python version, CUDA availability, every
  pipeline import (`torch`, `transformers`, `networkx`, `torch_geometric`, `numpy`, `sklearn`,
  `gensim`), and free VRAM (≥15GB threshold per spec), printing a pass/fail summary and exiting
  non-zero on any failure.
- Updated `.gitignore` to exclude `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`.

### Approach / key decisions

- Used `uv init --no-workspace --python 3.13` then removed the stub `main.py` it generates — this
  project has no single entry-point script, so a stub wasn't useful.
- Chose the `cu124` PyTorch index over `cu126`/nightly: driver reports CUDA 13.2 support (backward
  compatible), and `cu124` is PyTorch's current stable CUDA build with solid Windows wheel coverage.
  Resolved to `torch==2.6.0+cu124`.
- For the node2vec replacement, kept the public function signature
  (`create_node2vec_embeddings(G, dimensions, walk_length, num_walks, window, min_count, batch_words,
  embeddings_dir)`) unchanged except for added optional `p`/`q` keyword args (default `1.0`, i.e.
  unbiased — matches DeepWalk behavior when unspecified), so the call site in `main()` needed no
  changes.
- Verified the biased-walk + Word2Vec pipeline end-to-end against `networkx.karate_club_graph()`
  (see CLI commands below) since no real dataset fixture exists yet (that's Phase 0.2).

### Files modified or created

- `pyproject.toml` (new)
- `uv.lock` (new)
- `.python-version` (new)
- `scripts/check_env.py` (new)
- `GCL/get_graph_features/2_node2vec_deepwalk.py` (modified — removed `node2vec` import, added
  biased-walk functions, fixed CUDA device default)
- `.gitignore` (modified — added `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`)

### Challenges / assumptions / limitations

- **VRAM check fails on this Windows laptop at idle**: `check_env.py` reports 14.7GB free / 16.0GB
  total (need ≥15GB per spec) because Windows WDDM display driver + background processes (browser,
  IDE) reserve GPU memory even with no ML workload running. This is expected on a Windows desktop
  session, not a bug in the check — the script correctly surfaces real headroom. Actual pipeline runs
  in later phases should re-check VRAM after closing unnecessary GPU-using applications; the
  threshold itself is left as specified rather than silently loosened.
- No real dataset exists yet (`datasets/` is empty and DVC-tracked per the sprint's later phases), so
  `2_node2vec_deepwalk.py` and other pipeline scripts could not be run end-to-end against real data
  in this phase — only syntax-checked (`ast.parse`) and, for the node2vec replacement, smoke-tested
  against a synthetic graph.
- `torch-geometric` resolved to `2.8.0.post1` and `transformers` to `5.17.0` — both significantly
  newer than what the original A6000 experiments likely used (no lockfile/requirements.txt existed
  to pin against). Flagging in case any pipeline script hits an API break in later phases; none was
  observed via syntax/smoke checks in this phase.

### CLI commands to test / run this phase

```bash
# Install/sync the locked environment (includes dev group: jupyter)
uv sync

# Run the environment sanity check
uv run python scripts/check_env.py

# Verify CUDA is visible to torch
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Phase 0.x — Existing-architecture documentation

### What was implemented

Rewrote [dev_docs/existing_architecture.md](dev_docs/existing_architecture.md) from a
README-style run guide into a factual as-is architecture reference for the legacy pipeline, to be
used as the baseline input for refactor planning.

### Approach / key decisions

- Read every `.py` and `.sh` file in the working tree rather than relying on the prior document or
  the git index — the index still lists the pre-migration `.ipynb` files, so `git ls-files` does not
  reflect the current code.
- Verified the claimed artifacts against what is actually on disk, loading each `.npy`/`.pkl` to
  record real shapes and counts (1136 nodes = 203 ATT&CK + 933 CWE, 2078 edges, 34,080 / 43,168
  sample tuples) instead of restating the numbers in the old document.
- Kept strictly to describing current behavior: no fixes, refactor proposals, or production
  recommendations. Where the code did not determine a component's purpose, it is marked
  **[UNCERTAIN]** rather than guessed — applied to the `datasets/*_1`, `*_2` snapshot directories,
  the orphaned hop-similarity artifacts, and the 1181-row `SecRoBERTa` embedding array.
- Documented the gap between what the older document described and what exists: `prompt_generation/`,
  `embedding_generator/`, `GCL/gcl_analysis/`, `GCL/text_hop_analysis/`, `scripts/check_env.py`,
  `README.md`, and `1_text_hop_emb_model*.py` are all referenced somewhere but absent from the
  working tree.
- Added a closing section on cross-cutting properties (no orchestration, positional array coupling,
  unseeded randomness, no validation or tests, dead paths, no run provenance) as neutral
  observations, since these are the facts a refactor plan needs.

### Files modified or created

- `dev_docs/existing_architecture.md` (rewritten)
- `dev_docs/execution_logs/sprint0.md` (this section appended)

### Challenges / assumptions / limitations

- **The pipeline was not re-executed for this task** — the document reflects code reading plus
  inspection of existing on-disk artifacts. Stated behavior is derived from source, and stated
  shapes/counts from the artifacts a previous run left behind.
- The on-disk state reflects a *partial* sweep: only one of the four stage-5 combinations configured
  in `run_dual_loss.sh` has an output. Artifact counts in the document describe that partial state.
- Two artifacts are mutually inconsistent (`embeddings/SecRoBERTa/text_embeddings.npy` has 1181 rows
  against the current 1136-node mapping). The code carries no provenance metadata, so which run
  produced it could not be established; recorded as uncertain.
- Stage-1 outputs were not re-derived from the source `attack.json` / CWE XML, so the document does
  not verify that the current `datasets/ics_attack/` artifacts are reproducible from the current
  inputs.

### CLI commands to test / run this phase

Documentation only — nothing to execute. To re-verify the recorded artifact shapes:

```bash
uv run python -c "
import numpy as np, json
print(json.load(open('datasets/ics_attack/attack_weak_range.json')))
print(len(json.load(open('datasets/ics_attack/graph_edges.json'))))
for p in ['model_outputs/ics_attack/embeddings/pt_SecRoBERTa/text_embeddings.npy',
          'model_outputs/ics_attack/embeddings/deepwalk.npy',
          'model_outputs/ics_attack/gcl_data/pt_SecRoBERTa/sample_10/GCN/triplet/text_deepwalk_dual3_gm_1.0.npy']:
    print(p, np.load(p).shape)
"
```
