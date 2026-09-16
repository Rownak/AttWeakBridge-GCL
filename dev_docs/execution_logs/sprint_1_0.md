# Sprint 1.0 Execution Log — Move legacy code and data aside

## Summary
Moved all legacy code and configuration into a frozen `legacy/` directory, isolated from the new architecture. Created the new directory structure (`src/cga/`, `configs/`, `tests/`, `data/raw/`) and copied raw data files to their designated locations.

## Implementation

**Git-tracked moves (git mv):**
- `config.py` → `legacy/config.py`
- `data_preprocessing/` → `legacy/data_preprocessing/`
- `LLM_embeddings/` → `legacy/LLM_embeddings/`
- `GCL/` → `legacy/GCL/`
- `run.sh` → `legacy/run.sh`

**Filesystem moves (data directories, git-ignored):**
- `datasets/` → `legacy/datasets/` (empty directory; data not checked in)
- `model_outputs/` → `legacy/model_outputs/` (empty directory; data not checked in)

**New files created:**
- `legacy/README.md` — marked the directory as frozen reference code
- Updated `.gitignore`:
  - Added `legacy/datasets/` and `legacy/model_outputs/` to ignore paths
  - Added `data/`, `artifacts/`, `models/` as new output directories to ignore

**New directory structure:**
```
src/cga/              # new package root
configs/              # configuration files
tests/unit/           # unit tests
tests/integration/    # integration tests
data/raw/             # raw input data
  attack_cwe/         # (placeholder, awaiting real files)
  ncit_doid/          # (populated from benchmark/OAEI-Bio-ML/ncit-doid)
artifacts/            # pipeline outputs (git-ignored)
models/               # trained models (git-ignored)
```

**Data files copied:**
- `benchmark/OAEI-Bio-ML/ncit-doid/*` → `data/raw/ncit_doid/`
  - `ncit.owl`, `doid.owl`, `refs_equiv/*.tsv`, `refs_subs/*.tsv`

## Verification

✅ **Legacy isolation:** `python -c "import config"` fails from repo root (no module found)
✅ **Directory structure:** All new directories created
✅ **Data integrity:** ncit_doid files present and readable
✅ **Git state:** 70+ legacy files renamed/moved, .gitignore updated, ready to stage

## Files Modified
- `.gitignore`
- `legacy/README.md` (new)

## Files Created
- Directory structure: `src/cga/`, `configs/`, `tests/unit/`, `tests/integration/`, `data/raw/`, `artifacts/`, `models/`

## Notes
- `attack_cwe` raw data files (attack.json, cwec_v4.12.xml) are not yet present in the project; `data/raw/attack_cwe/` exists as an empty placeholder
- All legacy code moved in Phase 1.0; regression testing against legacy outputs deferred to separate sprint (per spec)
- Benchmark directory remains at root (git-ignored per original .gitignore)

## Next Phase
Phase 1.1: Package skeleton and `core/` modules — configure pyproject.toml, create core utilities (errors, registry, config, I/O, logging, manifest, artifacts, run context, pipeline base).
