# Sprint 1.1 Execution Log — Package skeleton and `core/`

## Summary
Built the `src/cga` package skeleton and the eight `core/` modules the rest of the pipeline depends on: errors, registry, config layering, I/O, seeding, logging, manifests, artifact resolution, run context, and the pipeline base class.

## Implementation

**`pyproject.toml`:**
- Switched to `src` layout (`[tool.hatch.build.targets.wheel] packages = ["src/cga"]`), added `hatchling` build backend
- Added `[project.scripts] cga = "cga.cli:main"` (cli.py itself is a Phase 1.3 deliverable — `uv run cga` will fail until then; `uv sync`, imports, and tests are unaffected)
- Added `pyarrow`, `pydantic`, `pyyaml` to dependencies; `pytest` to the dev group
- Created a minimal root `README.md` (required by hatchling's readme validation) — full install/usage docs are a Phase 1.7 deliverable

**`src/cga/core/`:**
- `errors.py` — `CgaError` base, `ConfigError`, `MissingInputError`, `ArtifactValidationError`
- `registry.py` — generic `Registry[T]` with `register(name)` decorator, `get`, `names`; module-level instances `pipelines`, `adapters`, `embedders`, `samplers`, `losses`
- `config.py` — `deep_merge`, `load_yaml`, `set_by_dotted_path`, `interpolate` (`${paths.x}` / `${env:VAR}`), `config_hash` (sha256 of canonical JSON), `build_config` orchestrating the full base→dataset→pipeline→experiment→`--config`→`--set` layering order
- `io.py` — `read_parquet`/`write_parquet` (schema cast + validation via pyarrow), `read_npy`/`write_npy`, `read_json`/`write_json`, `sha256_file`
- `seeds.py` — `seed_everything` covering `random`, `numpy`, `torch` (CPU + CUDA, imported lazily so torch isn't a hard dependency of this module), `PYTHONHASHSEED`
- `logging.py` — `setup_logging` (console + optional per-run file handler under `{run_dir}/run.log`), `get_logger` for `cga.<name>` child loggers
- `manifest.py` — `Manifest` dataclass matching the §3.6 schema exactly, `record_input`/`record_output` (hashing files as they're recorded), `mark_success`/`mark_failed`, `write()`; best-effort git commit/dirty detection that never raises
- `artifacts.py` — `ArtifactRef` dataclass resolving `"latest"` (via `LATEST` file), explicit run_id, or literal path; `write_latest`/`read_latest` helpers
- `run.py` — `RunContext.create()` (generates `run_id` per §3.5, creates the run directory, seeds RNGs, wires logging and manifest), `resolve()` as the single choke point that turns an `ArtifactRef` into a path *and* records it as a manifest input in one call, `resolve_device` for `auto`/`cpu`/`cuda:0`
- `pipeline.py` — `Step` protocol, `Pipeline` ABC with `build_steps()` abstract method and a `run()` that executes steps in order, writes the manifest, and marks success/failure (catching exceptions to write a failed manifest before re-raising)

**`configs/base.yaml`** — matches §7.2 exactly (project, seed, device, paths, logging, run.name).

## Tests
`tests/unit/core/`:
- `test_config.py` (10 tests) — deep-merge scalar/nested/list semantics, full layering order and `--set` precedence, dotted-path creation, `${paths.*}`/`${env:*}` interpolation (success and missing-reference failure), `config_hash` stability under key reordering and sensitivity to content changes
- `test_registry.py` (4 tests) — register/get roundtrip, duplicate-name `ConfigError`, unknown-name error listing known names, sorted `names()`
- `test_seeds.py` (2 tests) — reproducibility of `random`/`numpy` sequences under a fixed seed, divergence under different seeds
- `test_artifacts.py` (6 tests) — `latest` resolution via `LATEST` file, explicit run_id resolution, missing-`LATEST`/missing-run-dir `MissingInputError`, `write_latest`/`read_latest` roundtrip and missing-file `None`

All 23 tests pass: `uv run pytest tests/unit -q`.

## Verification
- `uv sync` succeeds (after adding the missing root `README.md` hatchling requires)
- All `core/` modules import cleanly together
- `uv run pytest tests/unit -q` → 23 passed

## Files Created
- `pyproject.toml` (modified), `README.md` (new placeholder)
- `src/cga/__init__.py`, `src/cga/core/__init__.py`
- `src/cga/core/{errors,registry,config,io,seeds,logging,manifest,artifacts,run,pipeline}.py`
- `configs/base.yaml`
- `tests/unit/core/{test_config,test_registry,test_seeds,test_artifacts}.py`

## Notes / assumptions
- `core/io.py` schema validation for `write_parquet` attempts a `cast()` to the declared schema before raising, so a schema-compatible table with a slightly different declared dtype ordering doesn't spuriously fail — full FK/pk enforcement is a Phase 1.2 (`data/graph_store.py`) concern per spec, not `core/io.py`.
- `Manifest`'s git-info lookup shells out to `git`; wrapped in try/except so it degrades to `None`/`None` outside a git repo or without git installed, rather than failing the run.
- `cli.py` does not exist yet (Phase 1.3), so the `[project.scripts]` entry point is registered but not runnable — this is expected and doesn't block `uv sync` or `pytest`.
- No `tests/unit/data/`, adapter, or pipeline tests yet — those belong to Phases 1.2+.

## CLI to verify
```
uv sync
uv run pytest tests/unit -q
```

## Next Phase
Phase 1.2: Canonical data layer — `data/ids.py`, `data/schemas.py`, `data/graph_store.py`, `data/aligned.py`, `data/embedding_store.py`, `data/example_store.py`.
