"""``cga`` command-line entry point (target_architecture.md §8.1).

```
cga list [pipelines|adapters|embedders|samplers]
cga run <pipeline> --dataset <id> [--experiment <name>] [--config f.yaml ...]
                   [--set key.path=value ...] [--run-name NAME] [--resume RUN_ID] [--dry-run]
cga sweep --spec <sweep_spec.yaml>
cga validate <artifact_dir>
cga inspect  <artifact_dir>
cga config show <pipeline> [...same layering flags]
```

Importing this module registers every built-in adapter and pipeline as a side
effect (``cga.adapters`` / ``cga.pipelines``), so ``cga list`` and ``cga run``
see the full built-in set with no other setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import cga.adapters  # noqa: F401  (registers adapters)
import cga.pipelines  # noqa: F401  (registers pipelines)
from cga import __version__
from cga.core.config import build_config, parse_set_value
from cga.core.errors import CgaError
from cga.core.io import read_json
from cga.core.registry import adapters, embedders, pipelines, samplers
from cga.core.run import RunContext

_CONFIG_ROOT = Path("configs")

_PIPELINE_OUTPUT_ROOT = {
    "ingest": "processed",
}


def _run_root(pipeline_name: str, config: dict[str, Any]) -> Path:
    """Return the parent directory ``{run_id}/`` is created under for this pipeline.

    Most pipelines write to ``{paths.<subdir>}/{dataset_id}``. ``embed`` nests
    one further level under the run's ``embedder_id`` (target_architecture.md
    §5.2), which depends on the resolved ``embed.embedder`` config block, so it
    cannot be expressed by the generic ``_PIPELINE_OUTPUT_ROOT`` lookup alone.
    ``training_data`` writes to
    ``artifacts/training_data/{dataset_id}/{sampler_id}`` (§5.3), partitioned
    by sampler configuration so a sweep over e.g. ``rank_topk.k`` keeps each
    value's ``LATEST`` pointer separate (mirrors ``gcl_train``'s
    ``model_id`` partitioning below).
    ``gcl_train`` writes to ``models/gcl/{dataset_id}/{model_id}`` (§5.4),
    partitioned by architecture so ``gcn``/``gat`` keep separate ``LATEST``
    pointers -- required for ``evaluate`` to compare both as ``latest``.
    ``evaluate`` writes to ``artifacts/evaluations/{dataset_id}`` (§5.5/§5.5a).
    ``plot`` writes to ``artifacts/plots/{dataset_id}`` (§5.6).
    """
    dataset_id = config["dataset"]["id"]
    if pipeline_name == "embed":
        from cga.pipelines.embed.pipeline import build_embedder_id

        embedder_id = build_embedder_id(config["embed"]["embedder"])
        return Path(config["paths"]["artifacts"]) / "embeddings" / dataset_id / embedder_id

    if pipeline_name == "training_data":
        from cga.pipelines.training_data.pipeline import build_sampler_id

        sampler_id = build_sampler_id(config["training_data"]["sampler"])
        return Path(config["paths"]["artifacts"]) / "training_data" / dataset_id / sampler_id

    if pipeline_name == "gcl_train":
        from cga.pipelines.gcl_train.pipeline import build_model_id

        model_id = build_model_id(config["gcl_train"]["model"])
        return Path(config["paths"]["models"]) / "gcl" / dataset_id / model_id

    if pipeline_name == "evaluate":
        return Path(config["paths"]["artifacts"]) / "evaluations" / dataset_id

    if pipeline_name == "plot":
        return Path(config["paths"]["artifacts"]) / "plots" / dataset_id

    output_subdir = _PIPELINE_OUTPUT_ROOT.get(pipeline_name, pipeline_name)
    return Path(config["paths"][output_subdir]) / dataset_id


def _dataset_config_path(dataset_id: str) -> Path:
    return _CONFIG_ROOT / "datasets" / f"{dataset_id}.yaml"


def _pipeline_config_path(pipeline_name: str) -> Path:
    return _CONFIG_ROOT / "pipelines" / f"{pipeline_name}.yaml"


def _experiment_config_path(experiment_name: str) -> Path:
    return _CONFIG_ROOT / "experiments" / f"{experiment_name}.yaml"


def _parse_overrides(pairs: Sequence[str] | None) -> dict[str, Any]:
    """Parse ``--set key.path=value`` strings into a dotted-path -> value dict."""
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise CgaError(f"--set expects key.path=value, got: {pair!r}")
        key, raw_value = pair.split("=", 1)
        overrides[key] = parse_set_value(raw_value)
    return overrides


def _resolve_config(args: argparse.Namespace, *, pipeline_name: str) -> dict[str, Any]:
    dataset_path = _dataset_config_path(args.dataset) if getattr(args, "dataset", None) else None
    pipeline_path = _pipeline_config_path(pipeline_name)
    experiment_path = (
        _experiment_config_path(args.experiment) if getattr(args, "experiment", None) else None
    )
    extra_paths = list(getattr(args, "config", None) or [])

    if dataset_path is not None and not dataset_path.is_file():
        raise CgaError(f"Unknown dataset '{args.dataset}': no such file {dataset_path}")
    if not pipeline_path.is_file():
        raise CgaError(f"Unknown pipeline '{pipeline_name}': no such file {pipeline_path}")

    overrides = _parse_overrides(getattr(args, "set", None))
    return build_config(
        base_path=_CONFIG_ROOT / "base.yaml",
        dataset_path=dataset_path,
        pipeline_path=pipeline_path,
        experiment_path=experiment_path,
        extra_config_paths=extra_paths,
        overrides=overrides,
    )


def cmd_list(args: argparse.Namespace) -> int:
    """``cga list [pipelines|adapters|embedders|samplers]``."""
    registries = {
        "pipelines": pipelines,
        "adapters": adapters,
        "embedders": embedders,
        "samplers": samplers,
    }
    kinds = [args.kind] if args.kind else list(registries)
    for kind in kinds:
        names = registries[kind].names()
        print(f"{kind}:")
        for name in names:
            print(f"  {name}")
        if not names:
            print("  (none registered)")
    return 0


def _run_pipeline(pipeline_name: str, config: dict[str, Any], *, sweep_point_id: str | None = None) -> RunContext:
    """Build a ``RunContext`` and run ``pipeline_name`` against ``config``.

    Shared by ``cga run`` and ``cga sweep`` so a swept combination goes
    through exactly the same ``build_config -> RunContext.create ->
    pipeline.run`` path as a standalone run (plan §2: "no change to
    ``core/config.py``"). ``sweep_point_id`` is stored on
    ``Manifest.sweep_point_id`` (a dedicated field, not a ``summary`` key)
    since every pipeline's manifest step assigns ``ctx.manifest.summary =
    {...}`` wholesale -- a plain dict key set here would be silently
    discarded.

    Raises:
        CgaError: Propagated from ``pipeline.run`` on failure; the caller
            decides whether that aborts (``cga run``) or is merely logged and
            skipped (``cga sweep``).
    """
    pipeline_cls = pipelines.get(pipeline_name)
    pipeline = pipeline_cls()
    run_root = _run_root(pipeline_name, config)

    ctx = RunContext.create(
        pipeline_name=pipeline_name,
        pipeline_version=__version__,
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=run_root,
    )
    if sweep_point_id is not None:
        ctx.manifest.sweep_point_id = sweep_point_id

    pipeline.run(ctx)
    return ctx


def cmd_run(args: argparse.Namespace) -> int:
    """``cga run <pipeline> --dataset <id> ...``."""
    pipeline_name = args.pipeline
    config = _resolve_config(args, pipeline_name=pipeline_name)

    if args.dry_run:
        print(f"Would run pipeline '{pipeline_name}' for dataset '{config['dataset']['id']}'")
        print(f"Resolved config keys: {sorted(config.keys())}")
        return 0

    try:
        ctx = _run_pipeline(pipeline_name, config)
    except CgaError as exc:
        print(f"Pipeline '{pipeline_name}' failed: {exc}", file=sys.stderr)
        return 1

    print(f"Run '{ctx.run_id}' succeeded: {ctx.run_dir}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """``cga sweep --spec <file>``: expand a sweep spec and run each combination.

    Builds each combination's config the normal ``--dataset``/pipeline-config
    layering way, then applies that combination's overrides last (same
    precedence ``--set`` would have). A combination that raises ``CgaError``
    is logged and skipped rather than aborting the whole sweep, per the plan's
    "continue past a failed combination" requirement -- the sweep's own exit
    code is non-zero if *any* combination failed, so CI still notices.
    """
    from cga.core.sweep import SweepSpec, expand_sweep

    spec = SweepSpec.from_file(args.spec)
    points = expand_sweep(spec)

    pipeline_path = _pipeline_config_path(spec.pipeline)
    dataset_path = _dataset_config_path(spec.dataset)
    if not pipeline_path.is_file():
        raise CgaError(f"Unknown pipeline '{spec.pipeline}': no such file {pipeline_path}")
    if not dataset_path.is_file():
        raise CgaError(f"Unknown dataset '{spec.dataset}': no such file {dataset_path}")

    n_failed = 0
    for point in points:
        config = build_config(
            base_path=_CONFIG_ROOT / "base.yaml",
            dataset_path=dataset_path,
            pipeline_path=pipeline_path,
            overrides=point.overrides,
        )
        print(f"[sweep] running point '{point.sweep_point_id}' ({point.overrides})")
        try:
            ctx = _run_pipeline(spec.pipeline, config, sweep_point_id=point.sweep_point_id)
        except CgaError as exc:
            n_failed += 1
            print(f"[sweep] point '{point.sweep_point_id}' FAILED: {exc}", file=sys.stderr)
            continue
        print(f"[sweep] point '{point.sweep_point_id}' succeeded: run '{ctx.run_id}' at {ctx.run_dir}")

    n_total = len(points)
    print(f"[sweep] {n_total - n_failed}/{n_total} combinations succeeded")
    return 1 if n_failed else 0


def cmd_validate(args: argparse.Namespace) -> int:
    """``cga validate <artifact_dir>``: schema + hash + alignment checks (read-side)."""
    from cga.data.graph_store import read_alignments, read_edges, read_nodes, read_texts

    artifact_dir = Path(args.artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"No manifest.json found in {artifact_dir}", file=sys.stderr)
        return 1

    manifest = read_json(manifest_path)
    if manifest.get("status") != "success":
        print(f"Run status is '{manifest.get('status')}', expected 'success'", file=sys.stderr)
        return 1

    checked = 0
    for table_name, reader in (
        ("nodes.parquet", read_nodes),
        ("edges.parquet", read_edges),
        ("texts.parquet", read_texts),
        ("alignments.parquet", read_alignments),
    ):
        path = artifact_dir / table_name
        if path.is_file():
            reader(path)
            checked += 1

    print(f"OK: {artifact_dir} ({checked} table(s) validated, manifest status=success)")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """``cga inspect <artifact_dir>``: print the manifest and walk the upstream chain."""
    artifact_dir = Path(args.artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"No manifest.json found in {artifact_dir}", file=sys.stderr)
        return 1

    visited: set[str] = set()

    def _walk(path: Path, depth: int) -> None:
        manifest = read_json(path)
        indent = "  " * depth
        print(
            f"{indent}{manifest.get('pipeline')} run={manifest.get('run_id')} "
            f"status={manifest.get('status')} dataset={manifest.get('dataset_id')}"
        )
        for inp in manifest.get("inputs", []):
            input_run_id = inp.get("run_id")
            print(f"{indent}  <- input '{inp.get('name')}' (run={input_run_id or 'raw'})")
            if not input_run_id or input_run_id in visited:
                continue
            visited.add(input_run_id)
            upstream_manifest = Path(inp["path"]) / "manifest.json"
            if upstream_manifest.is_file():
                _walk(upstream_manifest, depth + 1)

    _walk(manifest_path, 0)
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    """``cga config show <pipeline> [...layering flags]``."""
    config = _resolve_config(args, pipeline_name=args.pipeline)
    import json

    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


def _add_run_layering_args(parser: argparse.ArgumentParser, *, dataset_required: bool) -> None:
    parser.add_argument("--dataset", required=dataset_required, help="Dataset id, e.g. attack_cwe")
    parser.add_argument("--experiment", default=None, help="Experiment overlay name")
    parser.add_argument("--config", action="append", default=None, help="Extra config file (repeatable)")
    parser.add_argument("--set", action="append", default=None, help="key.path=value override (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``cga`` argument parser."""
    parser = argparse.ArgumentParser(prog="cga", description="Cross-graph alignment pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered components")
    list_parser.add_argument(
        "kind", nargs="?", choices=["pipelines", "adapters", "embedders", "samplers"], default=None
    )
    list_parser.set_defaults(func=cmd_list)

    run_parser = subparsers.add_parser("run", help="Run a pipeline")
    run_parser.add_argument("pipeline", help="Registered pipeline name, e.g. ingest")
    _add_run_layering_args(run_parser, dataset_required=True)
    run_parser.add_argument("--run-name", default=None, help="Optional human alias for this run")
    run_parser.add_argument("--resume", default=None, help="Run id to resume (pipeline-specific)")
    run_parser.add_argument("--dry-run", action="store_true", help="Resolve config and exit")
    run_parser.set_defaults(func=cmd_run)

    sweep_parser = subparsers.add_parser("sweep", help="Expand a sweep spec and run each combination")
    sweep_parser.add_argument("--spec", required=True, help="Path to a sweep spec YAML file")
    sweep_parser.set_defaults(func=cmd_sweep)

    validate_parser = subparsers.add_parser("validate", help="Validate an artifact directory")
    validate_parser.add_argument("artifact_dir")
    validate_parser.set_defaults(func=cmd_validate)

    inspect_parser = subparsers.add_parser("inspect", help="Print a run's manifest chain")
    inspect_parser.add_argument("artifact_dir")
    inspect_parser.set_defaults(func=cmd_inspect)

    config_parser = subparsers.add_parser("config", help="Config utilities")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_show_parser = config_subparsers.add_parser("show", help="Print the fully-resolved config")
    config_show_parser.add_argument("pipeline", help="Registered pipeline name")
    _add_run_layering_args(config_show_parser, dataset_required=False)
    config_show_parser.set_defaults(func=cmd_config_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, dispatch, catch ``CgaError`` and exit non-zero."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CgaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
