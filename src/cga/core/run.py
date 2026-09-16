"""RunContext: the per-invocation object threaded through every pipeline step.

Created once by the CLI and passed to every step. Bundles the generated
``run_id``, the immutable resolved config, seed, device, a logger writing to
console + ``{run_dir}/run.log``, and a manifest accumulator. ``resolve()`` is the
single choke point through which a pipeline reads an upstream artifact: it turns
an ``ArtifactRef`` into a concrete path *and* records it as a manifest input in
one call, so a pipeline cannot read something it failed to declare.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cga.core.artifacts import ArtifactRef
from cga.core.config import config_hash as compute_config_hash
from cga.core.logging import setup_logging
from cga.core.manifest import Manifest


def generate_run_id(resolved_config: dict[str, Any]) -> str:
    """Return ``{YYYYMMDD}T{HHMMSS}Z-{config_hash[:8]}``, per target_architecture §3.5."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_hash = compute_config_hash(resolved_config)[:8]
    return f"{timestamp}-{short_hash}"


@dataclass
class RunContext:
    """Everything a pipeline step needs, bound to one run.

    Attributes:
        run_id: This run's generated id.
        run_dir: The immutable directory this run writes into exclusively.
        dataset_id: Dataset pair this run operates on.
        config: Fully-resolved, validated config for this run.
        seed: RNG seed used for the run.
        device: Resolved device string (e.g. "cpu", "cuda:0").
        logger: Logger writing to console and ``{run_dir}/run.log``.
        manifest: Manifest accumulator for this run.
    """

    run_id: str
    run_dir: Path
    dataset_id: str
    config: dict[str, Any]
    seed: int
    device: str
    logger: logging.Logger
    manifest: Manifest

    @classmethod
    def create(
        cls,
        *,
        pipeline_name: str,
        pipeline_version: str,
        dataset_id: str,
        config: dict[str, Any],
        run_root: str | Path,
    ) -> "RunContext":
        """Build a fresh RunContext: generates run_id, creates run_dir, seeds RNGs.

        Args:
            pipeline_name: Registered pipeline name (e.g. "ingest").
            pipeline_version: Package version to record in the manifest.
            dataset_id: Dataset pair this run operates on.
            config: Fully-resolved config for this run.
            run_root: Parent directory under which ``{run_id}/`` is created.
        """
        run_id = generate_run_id(config)
        run_dir = Path(run_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        seed = config.get("seed", 42)
        device = resolve_device(config.get("device", "auto"))

        log_cfg = config.get("logging", {})
        logger = setup_logging(
            level=log_cfg.get("level", "INFO"),
            run_dir=run_dir,
            to_file=log_cfg.get("file", True),
        )

        manifest = Manifest(
            pipeline=pipeline_name,
            pipeline_version=pipeline_version,
            run_id=run_id,
            dataset_id=dataset_id,
            seed=seed,
            started_at=datetime.now(timezone.utc).isoformat(),
            config_hash=compute_config_hash(config),
            config_resolved_path="config.resolved.yaml",
        )

        return cls(
            run_id=run_id,
            run_dir=run_dir,
            dataset_id=dataset_id,
            config=config,
            seed=seed,
            device=device,
            logger=logger,
            manifest=manifest,
        )

    def resolve(self, name: str, ref: ArtifactRef) -> Path:
        """Resolve ``ref`` to a concrete directory and record it as a manifest input.

        Args:
            name: Logical name for this input (e.g. "texts", "vectors").
            ref: The artifact reference to resolve.

        Returns:
            The resolved, existing run directory.
        """
        run_dir = ref.resolve()
        self.manifest.record_input(
            name,
            pipeline=Path(ref.lineage_dir).name,
            run_id=run_dir.name,
            path=run_dir,
        )
        return run_dir


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda:0"`` if available, else ``"cpu"``; pass through otherwise."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
