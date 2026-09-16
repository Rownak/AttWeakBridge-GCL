"""Run manifest: the record of what a pipeline run consumed, produced, and how.

One ``manifest.json`` per run directory (see target_architecture.md §3.6). The
chain of ``inputs[].run_id`` links across manifests is what lets ``cga inspect``
walk a result back to the raw file hashes that produced it.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cga.core.io import sha256_file, write_json

MANIFEST_VERSION = 1


def _git_info() -> dict[str, Any]:
    """Best-effort current git commit and dirty state; never raises."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


@dataclass
class Manifest:
    """Accumulates a single run's inputs, outputs, and metadata, then writes itself.

    Attributes:
        pipeline: Registered pipeline name (e.g. "embed").
        pipeline_version: Package version at run time.
        run_id: This run's generated id.
        dataset_id: Dataset pair this run operates on.
        seed: RNG seed used for the run.
    """

    pipeline: str
    pipeline_version: str
    run_id: str
    dataset_id: str
    seed: int
    started_at: str
    config_hash: str
    config_resolved_path: str
    status: str = "running"
    finished_at: str | None = None
    duration_s: float | None = None
    env: dict[str, Any] = field(default_factory=dict)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.env.setdefault("python", platform.python_version())
        self.env.update(_git_info())

    def record_input(self, name: str, *, pipeline: str, run_id: str, path: str | Path) -> None:
        """Record an upstream artifact this run reads, hashing it now.

        This is the single place an input is declared; a pipeline that reads a
        file without calling this cannot have that read reflected in the
        manifest's traceability chain.
        """
        path = Path(path)
        self.inputs.append(
            {
                "name": name,
                "pipeline": pipeline,
                "run_id": run_id,
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )

    def record_output(self, name: str, path: str | Path, **extra: Any) -> None:
        """Record an artifact this run produced, hashing it now.

        Args:
            name: Logical name of the output (e.g. "vectors").
            path: Filesystem path to the produced file.
            **extra: Additional fields to attach (e.g. ``shape``, ``dtype``).
        """
        path = Path(path)
        entry: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        entry.update(extra)
        self.outputs.append(entry)

    def mark_success(self, finished_at: str, duration_s: float) -> None:
        """Mark the run as successfully completed."""
        self.status = "success"
        self.finished_at = finished_at
        self.duration_s = duration_s

    def mark_failed(self, finished_at: str, duration_s: float) -> None:
        """Mark the run as failed."""
        self.status = "failed"
        self.finished_at = finished_at
        self.duration_s = duration_s

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable manifest document."""
        return {
            "manifest_version": MANIFEST_VERSION,
            "pipeline": self.pipeline,
            "pipeline_version": self.pipeline_version,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "config_resolved_path": self.config_resolved_path,
            "code": {
                "git_commit": self.env.get("git_commit"),
                "git_dirty": self.env.get("git_dirty"),
                "package_version": self.pipeline_version,
            },
            "env": {k: v for k, v in self.env.items() if k not in ("git_commit", "git_dirty")},
            "inputs": self.inputs,
            "outputs": self.outputs,
            "summary": self.summary,
        }

    def write(self, run_dir: str | Path) -> Path:
        """Write ``manifest.json`` into ``run_dir`` and return its path."""
        path = Path(run_dir) / "manifest.json"
        write_json(self.to_dict(), path)
        return path
