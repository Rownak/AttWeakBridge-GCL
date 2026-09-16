"""Artifact reference resolution: ``latest`` / explicit run_id / literal path.

Each lineage directory (e.g. ``artifacts/embeddings/attack_cwe/hf_mean_pool/``)
holds a ``LATEST`` file naming its most recently *successful* run. Downstream
configs reference upstream artifacts by ``latest`` or an explicit run_id;
resolution happens once at run start, and the concrete run_id it resolved to is
what the manifest records (see target_architecture.md §3.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cga.core.errors import MissingInputError

LATEST_FILENAME = "LATEST"


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to an artifact run directory, before or after resolution.

    Attributes:
        lineage_dir: The directory containing one run_id subdirectory per run
            (e.g. ``artifacts/embeddings/attack_cwe/hf_mean_pool``).
        ref: ``"latest"``, an explicit run_id, or an absolute/relative path to a
            run directory.
    """

    lineage_dir: str | Path
    ref: str

    def resolve(self) -> Path:
        """Resolve this reference to a concrete, existing run directory.

        Raises:
            MissingInputError: If the resolved directory does not exist, or a
                ``LATEST`` file is missing/points at a non-existent directory.
        """
        lineage_dir = Path(self.lineage_dir)

        if self.ref == "latest":
            latest_file = lineage_dir / LATEST_FILENAME
            if not latest_file.is_file():
                raise MissingInputError(
                    f"No '{LATEST_FILENAME}' file found in {lineage_dir}; "
                    "has any run in this lineage succeeded yet?"
                )
            run_id = latest_file.read_text(encoding="utf-8").strip()
            run_dir = lineage_dir / run_id
        elif Path(self.ref).is_absolute() or Path(self.ref).exists():
            run_dir = Path(self.ref)
        else:
            run_dir = lineage_dir / self.ref

        if not run_dir.is_dir():
            raise MissingInputError(f"Artifact run directory not found: {run_dir}")
        return run_dir

    def resolved_run_id(self) -> str:
        """Return the concrete run_id this reference resolves to."""
        return self.resolve().name


def write_latest(lineage_dir: str | Path, run_id: str) -> None:
    """Write/overwrite the ``LATEST`` file in ``lineage_dir`` to point at ``run_id``.

    Only call this after a run has been marked successful — ``LATEST`` must
    never point at a failed or partial run.
    """
    lineage_dir = Path(lineage_dir)
    lineage_dir.mkdir(parents=True, exist_ok=True)
    (lineage_dir / LATEST_FILENAME).write_text(run_id, encoding="utf-8")


def read_latest(lineage_dir: str | Path) -> str | None:
    """Return the run_id in ``lineage_dir``'s ``LATEST`` file, or None if absent."""
    latest_file = Path(lineage_dir) / LATEST_FILENAME
    if not latest_file.is_file():
        return None
    return latest_file.read_text(encoding="utf-8").strip()
