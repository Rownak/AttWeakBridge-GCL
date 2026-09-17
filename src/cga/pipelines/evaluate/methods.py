"""``MethodSpec`` + ``load_method``: resolve one comparison method's embeddings.

A *method* is any per-node ``AlignedMatrix`` -- a baseline embedding store or
a trained model's export -- so baselines and trained models are loaded by the
same code and compared on the same node universe (target_architecture.md
§5.5). A view's declared reference (e.g. a generic-text store) reuses this
same spec and loader, so there is exactly one traceability path for every
embedding this pipeline reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cga.core.artifacts import ArtifactRef
from cga.core.errors import ConfigError
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix

_KINDS = {"embedding_store", "gcl_run"}


@dataclass(frozen=True)
class MethodSpec:
    """One configured method (or view reference) to load and score.

    Attributes:
        method_id: Unique id within this run's ``methods:`` list (or the
            view-scoped id for a reference, e.g. a view's ``view_id``).
        label: Human-readable label for figures/tables.
        kind: ``embedding_store`` or ``gcl_run``.
        ref: ``latest`` or an explicit run_id within the resolved lineage dir.
        embedder_id: Required when ``kind == "embedding_store"``.
        model_id: Required when ``kind == "gcl_run"``.
    """

    method_id: str
    label: str
    kind: str
    ref: str
    embedder_id: str | None = None
    model_id: str | None = None

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> "MethodSpec":
        """Build and validate a ``MethodSpec`` from one ``methods:``/reference config entry.

        Raises:
            ConfigError: If ``kind`` is unknown or its required id field is missing.
        """
        method_id = entry["method_id"]
        kind = entry["kind"]
        if kind not in _KINDS:
            raise ConfigError(
                f"Unknown method kind '{kind}' for method_id '{method_id}'. "
                f"Known kinds: {sorted(_KINDS)}"
            )
        if kind == "embedding_store" and not entry.get("embedder_id"):
            raise ConfigError(f"method '{method_id}' has kind=embedding_store but no embedder_id")
        if kind == "gcl_run" and not entry.get("model_id"):
            raise ConfigError(f"method '{method_id}' has kind=gcl_run but no model_id")
        return cls(
            method_id=method_id,
            label=entry.get("label", method_id),
            kind=kind,
            ref=entry.get("ref", "latest"),
            embedder_id=entry.get("embedder_id"),
            model_id=entry.get("model_id"),
        )


def load_method(ctx: RunContext, spec: MethodSpec) -> tuple[AlignedMatrix, str]:
    """Resolve, record, and load ``spec``'s embeddings.

    Resolves through ``ctx.resolve`` under the manifest input name
    ``method:{method_id}``, so every method (and every view's reference) is
    recorded in ``manifest.inputs[]`` with its concrete resolved run_id --
    the per-row lineage each ``MetricRow.method_run_id`` carries.

    Args:
        ctx: The active run context.
        spec: The method to load.

    Returns:
        ``(matrix, resolved_run_id)``.

    Raises:
        ConfigError: If ``spec.kind`` is not recognized (should not happen for
            a spec built via :meth:`MethodSpec.from_config`).
    """
    dataset_id = ctx.dataset_id
    if spec.kind == "embedding_store":
        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "embeddings" / dataset_id / spec.embedder_id
        run_dir = ctx.resolve(f"method:{spec.method_id}", ArtifactRef(lineage_dir=lineage_dir, ref=spec.ref))
        matrix = AlignedMatrix.load(run_dir)
    elif spec.kind == "gcl_run":
        lineage_dir = Path(ctx.config["paths"]["models"]) / "gcl" / dataset_id / spec.model_id
        run_dir = ctx.resolve(f"method:{spec.method_id}", ArtifactRef(lineage_dir=lineage_dir, ref=spec.ref))
        matrix = AlignedMatrix.load(run_dir / "node_embeddings")
    else:  # pragma: no cover -- guarded by MethodSpec.from_config
        raise ConfigError(f"Unknown method kind '{spec.kind}'")
    return matrix, run_dir.name
