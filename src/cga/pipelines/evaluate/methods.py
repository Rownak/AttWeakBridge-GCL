"""``MethodSpec`` + ``load_method``: resolve one comparison method's embeddings.

A *method* is any per-node ``AlignedMatrix`` -- a baseline embedding store or
a trained model's export -- so baselines and trained models are loaded by the
same code and compared on the same node universe (target_architecture.md
§5.5). A view's declared reference (e.g. a generic-text store) reuses this
same spec and loader, so there is exactly one traceability path for every
embedding this pipeline reads.

A ``fusion`` method combines several members at the score level (Sprint 6.1):
each member's vectors are L2-normalised, scaled by ``sqrt(weight)`` and
concatenated, so a dot product of two fused vectors is the weighted sum of the
members' cosines and the cosine is their weighted mean. No training is
involved; the fused matrix is scored by the unchanged evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cga.core.artifacts import ArtifactRef
from cga.core.errors import ConfigError
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix

_KINDS = {"embedding_store", "gcl_run", "fusion"}
_MEMBER_KINDS = {"embedding_store", "gcl_run"}


@dataclass(frozen=True)
class MethodSpec:
    """One configured method (or view reference) to load and score.

    Attributes:
        method_id: Unique id within this run's ``methods:`` list (or the
            view-scoped id for a reference, e.g. a view's ``view_id``).
        label: Human-readable label for figures/tables.
        kind: ``embedding_store``, ``gcl_run`` or ``fusion``.
        ref: ``latest`` or an explicit run_id within the resolved lineage dir.
        embedder_id: Required when ``kind == "embedding_store"``.
        model_id: Required when ``kind == "gcl_run"``.
        members: Required when ``kind == "fusion"``: ``(MethodSpec, weight)`` pairs
            with positive weights (zero-weight members are already dropped).
    """

    method_id: str
    label: str
    kind: str
    ref: str
    embedder_id: str | None = None
    model_id: str | None = None
    members: tuple[tuple["MethodSpec", float], ...] = field(default_factory=tuple)

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> "MethodSpec":
        """Build and validate a ``MethodSpec`` from one ``methods:``/reference config entry.

        Raises:
            ConfigError: If ``kind`` is unknown or its required id field is missing,
                or a fusion has no members, a member of kind ``fusion``, a negative
                weight, or only zero weights.
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
        members: list[tuple[MethodSpec, float]] = []
        if kind == "fusion":
            raw = entry.get("members") or []
            if not raw:
                raise ConfigError(f"method '{method_id}' has kind=fusion but no members")
            for i, m in enumerate(raw):
                if m.get("kind") not in _MEMBER_KINDS:
                    raise ConfigError(
                        f"fusion '{method_id}' member {i} must have kind in {sorted(_MEMBER_KINDS)}, got {m.get('kind')!r}"
                    )
                weight = float(m.get("weight", 1.0))
                if weight < 0:
                    raise ConfigError(f"fusion '{method_id}' member {i} has negative weight {weight}")
                if weight > 0:
                    sub = cls.from_config({**m, "method_id": f"{method_id}:m{i}"})
                    members.append((sub, weight))
            if not members:
                raise ConfigError(f"fusion '{method_id}' has only zero weights")
        return cls(
            method_id=method_id,
            label=entry.get("label", method_id),
            kind=kind,
            ref=entry.get("ref", "latest"),
            embedder_id=entry.get("embedder_id"),
            model_id=entry.get("model_id"),
            members=tuple(members),
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
    elif spec.kind == "fusion":
        loaded = [(*load_method(ctx, sub), w) for sub, w in spec.members]
        matrix = fuse_matrices([(m, w) for m, _, w in loaded])
        return matrix, "+".join(run_id for _, run_id, _ in loaded)
    else:  # pragma: no cover -- guarded by MethodSpec.from_config
        raise ConfigError(f"Unknown method kind '{spec.kind}'")
    return matrix, run_dir.name


def fuse_matrices(members: list[tuple[AlignedMatrix, float]]) -> AlignedMatrix:
    """Fuse member embeddings so the fused cosine is the weighted mean of member cosines.

    Each member's rows are L2-normalised and scaled by ``sqrt(weight)`` before
    concatenation, so ``z_a . z_d = sum_i w_i cos_i(a, d)`` and ``||z||^2 = sum_i w_i``.

    Args:
        members: ``(matrix, weight)`` pairs with positive weights.

    Returns:
        An in-memory ``AlignedMatrix`` over the nodes every member shares, in the
        first member's order.

    Raises:
        ConfigError: If ``members`` is empty, a weight is not positive, or the members
            share no node.
    """
    if not members or any(w <= 0 for _, w in members):
        raise ConfigError("fusion needs at least one member and all weights must be positive")
    shared = set(members[0][0].node_ids)
    for matrix, _ in members[1:]:
        shared &= set(matrix.node_ids)
    if not shared:
        raise ConfigError("fusion members share no nodes (empty intersection)")
    ids = [n for n in members[0][0].node_ids if n in shared]
    blocks = []
    for matrix, weight in members:
        v = matrix.select(ids).vectors.astype(np.float32)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        blocks.append(np.sqrt(weight) * v / np.maximum(norm, 1e-12))
    return AlignedMatrix(vectors=np.concatenate(blocks, axis=1).astype(np.float32), node_ids=ids)
