"""``AlignedMatrix`` (vectors + id index) and ``NodeUniverse`` (ordered id list).

An ``.npy`` array is never written or read without its ``index.parquet``
sidecar naming which row belongs to which node_id. ``NodeUniverse`` is the
in-process replacement for the legacy ``attack_weak_range.json``: every array a
training run touches is ``select()``-ed onto the same universe *by id* before a
tensor exists, so two stores with different row counts can no longer silently
mis-align.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from cga.core.errors import ArtifactValidationError
from cga.core.io import read_json, read_npy, read_parquet, sha256_file, write_json, write_npy, write_parquet
from cga.data.schemas import INDEX_SCHEMA, UNIVERSE_SCHEMA


@dataclass
class AlignedMatrix:
    """A dense ``(n, d)`` float32 array paired 1:1 with an ordered list of node ids.

    Attributes:
        vectors: The ``(n, d)`` float32 array.
        node_ids: ``node_ids[i]`` is the id for ``vectors[i]``; length ``n``, unique.
    """

    vectors: np.ndarray
    node_ids: list[str]

    def __post_init__(self) -> None:
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ArtifactValidationError("AlignedMatrix node_ids must be unique")
        if self.vectors.shape[0] != len(self.node_ids):
            raise ArtifactValidationError(
                f"AlignedMatrix row count {self.vectors.shape[0]} != len(node_ids) {len(self.node_ids)}"
            )
        self._row_of = {node_id: i for i, node_id in enumerate(self.node_ids)}

    @classmethod
    def load(cls, directory: str | Path) -> "AlignedMatrix":
        """Load and validate an AlignedMatrix directory (vectors.npy + index.parquet).

        Raises:
            ArtifactValidationError: If row counts mismatch, ids are non-unique,
                or recorded hashes in ``manifest.json`` (when present) don't match.
        """
        directory = Path(directory)
        vectors = read_npy(directory / "vectors.npy")
        index = read_parquet(directory / "index.parquet", schema=INDEX_SCHEMA)

        rows = index.column("row").to_pylist()
        node_ids_by_row = index.column("node_id").to_pylist()
        if sorted(rows) != list(range(len(rows))):
            raise ArtifactValidationError(
                f"index.parquet 'row' values must be exactly 0..{len(rows) - 1}"
            )
        ordered_ids = [node_ids_by_row[rows.index(i)] for i in range(len(rows))]

        if vectors.shape[0] != len(ordered_ids):
            raise ArtifactValidationError(
                f"vectors.npy has {vectors.shape[0]} rows but index.parquet has {len(ordered_ids)}"
            )

        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if "sha256_vectors" in manifest and manifest["sha256_vectors"] != sha256_file(directory / "vectors.npy"):
                raise ArtifactValidationError(f"vectors.npy hash mismatch against manifest in {directory}")
            if "sha256_index" in manifest and manifest["sha256_index"] != sha256_file(directory / "index.parquet"):
                raise ArtifactValidationError(f"index.parquet hash mismatch against manifest in {directory}")

        return cls(vectors=vectors, node_ids=ordered_ids)

    def row_of(self, node_id: str) -> int:
        """Return the row index for ``node_id``.

        Raises:
            ArtifactValidationError: If ``node_id`` is not present.
        """
        try:
            return self._row_of[node_id]
        except KeyError as exc:
            raise ArtifactValidationError(f"node_id '{node_id}' not found in AlignedMatrix") from exc

    def select(self, node_ids: list[str]) -> "AlignedMatrix":
        """Return a new AlignedMatrix reordered/subset to exactly ``node_ids``.

        Raises:
            ArtifactValidationError: If any requested id is missing.
        """
        missing = [nid for nid in node_ids if nid not in self._row_of]
        if missing:
            raise ArtifactValidationError(f"select() missing ids: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        rows = [self._row_of[nid] for nid in node_ids]
        return AlignedMatrix(vectors=self.vectors[rows], node_ids=list(node_ids))

    def save(self, directory: str | Path) -> None:
        """Write ``vectors.npy``, ``index.parquet``, and a hash ``manifest.json``."""
        directory = Path(directory)
        write_npy(self.vectors.astype(np.float32, copy=False), directory / "vectors.npy")

        index_table = pa.table(
            {
                "row": pa.array(range(len(self.node_ids)), type=pa.int32()),
                "node_id": pa.array(self.node_ids, type=pa.string()),
            },
            schema=INDEX_SCHEMA,
        )
        write_parquet(index_table, directory / "index.parquet", schema=INDEX_SCHEMA)

        write_json(
            {
                "shape": list(self.vectors.shape),
                "dtype": str(self.vectors.dtype),
                "sha256_vectors": sha256_file(directory / "vectors.npy"),
                "sha256_index": sha256_file(directory / "index.parquet"),
            },
            directory / "manifest.json",
        )


@dataclass
class NodeUniverse:
    """An ordered list of node ids a training run operates on, with each id's graph_role."""

    node_ids: list[str]
    graph_roles: list[str]

    def __post_init__(self) -> None:
        if len(self.node_ids) != len(self.graph_roles):
            raise ArtifactValidationError("NodeUniverse node_ids and graph_roles must be the same length")
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ArtifactValidationError("NodeUniverse node_ids must be unique")

    @classmethod
    def from_parquet(cls, path: str | Path) -> "NodeUniverse":
        """Load a NodeUniverse from a ``universe.parquet`` file, ordered by ``row``."""
        table = read_parquet(path, schema=UNIVERSE_SCHEMA)
        rows = table.column("row").to_pylist()
        node_ids = table.column("node_id").to_pylist()
        graph_roles = table.column("graph_role").to_pylist()
        order = sorted(range(len(rows)), key=lambda i: rows[i])
        return cls(
            node_ids=[node_ids[i] for i in order],
            graph_roles=[graph_roles[i] for i in order],
        )

    def save(self, path: str | Path) -> None:
        """Write this universe to ``universe.parquet``."""
        table = pa.table(
            {
                "row": pa.array(range(len(self.node_ids)), type=pa.int32()),
                "node_id": pa.array(self.node_ids, type=pa.string()),
                "graph_role": pa.array(self.graph_roles, type=pa.string()),
            },
            schema=UNIVERSE_SCHEMA,
        )
        write_parquet(table, path, schema=UNIVERSE_SCHEMA)
