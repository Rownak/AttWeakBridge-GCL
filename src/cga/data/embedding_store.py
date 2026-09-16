"""Embedding store: append-only shards + per-node status, materialized to an AlignedMatrix.

``status`` is the ground truth of what happened to each node
(``ok``/``missing_text``/``empty_text``/``failed``/``skipped``). Generation
appends shards incrementally and upserts status per node so a crash mid-run
loses no completed work; ``materialize()`` builds the final ``vectors.npy`` +
``index.parquet`` from ``ok`` rows only; ``pending_nodes()`` drives
``--resume`` by returning every node that is not yet ``ok``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa

from cga.core.io import read_parquet, write_parquet
from cga.data.aligned import AlignedMatrix
from cga.data.schemas import INDEX_SCHEMA, SHARD_SCHEMA, STATUS_SCHEMA

_TERMINAL_OK = "ok"


class EmbeddingStore:
    """One store per ``(dataset_id, embedder_id, run_id)``, rooted at ``directory``.

    Layout:
        ``manifest.json``, ``status.parquet``, ``shards/part-*.parquet``,
        ``vectors.npy``, ``index.parquet`` (the latter two only after
        :meth:`materialize`).
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.shards_dir = self.directory / "shards"
        self.status_path = self.directory / "status.parquet"

    def _read_status(self) -> pa.Table:
        if not self.status_path.is_file():
            return pa.table(
                {
                    "node_id": pa.array([], type=pa.string()),
                    "status": pa.array([], type=pa.string()),
                    "attempts": pa.array([], type=pa.int16()),
                    "error": pa.array([], type=pa.string()),
                    "input_sha256": pa.array([], type=pa.string()),
                    "n_chunks": pa.array([], type=pa.int16()),
                    "updated_at": pa.array([], type=pa.timestamp("us")),
                },
                schema=STATUS_SCHEMA,
            )
        return read_parquet(self.status_path, schema=STATUS_SCHEMA)

    def upsert_status(
        self,
        node_id: str,
        status: str,
        *,
        error: str | None = None,
        input_sha256: str | None = None,
        n_chunks: int | None = None,
    ) -> None:
        """Insert or update ``node_id``'s status row, incrementing ``attempts``.

        Existing rows for ``node_id`` are replaced (not appended), so
        ``status.parquet`` always holds exactly one row per node.
        """
        existing = self._read_status()
        existing_ids = existing.column("node_id").to_pylist()
        prev_attempts = 0
        if node_id in existing_ids:
            idx = existing_ids.index(node_id)
            prev_attempts = existing.column("attempts")[idx].as_py()
            keep_mask = [i != idx for i in range(existing.num_rows)]
            existing = existing.filter(pa.array(keep_mask))

        new_row = pa.table(
            {
                "node_id": pa.array([node_id], type=pa.string()),
                "status": pa.array([status], type=pa.string()),
                "attempts": pa.array([prev_attempts + 1], type=pa.int16()),
                "error": pa.array([error], type=pa.string()),
                "input_sha256": pa.array([input_sha256], type=pa.string()),
                "n_chunks": pa.array([n_chunks], type=pa.int16()),
                "updated_at": pa.array([datetime.now(timezone.utc)], type=pa.timestamp("us")),
            },
            schema=STATUS_SCHEMA,
        )
        combined = pa.concat_tables([existing, new_row])
        write_parquet(combined, self.status_path, schema=STATUS_SCHEMA)

    def append_shard(self, node_id: str, vector: list[float], *, input_sha256: str | None = None, n_chunks: int | None = None) -> None:
        """Append one node's vector as a new shard file (incremental, crash-safe)."""
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        existing_shards = list(self.shards_dir.glob("part-*.parquet"))
        part_num = len(existing_shards)
        table = pa.table(
            {
                "node_id": pa.array([node_id], type=pa.string()),
                "vector": pa.array([vector], type=pa.list_(pa.float32())),
                "input_sha256": pa.array([input_sha256], type=pa.string()),
                "n_chunks": pa.array([n_chunks], type=pa.int16()),
            },
            schema=SHARD_SCHEMA,
        )
        write_parquet(table, self.shards_dir / f"part-{part_num:05d}.parquet", schema=SHARD_SCHEMA)

    def pending_nodes(self, all_node_ids: list[str]) -> list[str]:
        """Return every id in ``all_node_ids`` whose current status is not ``ok``.

        Nodes with no status row yet (never attempted) are also pending.
        """
        status = self._read_status()
        ok_ids = {
            nid
            for nid, st in zip(status.column("node_id").to_pylist(), status.column("status").to_pylist())
            if st == _TERMINAL_OK
        }
        return [nid for nid in all_node_ids if nid not in ok_ids]

    def materialize(self) -> AlignedMatrix:
        """Build ``vectors.npy`` + ``index.parquet`` from ``ok``-status shard rows only.

        Later shards for the same ``node_id`` win (most recent attempt), so a
        retried-then-succeeded node's stale failed shard is superseded.
        """
        status = self._read_status()
        ok_ids_ordered = [
            nid
            for nid, st in zip(status.column("node_id").to_pylist(), status.column("status").to_pylist())
            if st == _TERMINAL_OK
        ]
        ok_id_set = set(ok_ids_ordered)

        shard_paths = sorted(self.shards_dir.glob("part-*.parquet")) if self.shards_dir.is_dir() else []
        latest_vector: dict[str, list[float]] = {}
        for shard_path in shard_paths:
            shard = read_parquet(shard_path, schema=SHARD_SCHEMA)
            for node_id, vector in zip(shard.column("node_id").to_pylist(), shard.column("vector").to_pylist()):
                if node_id in ok_id_set:
                    latest_vector[node_id] = vector

        final_ids = [nid for nid in ok_ids_ordered if nid in latest_vector]
        vectors = np.array([latest_vector[nid] for nid in final_ids], dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(len(final_ids), -1)

        matrix = AlignedMatrix(vectors=vectors, node_ids=final_ids)
        matrix.save(self.directory)
        return matrix
