"""Schema-enforcing readers/writers for the artifact formats used across pipelines.

Parquet is used for all record-oriented tables (typed columns, cross-Python-version
stability, column-selective reads); ``.npy`` for dense vectors; JSON only for small
human-readable summaries. No pickle crosses a pipeline boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cga.core.errors import ArtifactValidationError


def sha256_file(path: str | Path) -> str:
    """Return the sha256 hex digest of a file's contents, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(table: pa.Table, path: str | Path, schema: pa.Schema | None = None) -> None:
    """Write a pyarrow Table to Parquet, enforcing ``schema`` if provided.

    Raises:
        ArtifactValidationError: If ``table``'s schema does not match ``schema``.
    """
    path = Path(path)
    if schema is not None and not table.schema.equals(schema, check_metadata=False):
        try:
            table = table.cast(schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise ArtifactValidationError(
                f"Table schema does not match expected schema for {path}: {exc}"
            ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_parquet(path: str | Path, schema: pa.Schema | None = None) -> pa.Table:
    """Read a Parquet file, validating against ``schema`` if provided.

    Raises:
        ArtifactValidationError: If the file is missing or its schema mismatches.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactValidationError(f"Parquet file not found: {path}")
    table = pq.read_table(path)
    if schema is not None and not table.schema.equals(schema, check_metadata=False):
        raise ArtifactValidationError(
            f"Parquet file {path} has schema {table.schema} but expected {schema}"
        )
    return table


def write_npy(array: np.ndarray, path: str | Path) -> None:
    """Write a numpy array to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def read_npy(path: str | Path) -> np.ndarray:
    """Read a numpy array from ``path``.

    Raises:
        ArtifactValidationError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactValidationError(f"Numpy file not found: {path}")
    return np.load(path)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Write ``obj`` as JSON to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)


def read_json(path: str | Path) -> Any:
    """Read JSON from ``path``.

    Raises:
        ArtifactValidationError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactValidationError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
