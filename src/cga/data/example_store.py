"""Read/write ``examples.parquet`` with referential integrity against a NodeUniverse.

Every id an example references (anchor, text pos/neg, graph pos/neg) must
resolve inside the universe the training run actually operates on — an id
that only exists in some upstream store but was excluded from the universe
must not silently leak into training.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from cga.core.errors import ArtifactValidationError
from cga.core.io import read_parquet, write_parquet
from cga.data.aligned import NodeUniverse
from cga.data.schemas import EXAMPLES_SCHEMA

_ID_COLUMNS = ["anchor_id", "text_pos_id", "text_neg_id", "graph_pos_id", "graph_neg_id"]


def write_examples(table: pa.Table, path: str | Path, *, universe: NodeUniverse) -> None:
    """Write ``examples.parquet``, enforcing schema, pk uniqueness, and universe FK.

    Raises:
        ArtifactValidationError: If any referenced id is outside ``universe``,
            or ``example_id`` is not unique.
    """
    universe_ids = set(universe.node_ids)
    for column in _ID_COLUMNS:
        for example_id, ref_id in zip(table.column("example_id").to_pylist(), table.column(column).to_pylist()):
            if ref_id not in universe_ids:
                raise ArtifactValidationError(
                    f"Example '{example_id}' references '{column}'='{ref_id}' outside the universe"
                )

    write_parquet(table, path, schema=EXAMPLES_SCHEMA)

    written = read_parquet(path, schema=EXAMPLES_SCHEMA)
    seen: set = set()
    for example_id in written.column("example_id").to_pylist():
        if example_id in seen:
            raise ArtifactValidationError(f"Duplicate example_id '{example_id}' in examples.parquet")
        seen.add(example_id)


def read_examples(path: str | Path) -> pa.Table:
    """Read ``examples.parquet`` with schema validation."""
    return read_parquet(path, schema=EXAMPLES_SCHEMA)
