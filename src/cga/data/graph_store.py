"""Read/write the canonical ingest tables with pk-uniqueness and FK checks.

This is the only place ``nodes``/``edges``/``texts``/``alignments``/``splits``
are written, so referential integrity (edges within one graph_role, alignments
crossing graph_role, splits covering every node exactly once) is enforced
exactly once, at the boundary between an adapter's raw output and everything
downstream that trusts these tables blindly.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from cga.core.errors import ArtifactValidationError
from cga.core.io import read_parquet, write_parquet
from cga.data.schemas import ALIGNMENTS_SCHEMA, EDGES_SCHEMA, NODES_SCHEMA, SPLITS_SCHEMA, TEXTS_SCHEMA


def _check_pk_unique(table: pa.Table, pk_column: str, table_name: str) -> None:
    values = table.column(pk_column).to_pylist()
    seen: set = set()
    for value in values:
        if value in seen:
            raise ArtifactValidationError(
                f"Duplicate primary key '{value}' in {table_name}.{pk_column}"
            )
        seen.add(value)


def _check_composite_pk_unique(table: pa.Table, pk_columns: list[str], table_name: str) -> None:
    columns = [table.column(c).to_pylist() for c in pk_columns]
    seen: set = set()
    for row in zip(*columns):
        if row in seen:
            raise ArtifactValidationError(
                f"Duplicate composite primary key {row} in {table_name}.{pk_columns}"
            )
        seen.add(row)


def write_nodes(table: pa.Table, path: str | Path) -> None:
    """Write ``nodes.parquet``, enforcing schema and ``node_id`` uniqueness."""
    write_parquet(table, path, schema=NODES_SCHEMA)
    written = read_parquet(path, schema=NODES_SCHEMA)
    _check_pk_unique(written, "node_id", "nodes")


def read_nodes(path: str | Path) -> pa.Table:
    """Read ``nodes.parquet`` with schema validation."""
    return read_parquet(path, schema=NODES_SCHEMA)


def write_edges(table: pa.Table, path: str | Path, *, nodes: pa.Table) -> None:
    """Write ``edges.parquet``, enforcing schema, pk uniqueness, and FK/role checks.

    Raises:
        ArtifactValidationError: If any ``src_id``/``dst_id`` is not in ``nodes``,
            or if an edge's endpoints have differing ``graph_role``.
    """
    node_roles = dict(zip(nodes.column("node_id").to_pylist(), nodes.column("graph_role").to_pylist()))

    src_ids = table.column("src_id").to_pylist()
    dst_ids = table.column("dst_id").to_pylist()
    edge_roles = table.column("graph_role").to_pylist()

    for edge_id, src, dst, role in zip(table.column("edge_id").to_pylist(), src_ids, dst_ids, edge_roles):
        if src not in node_roles:
            raise ArtifactValidationError(f"Edge '{edge_id}' src_id '{src}' not found in nodes")
        if dst not in node_roles:
            raise ArtifactValidationError(f"Edge '{edge_id}' dst_id '{dst}' not found in nodes")
        if node_roles[src] != node_roles[dst]:
            raise ArtifactValidationError(
                f"Edge '{edge_id}' crosses graph_role: src={node_roles[src]}, dst={node_roles[dst]}"
            )
        if role != node_roles[src]:
            raise ArtifactValidationError(
                f"Edge '{edge_id}' graph_role '{role}' does not match endpoints' graph_role '{node_roles[src]}'"
            )

    write_parquet(table, path, schema=EDGES_SCHEMA)
    written = read_parquet(path, schema=EDGES_SCHEMA)
    _check_pk_unique(written, "edge_id", "edges")


def read_edges(path: str | Path) -> pa.Table:
    """Read ``edges.parquet`` with schema validation."""
    return read_parquet(path, schema=EDGES_SCHEMA)


def write_texts(table: pa.Table, path: str | Path, *, nodes: pa.Table) -> None:
    """Write ``texts.parquet``, enforcing schema, composite pk uniqueness, and FK.

    Raises:
        ArtifactValidationError: If any ``node_id`` is not present in ``nodes``.
    """
    node_ids = set(nodes.column("node_id").to_pylist())
    for node_id in table.column("node_id").to_pylist():
        if node_id not in node_ids:
            raise ArtifactValidationError(f"texts.parquet node_id '{node_id}' not found in nodes")

    write_parquet(table, path, schema=TEXTS_SCHEMA)
    written = read_parquet(path, schema=TEXTS_SCHEMA)
    _check_composite_pk_unique(written, ["node_id", "text_variant"], "texts")


def read_texts(path: str | Path) -> pa.Table:
    """Read ``texts.parquet`` with schema validation."""
    return read_parquet(path, schema=TEXTS_SCHEMA)


def write_alignments(table: pa.Table, path: str | Path, *, nodes: pa.Table) -> None:
    """Write ``alignments.parquet``, enforcing schema, pk uniqueness, and role FK.

    Raises:
        ArtifactValidationError: If ``node_1_id``/``node_2_id`` are missing from
            ``nodes`` or are not in opposite ``graph_role``s (``node_1_id`` must
            be ``graph_1``, ``node_2_id`` must be ``graph_2``). An empty table
            (datasets without ground truth) is always valid.
    """
    if table.num_rows == 0:
        write_parquet(table, path, schema=ALIGNMENTS_SCHEMA)
        return

    node_roles = dict(zip(nodes.column("node_id").to_pylist(), nodes.column("graph_role").to_pylist()))

    for alignment_id, n1, n2 in zip(
        table.column("alignment_id").to_pylist(),
        table.column("node_1_id").to_pylist(),
        table.column("node_2_id").to_pylist(),
    ):
        if n1 not in node_roles:
            raise ArtifactValidationError(f"Alignment '{alignment_id}' node_1_id '{n1}' not found in nodes")
        if n2 not in node_roles:
            raise ArtifactValidationError(f"Alignment '{alignment_id}' node_2_id '{n2}' not found in nodes")
        if node_roles[n1] != "graph_1":
            raise ArtifactValidationError(
                f"Alignment '{alignment_id}' node_1_id '{n1}' must have graph_role graph_1, got {node_roles[n1]}"
            )
        if node_roles[n2] != "graph_2":
            raise ArtifactValidationError(
                f"Alignment '{alignment_id}' node_2_id '{n2}' must have graph_role graph_2, got {node_roles[n2]}"
            )

    write_parquet(table, path, schema=ALIGNMENTS_SCHEMA)
    written = read_parquet(path, schema=ALIGNMENTS_SCHEMA)
    _check_pk_unique(written, "alignment_id", "alignments")


def read_alignments(path: str | Path) -> pa.Table:
    """Read ``alignments.parquet`` with schema validation."""
    return read_parquet(path, schema=ALIGNMENTS_SCHEMA)


def write_splits(table: pa.Table, path: str | Path, *, nodes: pa.Table) -> None:
    """Write ``splits.parquet``, enforcing schema, FK, and one-row-per-node completeness.

    Raises:
        ArtifactValidationError: If any ``node_id`` is not present in ``nodes``,
            a node is missing a split row, or a ``node_id`` appears more than once.
    """
    node_ids = nodes.column("node_id").to_pylist()
    node_id_set = set(node_ids)

    split_node_ids = table.column("node_id").to_pylist()
    for node_id in split_node_ids:
        if node_id not in node_id_set:
            raise ArtifactValidationError(f"splits.parquet node_id '{node_id}' not found in nodes")

    write_parquet(table, path, schema=SPLITS_SCHEMA)
    written = read_parquet(path, schema=SPLITS_SCHEMA)
    _check_pk_unique(written, "node_id", "splits")

    written_ids = set(written.column("node_id").to_pylist())
    missing = node_id_set - written_ids
    if missing:
        raise ArtifactValidationError(
            f"splits.parquet is missing {len(missing)} node(s), e.g. {sorted(missing)[:5]}"
        )


def read_splits(path: str | Path) -> pa.Table:
    """Read ``splits.parquet`` with schema validation."""
    return read_parquet(path, schema=SPLITS_SCHEMA)
