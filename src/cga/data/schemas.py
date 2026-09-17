"""Canonical table schemas: pyarrow schemas for storage, pydantic models for rows.

Every adapter produces exactly the tables declared here (target_architecture.md
§6). The pyarrow schema is what's enforced on Parquet write/read; the pydantic
model is what adapters and pipeline code build rows with before they're batched
into a table, giving field-level validation before anything touches disk.
"""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

GraphRole = Literal["graph_1", "graph_2"]
EmbeddingStatus = Literal["ok", "missing_text", "empty_text", "failed", "skipped"]
AlignmentSplit = Literal["train", "test", "unassigned"]
NodeSplit = Literal["train", "val", "test"]
ExampleSplit = Literal["train", "val", "test"]
Sampler = Literal["rank_topk", "percentile", "supervised"]

# --- 6.1 nodes.parquet -------------------------------------------------------

NODES_SCHEMA = pa.schema(
    [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("graph_key", pa.string(), nullable=False),
        pa.field("graph_role", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("external_id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("node_type", pa.string(), nullable=True),
        pa.field("metadata", pa.string(), nullable=False),
        pa.field("has_text", pa.bool_(), nullable=False),
        pa.field("source_version", pa.string(), nullable=False),
        pa.field("raw_sha256", pa.string(), nullable=False),
    ]
)


class NodeRecord(BaseModel):
    """One row of ``nodes.parquet``."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    graph_key: str
    graph_role: GraphRole
    source_id: str
    external_id: str
    name: str
    node_type: str | None = None
    metadata: str = "{}"
    has_text: bool
    source_version: str
    raw_sha256: str


# --- 6.2 edges.parquet -------------------------------------------------------

EDGES_SCHEMA = pa.schema(
    [
        pa.field("edge_id", pa.string(), nullable=False),
        pa.field("src_id", pa.string(), nullable=False),
        pa.field("dst_id", pa.string(), nullable=False),
        pa.field("graph_role", pa.string(), nullable=False),
        pa.field("relation", pa.string(), nullable=False),
        pa.field("directed", pa.bool_(), nullable=False),
        pa.field("weight", pa.float32(), nullable=False),
    ]
)


class EdgeRecord(BaseModel):
    """One row of ``edges.parquet``."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    src_id: str
    dst_id: str
    graph_role: GraphRole
    relation: str
    directed: bool
    weight: float = 1.0


# --- 6.3 texts.parquet -------------------------------------------------------

TEXTS_SCHEMA = pa.schema(
    [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("text_variant", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
        pa.field("n_chars", pa.int32(), nullable=False),
    ]
)


class TextRecord(BaseModel):
    """One row of ``texts.parquet``. ``text`` may be empty but is never null."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    text_variant: str
    text: str = ""
    text_sha256: str
    n_chars: int


# --- 6.4 alignments.parquet --------------------------------------------------

ALIGNMENTS_SCHEMA = pa.schema(
    [
        pa.field("alignment_id", pa.string(), nullable=False),
        pa.field("node_1_id", pa.string(), nullable=False),
        pa.field("node_2_id", pa.string(), nullable=False),
        pa.field("relation", pa.string(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("candidates", pa.list_(pa.string()), nullable=True),
    ]
)


class AlignmentRecord(BaseModel):
    """One row of ``alignments.parquet``. Table is empty for datasets with no ground truth."""

    model_config = ConfigDict(extra="forbid")

    alignment_id: str
    node_1_id: str
    node_2_id: str
    relation: str = "equivalence"
    score: float = 1.0
    split: AlignmentSplit = "unassigned"
    candidates: list[str] | None = None


# --- 6.5 splits.parquet -------------------------------------------------------

SPLITS_SCHEMA = pa.schema(
    [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("graph_role", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)


class SplitRecord(BaseModel):
    """One row of ``splits.parquet``: this node's train/val/test partition.

    Written by ``ingest.split_nodes`` for every dataset, including those with
    no alignment ground truth — a proportional, seeded per-``graph_role``
    partition (target_architecture.md §5.1.1), independent of whatever split a
    dataset's own alignment source may provide.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    graph_role: GraphRole
    split: NodeSplit


# --- 6.6 examples.parquet ----------------------------------------------------

EXAMPLES_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("anchor_id", pa.string(), nullable=False),
        pa.field("text_pos_id", pa.string(), nullable=False),
        pa.field("text_neg_id", pa.string(), nullable=False),
        pa.field("graph_pos_id", pa.string(), nullable=False),
        pa.field("graph_neg_id", pa.string(), nullable=False),
        pa.field("sampler", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)


class ExampleRecord(BaseModel):
    """One row of ``examples.parquet``."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    anchor_id: str
    text_pos_id: str
    text_neg_id: str
    graph_pos_id: str
    graph_neg_id: str
    sampler: Sampler
    split: ExampleSplit


# --- 6.7 Embedding store tables ----------------------------------------------

STATUS_SCHEMA = pa.schema(
    [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("attempts", pa.int16(), nullable=False),
        pa.field("error", pa.string(), nullable=True),
        pa.field("input_sha256", pa.string(), nullable=True),
        pa.field("n_chunks", pa.int16(), nullable=True),
        pa.field("updated_at", pa.timestamp("us"), nullable=False),
    ]
)


class StatusRecord(BaseModel):
    """One row of an embedding store's ``status.parquet``."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    status: EmbeddingStatus
    attempts: int = 0
    error: str | None = None
    input_sha256: str | None = None
    n_chunks: int | None = None
    updated_at: str


INDEX_SCHEMA = pa.schema(
    [
        pa.field("row", pa.int32(), nullable=False),
        pa.field("node_id", pa.string(), nullable=False),
    ]
)


class IndexRecord(BaseModel):
    """One row of an ``index.parquet`` sidecar for an ``AlignedMatrix``."""

    model_config = ConfigDict(extra="forbid")

    row: int
    node_id: str


SHARD_SCHEMA = pa.schema(
    [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32()), nullable=False),
        pa.field("input_sha256", pa.string(), nullable=True),
        pa.field("n_chunks", pa.int16(), nullable=True),
    ]
)


class ShardRecord(BaseModel):
    """One row of an embedding store's ``shards/part-*.parquet``."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    vector: list[float]
    input_sha256: str | None = None
    n_chunks: int | None = None


UNIVERSE_SCHEMA = pa.schema(
    [
        pa.field("row", pa.int32(), nullable=False),
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("graph_role", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("held_out", pa.bool_(), nullable=False),
    ]
)


class UniverseRecord(BaseModel):
    """One row of ``universe.parquet``.

    ``split`` is carried verbatim from the ingest run's ``splits.parquet``
    (target_architecture.md §5.1.1). ``held_out`` is derived from it: True when
    ``split in {val, test}`` **or** the node appears in a ``test``-split
    ``alignments.parquet`` row (§5.3 step 1). Held-out nodes are excluded from
    unsupervised similarity/sampling in ``training_data`` so they cannot leak
    into contrastive examples the model trains on.
    """

    model_config = ConfigDict(extra="forbid")

    row: int
    node_id: str
    graph_role: GraphRole
    split: NodeSplit = "train"
    held_out: bool = False
