"""``ingest`` pipeline: dataset adapter -> canonical graph artifacts (target_architecture.md §5.1).

The only dataset-aware pipeline — and only through the adapter registry. Six
steps drain a `DatasetAdapter`'s iterators into the four canonical Parquet
tables under ``data/processed/{dataset_id}/{run_id}/``, enforcing schema,
primary-key uniqueness, and FK/role integrity via ``cga.data.graph_store``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pyarrow as pa

from cga import __version__ as PIPELINE_VERSION
from cga.adapters.base import DatasetAdapter
from cga.core.artifacts import write_latest
from cga.core.errors import ConfigError
from cga.core.io import write_json
from cga.core.pipeline import Pipeline, Step
from cga.core.registry import adapters as adapter_registry
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.graph_store import write_alignments, write_edges, write_nodes, write_texts
from cga.data.schemas import ALIGNMENTS_SCHEMA, EDGES_SCHEMA, NODES_SCHEMA, TEXTS_SCHEMA


class _IngestState:
    """Mutable state threaded across this run's steps (one instance per run)."""

    def __init__(self) -> None:
        self.adapter: DatasetAdapter | None = None
        self.nodes_table: pa.Table | None = None
        self.edges_table: pa.Table | None = None
        self.texts_table: pa.Table | None = None
        self.alignments_table: pa.Table | None = None
        self.n_dropped_edges: int = 0
        self.output_dir: Path | None = None


def _build_adapter(config: dict[str, Any]) -> DatasetAdapter:
    """Construct the registered adapter named by ``dataset.adapter`` from ``config``."""
    dataset_cfg = config.get("dataset")
    if not dataset_cfg:
        raise ConfigError("Config is missing a top-level 'dataset' block")

    adapter_name = dataset_cfg.get("adapter")
    if not adapter_name:
        raise ConfigError("dataset.adapter is required")

    adapter_cls = adapter_registry.get(adapter_name)
    adapter_config = adapter_cls.ConfigModel(**(dataset_cfg.get("adapter_config") or {}))

    kwargs: dict[str, Any] = {
        "root": dataset_cfg["root"],
        "text_variants": dataset_cfg.get("text_variants") or {},
    }
    accepted_params = inspect.signature(adapter_cls.__init__).parameters
    alignments_cfg = dataset_cfg.get("alignments")
    if alignments_cfg and "alignments" in accepted_params:
        # configs/datasets/*.yaml nests per-relation blocks under `alignments`
        # (e.g. `alignments.equivalence: {train, test, candidates}`, §7.4); an
        # adapter that accepts alignments takes the relation's own block.
        kwargs["alignments"] = alignments_cfg.get("equivalence", alignments_cfg)

    return adapter_cls(adapter_config, **kwargs)


class ResolveAdapterStep:
    """Step 1: look up and build the dataset adapter, validate sources, record fingerprint."""

    name = "resolve_adapter"

    def __init__(self, state: _IngestState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        adapter = _build_adapter(ctx.config)
        adapter.validate_sources()
        self._state.adapter = adapter

        root = Path(ctx.config["dataset"]["root"])
        for rel_path in sorted(adapter.source_fingerprint()):
            ctx.manifest.record_input(rel_path, pipeline="raw", run_id="", path=root / rel_path)

        self._state.output_dir = ctx.run_dir


class IngestNodesStep:
    """Step 2: drain ``iter_nodes()``, enforce ``node_id`` uniqueness on write."""

    name = "ingest_nodes"

    def __init__(self, state: _IngestState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        assert self._state.adapter is not None
        rows = [n.model_dump() for n in self._state.adapter.iter_nodes()]
        table = pa.Table.from_pylist(rows, schema=NODES_SCHEMA)
        write_nodes(table, self._state.output_dir / "nodes.parquet")
        self._state.nodes_table = table
        ctx.logger.info("ingest_nodes: wrote %d nodes", table.num_rows)


class IngestEdgesStep:
    """Step 3: drain ``iter_edges()``; both endpoints must exist and share ``graph_role``."""

    name = "ingest_edges"

    def __init__(self, state: _IngestState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        assert self._state.adapter is not None and self._state.nodes_table is not None
        node_ids = set(self._state.nodes_table.column("node_id").to_pylist())

        rows = []
        seen_edge_ids: set[str] = set()
        n_dropped = 0
        for edge in self._state.adapter.iter_edges():
            if edge.src_id not in node_ids or edge.dst_id not in node_ids:
                n_dropped += 1
                continue
            if edge.edge_id in seen_edge_ids:
                continue
            seen_edge_ids.add(edge.edge_id)
            rows.append(edge.model_dump())

        table = pa.Table.from_pylist(rows, schema=EDGES_SCHEMA)
        write_edges(table, self._state.output_dir / "edges.parquet", nodes=self._state.nodes_table)
        self._state.edges_table = table
        self._state.n_dropped_edges = n_dropped
        ctx.logger.info(
            "ingest_edges: wrote %d edges (%d dropped: dangling endpoint)", table.num_rows, n_dropped
        )


class IngestTextsStep:
    """Step 4: drain ``iter_texts()``, enforce ``(node_id, text_variant)`` uniqueness."""

    name = "ingest_texts"

    def __init__(self, state: _IngestState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        assert self._state.adapter is not None and self._state.nodes_table is not None
        rows = [t.model_dump() for t in self._state.adapter.iter_texts()]
        table = pa.Table.from_pylist(rows, schema=TEXTS_SCHEMA)
        write_texts(table, self._state.output_dir / "texts.parquet", nodes=self._state.nodes_table)
        self._state.texts_table = table
        ctx.logger.info("ingest_texts: wrote %d text rows", table.num_rows)


class IngestAlignmentsStep:
    """Step 5: drain ``iter_alignments()``; endpoints must exist and be in opposite graphs."""

    name = "ingest_alignments"

    def __init__(self, state: _IngestState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        assert self._state.adapter is not None and self._state.nodes_table is not None
        rows = [a.model_dump() for a in self._state.adapter.iter_alignments()]
        table = pa.Table.from_pylist(rows, schema=ALIGNMENTS_SCHEMA)
        write_alignments(
            table, self._state.output_dir / "alignments.parquet", nodes=self._state.nodes_table
        )
        self._state.alignments_table = table
        ctx.logger.info("ingest_alignments: wrote %d alignment rows", table.num_rows)


def _degree_summary(edges_table: pa.Table) -> dict[str, float]:
    if edges_table.num_rows == 0:
        return {"mean_degree": 0.0, "max_degree": 0}
    from collections import Counter

    counts: Counter[str] = Counter()
    for src, dst in zip(edges_table.column("src_id").to_pylist(), edges_table.column("dst_id").to_pylist()):
        counts[src] += 1
        counts[dst] += 1
    values = list(counts.values())
    return {"mean_degree": sum(values) / len(values), "max_degree": max(values)}


class ValidateAndManifestStep:
    """Step 6: write ``stats.json``, update ``LATEST``, record outputs."""

    name = "validate_and_manifest"

    def __init__(self, state: _IngestState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert (
            state.nodes_table is not None
            and state.edges_table is not None
            and state.texts_table is not None
            and state.alignments_table is not None
        )

        nodes_by_role: dict[str, int] = {}
        for role in state.nodes_table.column("graph_role").to_pylist():
            nodes_by_role[role] = nodes_by_role.get(role, 0) + 1

        edges_by_role: dict[str, int] = {}
        for role in state.edges_table.column("graph_role").to_pylist():
            edges_by_role[role] = edges_by_role.get(role, 0) + 1

        n_with_text = sum(1 for has_text in state.nodes_table.column("has_text").to_pylist() if has_text)

        alignment_split_counts: dict[str, int] = {}
        for split in state.alignments_table.column("split").to_pylist():
            alignment_split_counts[split] = alignment_split_counts.get(split, 0) + 1

        stats = {
            "n_nodes": state.nodes_table.num_rows,
            "n_nodes_by_graph_role": nodes_by_role,
            "n_edges": state.edges_table.num_rows,
            "n_edges_by_graph_role": edges_by_role,
            "n_edges_dropped_dangling": state.n_dropped_edges,
            "n_texts": state.texts_table.num_rows,
            "n_nodes_with_text": n_with_text,
            "degree": _degree_summary(state.edges_table),
            "n_alignments": state.alignments_table.num_rows,
            "n_alignments_by_split": alignment_split_counts,
        }
        stats_path = state.output_dir / "stats.json"
        write_json(stats, stats_path)

        for name, path in (
            ("nodes", state.output_dir / "nodes.parquet"),
            ("edges", state.output_dir / "edges.parquet"),
            ("texts", state.output_dir / "texts.parquet"),
            ("alignments", state.output_dir / "alignments.parquet"),
            ("stats", stats_path),
        ):
            ctx.manifest.record_output(name, path)

        ctx.manifest.summary = {
            "n_nodes": stats["n_nodes"],
            "n_edges": stats["n_edges"],
            "n_alignments": stats["n_alignments"],
        }


@pipelines.register("ingest")
class IngestPipeline(Pipeline):
    """Runs the selected dataset adapter and writes the four canonical tables + ``stats.json``."""

    name = "ingest"

    def build_steps(self, ctx: RunContext) -> list[Step]:
        state = _IngestState()
        self._state = state
        return [
            ResolveAdapterStep(state),
            IngestNodesStep(state),
            IngestEdgesStep(state),
            IngestTextsStep(state),
            IngestAlignmentsStep(state),
            ValidateAndManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        dataset_id = ctx.config["dataset"]["id"]
        lineage_dir = Path(ctx.config["paths"]["processed"]) / dataset_id
        write_latest(lineage_dir, ctx.run_id)
