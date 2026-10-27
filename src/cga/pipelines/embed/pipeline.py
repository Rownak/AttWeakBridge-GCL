"""``embed`` pipeline: text and structural embedding generation (target_architecture.md §5.2).

Text and graph embedders are both node -> vector functions sharing one store
contract (``cga.data.embedding_store``), so one pipeline drives both;
``embedder.kind`` selects which. Only ``generate`` differs per kind —
``materialize`` and ``validate`` are identical either way. Per-node failures are
caught, recorded as ``status = failed``, and generation continues
(``on_error: continue``); nothing here re-raises for a single bad node.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from cga.core.artifacts import ArtifactRef, write_latest
from cga.core.errors import ConfigError
from cga.core.io import write_json
from cga.core.pipeline import Pipeline, Step
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.embedding_store import EmbeddingStore
from cga.data.graph_store import read_edges, read_nodes, read_texts
from cga.pipelines.embed.graph_embedders import build_graph, deepwalk_embed, node2vec_embed
from cga.pipelines.embed.text_embedders import HfMeanPoolEmbedder, SentenceTransformerEmbedder

_GRAPH_KINDS = {"deepwalk", "node2vec"}
_TEXT_KINDS = {"hf_mean_pool", "sentence_transformer"}


def slugify(value: str) -> str:
    """Turn a model ref / label into a filesystem- and id-safe slug."""
    return value.strip().replace("/", "_").replace(" ", "_")


def build_embedder_id(embedder_cfg: dict[str, Any]) -> str:
    """Build ``{kind}-{label}[-ft:{finetune_run_id[:8]}]`` per target_architecture.md §5.2."""
    kind = embedder_cfg["kind"]
    label = embedder_cfg.get("label") or slugify(embedder_cfg.get("model_ref", kind))
    embedder_id = f"{kind}-{label}"
    finetune_run_id = embedder_cfg.get("finetune_run_id")
    if finetune_run_id:
        embedder_id += f"-ft:{finetune_run_id[:8]}"
    return embedder_id


class _EmbedState:
    """Mutable state threaded across this run's steps (one instance per run)."""

    def __init__(self) -> None:
        self.store: EmbeddingStore | None = None
        self.all_node_ids: list[str] = []
        self.n_ok = 0
        self.n_failed = 0
        self.n_missing_text = 0
        self.n_empty_text = 0
        self.n_skipped = 0


def _load_texts_by_node(ingest_dir: Path, text_variant: str) -> dict[str, str]:
    texts = read_texts(ingest_dir / "texts.parquet")
    result: dict[str, str] = {}
    for node_id, variant, text in zip(
        texts.column("node_id").to_pylist(),
        texts.column("text_variant").to_pylist(),
        texts.column("text").to_pylist(),
    ):
        if variant == text_variant:
            result[node_id] = text
    return result


class GenerateStep:
    """Step 1: embed every pending node, recording per-node status and appending shards."""

    name = "generate"

    def __init__(self, state: _EmbedState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        cfg = ctx.config["embed"]
        embedder_cfg = cfg["embedder"]
        kind = embedder_cfg["kind"]
        on_error = cfg.get("on_error", "continue")

        ingest_ref = ArtifactRef(
            lineage_dir=Path(ctx.config["paths"]["processed"]) / ctx.dataset_id,
            ref=cfg["inputs"]["ingest_run"],
        )
        ingest_dir = ctx.resolve("ingest", ingest_ref)

        nodes_table = read_nodes(ingest_dir / "nodes.parquet")
        all_node_ids = nodes_table.column("node_id").to_pylist()
        self._state.all_node_ids = all_node_ids

        # ``--resume`` re-opens a prior run's store directory in place, so
        # already-`ok` shards are kept and only pending nodes are re-attempted;
        # a fresh run always gets its own immutable `ctx.run_dir`.
        resume_from = cfg.get("resume")
        store_dir = ctx.run_dir.parent / resume_from if resume_from else ctx.run_dir
        store = EmbeddingStore(store_dir)
        self._state.store = store
        pending = store.pending_nodes(all_node_ids)

        if kind in _TEXT_KINDS:
            self._generate_text(ctx, store, embedder_cfg, ingest_dir, pending, on_error)
        elif kind in _GRAPH_KINDS:
            self._generate_graph(ctx, store, embedder_cfg, ingest_dir, all_node_ids, pending, kind, on_error)
        else:
            raise ConfigError(f"Unknown embedder.kind '{kind}'; expected one of hf_mean_pool, sentence_transformer, deepwalk, node2vec")

    def _generate_text(
        self,
        ctx: RunContext,
        store: EmbeddingStore,
        embedder_cfg: dict[str, Any],
        ingest_dir: Path,
        pending: list[str],
        on_error: str,
    ) -> None:
        text_variant = ctx.config["embed"]["inputs"].get("text_variant", "description")
        texts_by_node = _load_texts_by_node(ingest_dir, text_variant)

        if embedder_cfg["kind"] == "sentence_transformer":
            embedder: Any = SentenceTransformerEmbedder(
                model_ref=embedder_cfg["model_ref"],
                batch_size=embedder_cfg.get("batch_size", 32),
                device=ctx.device,
                # ``is True``: the base embed.yaml carries the legacy string ``chunking: fixed`` for hf_mean_pool,
                # which must not switch this embedder to chunked mode.
                chunking=embedder_cfg.get("chunking") is True,
            )
            # Batching is what makes the GPU worth using; the per-node path below stays the fallback.
            batch_size = max(1, int(embedder_cfg.get("batch_size", 32)))
        else:
            embedder = HfMeanPoolEmbedder(
                model_ref=embedder_cfg["model_ref"],
                max_length=embedder_cfg.get("max_length", 512),
                add_special_tokens=embedder_cfg.get("add_special_tokens", False),
                device=ctx.device,
            )
            batch_size = 1

        todo: list[tuple[str, str]] = []
        for node_id in pending:
            text = texts_by_node.get(node_id)
            if text is None:
                store.upsert_status(node_id, "missing_text")
                self._state.n_missing_text += 1
            elif not text:
                store.upsert_status(node_id, "empty_text")
                self._state.n_empty_text += 1
            else:
                todo.append((node_id, text))

        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            results: list[tuple[list[float], int] | None]
            try:
                results = list(embedder.embed_batch([t for _, t in batch])) if batch_size > 1 else [None]
            except Exception:  # noqa: BLE001 - retry node by node so one bad text cannot fail its batch
                results = [None] * len(batch)
            for (node_id, text), result in zip(batch, results, strict=True):
                input_sha256 = None
                try:
                    vector, n_chunks = result if result is not None else embedder.embed(text)
                    input_sha256 = _sha256_text(text)
                    store.append_shard(node_id, vector, input_sha256=input_sha256, n_chunks=n_chunks)
                    store.upsert_status(node_id, "ok", input_sha256=input_sha256, n_chunks=n_chunks)
                    self._state.n_ok += 1
                except Exception as exc:  # noqa: BLE001 - per-node isolation is the point
                    if on_error == "abort":
                        raise
                    ctx.logger.warning("generate: node '%s' failed: %s", node_id, exc)
                    store.upsert_status(node_id, "failed", error=str(exc), input_sha256=input_sha256)
                    self._state.n_failed += 1

    def _generate_graph(
        self,
        ctx: RunContext,
        store: EmbeddingStore,
        embedder_cfg: dict[str, Any],
        ingest_dir: Path,
        all_node_ids: list[str],
        pending: list[str],
        kind: str,
        on_error: str,
    ) -> None:
        edges_table = read_edges(ingest_dir / "edges.parquet")
        edges = list(zip(edges_table.column("src_id").to_pylist(), edges_table.column("dst_id").to_pylist()))
        graph = build_graph(all_node_ids, edges)

        walk_kwargs = {
            "dimensions": embedder_cfg.get("dimensions", 128),
            "walk_length": embedder_cfg.get("walk_length", 30),
            "num_walks": embedder_cfg.get("num_walks", 200),
            "window": embedder_cfg.get("window", 5),
            "workers": embedder_cfg.get("workers", 1),
            "epochs": embedder_cfg.get("epochs", 10),
        }

        edges_sha256 = _sha256_text(repr(sorted(edges)))
        pending_set = set(pending)

        try:
            if kind == "deepwalk":
                vectors_by_node = deepwalk_embed(graph, **walk_kwargs)
            else:
                vectors_by_node = node2vec_embed(
                    graph, p=embedder_cfg.get("p", 1.0), q=embedder_cfg.get("q", 1.0), **walk_kwargs
                )
        except Exception as exc:  # noqa: BLE001 - per-node isolation is the point
            if on_error == "abort":
                raise
            ctx.logger.warning("generate: walk generation failed: %s", exc)
            for node_id in pending:
                store.upsert_status(node_id, "failed", error=str(exc))
                self._state.n_failed += 1
            return

        for node_id in pending:
            if node_id not in pending_set:
                continue
            if graph.degree(node_id) == 0:
                store.upsert_status(node_id, "skipped")
                self._state.n_skipped += 1
                continue
            vector = vectors_by_node.get(node_id)
            if vector is None:
                store.upsert_status(node_id, "skipped")
                self._state.n_skipped += 1
                continue
            store.append_shard(node_id, vector, input_sha256=edges_sha256)
            store.upsert_status(node_id, "ok", input_sha256=edges_sha256)
            self._state.n_ok += 1


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MaterializeStep:
    """Step 2: build ``vectors.npy`` + ``index.parquet`` from ``ok``-status shard rows."""

    name = "materialize"

    def __init__(self, state: _EmbedState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        assert self._state.store is not None
        matrix = self._state.store.materialize()
        ctx.logger.info("materialize: wrote %d vectors of dim %d", *matrix.vectors.shape)


class ValidateAndManifestStep:
    """Step 3: write ``stats.json``, update ``LATEST``, record outputs."""

    name = "validate_and_manifest"

    def __init__(self, state: _EmbedState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert state.store is not None

        store_dir = state.store.directory
        stats = {
            "n_total": len(state.all_node_ids),
            "n_ok": state.n_ok,
            "n_failed": state.n_failed,
            "n_missing_text": state.n_missing_text,
            "n_empty_text": state.n_empty_text,
            "n_skipped": state.n_skipped,
        }
        stats_path = store_dir / "stats.json"
        write_json(stats, stats_path)

        for name, path in (
            ("vectors", store_dir / "vectors.npy"),
            ("index", store_dir / "index.parquet"),
            ("status", store_dir / "status.parquet"),
            ("stats", stats_path),
        ):
            if path.is_file():
                extra: dict[str, Any] = {}
                if name == "vectors":
                    shape = np.load(path).shape
                    extra = {"shape": list(shape), "dtype": "float32"}
                ctx.manifest.record_output(name, path, **extra)

        ctx.manifest.summary = stats


@pipelines.register("embed")
class EmbedPipeline(Pipeline):
    """Runs the configured embedder and writes the embedding store + ``LATEST``."""

    name = "embed"

    def build_steps(self, ctx: RunContext) -> list[Step]:
        state = _EmbedState()
        self._state = state
        return [
            GenerateStep(state),
            MaterializeStep(state),
            ValidateAndManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        embedder_cfg = ctx.config["embed"]["embedder"]
        embedder_id = build_embedder_id(embedder_cfg)
        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "embeddings" / ctx.dataset_id / embedder_id
        write_latest(lineage_dir, ctx.run_id)
