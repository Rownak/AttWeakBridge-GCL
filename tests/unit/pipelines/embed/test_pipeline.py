"""Tests for the ``embed`` pipeline against hand-built ingest-shaped fixtures."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import cga.pipelines  # noqa: F401  (registers pipelines)
from cga.core.artifacts import write_latest
from cga.core.io import read_json
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix
from cga.data.embedding_store import EmbeddingStore
from cga.data.schemas import EDGES_SCHEMA, NODES_SCHEMA, TEXTS_SCHEMA
from cga.pipelines.embed.pipeline import build_embedder_id

NODE_IDS = ["attack:1", "attack:2", "attack:3"]


def _write_ingest_run(ingest_run_dir: Path, *, texts: dict[str, str]) -> None:
    ingest_run_dir.mkdir(parents=True, exist_ok=True)

    nodes = pa.Table.from_pylist(
        [
            {
                "node_id": nid,
                "graph_key": "attack",
                "graph_role": "graph_1",
                "source_id": nid.split(":")[1],
                "external_id": nid,
                "name": nid,
                "node_type": None,
                "metadata": "{}",
                "has_text": nid in texts,
                "source_version": "test",
                "raw_sha256": "0" * 64,
            }
            for nid in NODE_IDS
        ],
        schema=NODES_SCHEMA,
    )
    pq.write_table(nodes, ingest_run_dir / "nodes.parquet")

    edges = pa.Table.from_pylist(
        [
            {
                "edge_id": "e1",
                "src_id": "attack:1",
                "dst_id": "attack:2",
                "graph_role": "graph_1",
                "relation": "uses",
                "directed": True,
                "weight": 1.0,
            }
        ],
        schema=EDGES_SCHEMA,
    )
    pq.write_table(edges, ingest_run_dir / "edges.parquet")

    text_rows = [
        {
            "node_id": nid,
            "text_variant": "description",
            "text": text,
            "text_sha256": "0" * 64,
            "n_chars": len(text),
        }
        for nid, text in texts.items()
    ]
    texts_table = pa.Table.from_pylist(text_rows, schema=TEXTS_SCHEMA)
    pq.write_table(texts_table, ingest_run_dir / "texts.parquet")

    write_latest(ingest_run_dir.parent, ingest_run_dir.name)


class _FakeEmbedder:
    """Stand-in for HfMeanPoolEmbedder: no HF download, deterministic vectors."""

    hidden_size = 4

    def __init__(self, *, texts_that_raise=(), **kwargs):
        self._texts_that_raise = set(texts_that_raise)

    def embed(self, text: str):
        if text in self._texts_that_raise:
            raise RuntimeError(f"boom on {text!r}")
        vector = [float(len(text))] * self.hidden_size
        return vector, 1


def _build_config(tmp_path: Path) -> dict:
    return {
        "project": "cga",
        "seed": 42,
        "device": "cpu",
        "paths": {
            "raw": str(tmp_path / "raw"),
            "processed": str(tmp_path / "processed"),
            "artifacts": str(tmp_path / "artifacts"),
            "models": str(tmp_path / "models"),
        },
        "logging": {"level": "INFO", "file": False},
        "run": {"name": None},
        "dataset": {"id": "attack_cwe"},
        "embed": {
            "inputs": {"ingest_run": "latest", "text_variant": "description"},
            "embedder": {
                "kind": "hf_mean_pool",
                "model_ref": "fake/model",
                "label": "fake_model",
                "max_length": 512,
                "add_special_tokens": False,
            },
            "on_error": "continue",
            "resume": None,
        },
    }


def _run_embed(config: dict, tmp_path: Path) -> RunContext:
    pipeline_cls = pipelines.get("embed")
    pipeline = pipeline_cls()
    embedder_id = build_embedder_id(config["embed"]["embedder"])
    run_root = Path(config["paths"]["artifacts"]) / "embeddings" / config["dataset"]["id"] / embedder_id
    ctx = RunContext.create(
        pipeline_name="embed",
        pipeline_version="0.0.0-test",
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=run_root,
    )
    pipeline.run(ctx)
    return ctx


def test_build_embedder_id_default_label_from_model_ref():
    embedder_id = build_embedder_id({"kind": "hf_mean_pool", "model_ref": "jackaduma/SecRoBERTa"})
    assert embedder_id == "hf_mean_pool-jackaduma_SecRoBERTa"


def test_build_embedder_id_uses_explicit_label():
    embedder_id = build_embedder_id({"kind": "hf_mean_pool", "model_ref": "x", "label": "pt_SecRoBERTa"})
    assert embedder_id == "hf_mean_pool-pt_SecRoBERTa"


def test_build_embedder_id_includes_finetune_run_id():
    embedder_id = build_embedder_id(
        {"kind": "hf_mean_pool", "label": "pt_SecRoBERTa", "finetune_run_id": "20260101T000000Z-abcd1234"}
    )
    assert embedder_id == "hf_mean_pool-pt_SecRoBERTa-ft:20260101"


def test_embed_pipeline_ok_and_empty_text(tmp_path, monkeypatch):
    monkeypatch.setattr("cga.pipelines.embed.pipeline.HfMeanPoolEmbedder", _FakeEmbedder)

    config = _build_config(tmp_path)
    ingest_run_dir = Path(config["paths"]["processed"]) / "attack_cwe" / "20260101T000000Z-aaaaaaaa"
    _write_ingest_run(ingest_run_dir, texts={"attack:1": "hello world", "attack:2": ""})
    # attack:3 has no text row at all -> missing_text

    ctx = _run_embed(config, tmp_path)

    assert ctx.manifest.status == "success"
    store = EmbeddingStore(ctx.run_dir)
    status = store._read_status()
    status_by_node = dict(zip(status.column("node_id").to_pylist(), status.column("status").to_pylist()))
    assert status_by_node["attack:1"] == "ok"
    assert status_by_node["attack:2"] == "empty_text"
    assert status_by_node["attack:3"] == "missing_text"

    matrix = AlignedMatrix.load(ctx.run_dir)
    assert matrix.node_ids == ["attack:1"]

    stats = read_json(ctx.run_dir / "stats.json")
    assert stats["n_ok"] == 1
    assert stats["n_empty_text"] == 1
    assert stats["n_missing_text"] == 1


def test_embed_pipeline_records_failed_and_continues(tmp_path, monkeypatch):
    def _make_fake_embedder(**kwargs):
        return _FakeEmbedder(texts_that_raise={"hello world"})

    monkeypatch.setattr("cga.pipelines.embed.pipeline.HfMeanPoolEmbedder", _make_fake_embedder)

    config = _build_config(tmp_path)
    ingest_run_dir = Path(config["paths"]["processed"]) / "attack_cwe" / "20260101T000000Z-aaaaaaaa"
    _write_ingest_run(
        ingest_run_dir, texts={"attack:1": "hello world", "attack:2": "safe text", "attack:3": "also safe"}
    )

    ctx = _run_embed(config, tmp_path)

    assert ctx.manifest.status == "success"
    store = EmbeddingStore(ctx.run_dir)
    status = store._read_status()
    status_by_node = dict(zip(status.column("node_id").to_pylist(), status.column("status").to_pylist()))
    assert status_by_node["attack:1"] == "failed"
    assert status_by_node["attack:2"] == "ok"
    assert status_by_node["attack:3"] == "ok"

    matrix = AlignedMatrix.load(ctx.run_dir)
    assert set(matrix.node_ids) == {"attack:2", "attack:3"}


def test_embed_pipeline_writes_latest(tmp_path, monkeypatch):
    monkeypatch.setattr("cga.pipelines.embed.pipeline.HfMeanPoolEmbedder", _FakeEmbedder)

    config = _build_config(tmp_path)
    ingest_run_dir = Path(config["paths"]["processed"]) / "attack_cwe" / "20260101T000000Z-aaaaaaaa"
    _write_ingest_run(ingest_run_dir, texts={"attack:1": "hello world", "attack:2": "more text", "attack:3": "text"})

    ctx = _run_embed(config, tmp_path)

    embedder_id = build_embedder_id(config["embed"]["embedder"])
    lineage_dir = Path(config["paths"]["artifacts"]) / "embeddings" / "attack_cwe" / embedder_id
    latest_run_id = (lineage_dir / "LATEST").read_text(encoding="utf-8").strip()
    assert latest_run_id == ctx.run_id


def test_embed_pipeline_deepwalk_skips_isolated_node(tmp_path):
    config = _build_config(tmp_path)
    config["embed"]["embedder"] = {
        "kind": "deepwalk",
        "label": "deepwalk",
        "dimensions": 8,
        "walk_length": 5,
        "num_walks": 5,
        "window": 2,
        "workers": 1,
        "epochs": 2,
    }
    ingest_run_dir = Path(config["paths"]["processed"]) / "attack_cwe" / "20260101T000000Z-aaaaaaaa"
    _write_ingest_run(ingest_run_dir, texts={})  # attack:3 is isolated (no edges touch it)

    ctx = _run_embed(config, tmp_path)

    assert ctx.manifest.status == "success"
    store = EmbeddingStore(ctx.run_dir)
    status = store._read_status()
    status_by_node = dict(zip(status.column("node_id").to_pylist(), status.column("status").to_pylist()))
    assert status_by_node["attack:1"] == "ok"
    assert status_by_node["attack:2"] == "ok"
    assert status_by_node["attack:3"] == "skipped"


class _FakeBatchEmbedder(_FakeEmbedder):
    """Batch-capable stand-in for SentenceTransformerEmbedder; records the batches it saw."""

    batches: list[list[str]] = []
    fail_batches = False

    def __init__(self, **kwargs):
        super().__init__()
        type(self).batches = []
        type(self).kwargs = kwargs

    def embed_batch(self, texts):
        type(self).batches.append(list(texts))
        if type(self).fail_batches:
            raise RuntimeError("batch boom")
        return [self.embed(t) for t in texts]


def _run_sentence_transformer(tmp_path, monkeypatch, *, fail_batches, batch_size):
    _FakeBatchEmbedder.fail_batches = fail_batches
    monkeypatch.setattr("cga.pipelines.embed.pipeline.SentenceTransformerEmbedder", _FakeBatchEmbedder)
    config = _build_config(tmp_path)
    config["embed"]["embedder"] = {
        "kind": "sentence_transformer",
        "model_ref": "fake/st",
        "label": "fake_st",
        "batch_size": batch_size,
    }
    ingest_run_dir = Path(config["paths"]["processed"]) / "attack_cwe" / "20260101T000000Z-aaaaaaaa"
    _write_ingest_run(ingest_run_dir, texts={"attack:1": "hello world", "attack:2": "", "attack:3": "text"})
    return _run_embed(config, tmp_path)


def test_embed_pipeline_sentence_transformer_batches_texts_and_skips_empty(tmp_path, monkeypatch):
    ctx = _run_sentence_transformer(tmp_path, monkeypatch, fail_batches=False, batch_size=2)

    assert ctx.manifest.status == "success"
    assert _FakeBatchEmbedder.batches == [["hello world", "text"]]  # empty text is never sent to the model
    matrix = AlignedMatrix.load(ctx.run_dir)
    assert matrix.node_ids == ["attack:1", "attack:3"]
    assert build_embedder_id({"kind": "sentence_transformer", "model_ref": "x", "label": "fake_st"}) == "sentence_transformer-fake_st"


def test_embed_pipeline_chunking_is_on_only_for_boolean_true(tmp_path, monkeypatch):
    """The legacy string ``chunking: fixed`` (base embed.yaml, hf_mean_pool) must not turn chunking on."""
    chunked = {}
    for value in ("fixed", False, True):
        monkeypatch.setattr("cga.pipelines.embed.pipeline.SentenceTransformerEmbedder", _FakeBatchEmbedder)
        config = _build_config(tmp_path / str(value))
        config["embed"]["embedder"] = {"kind": "sentence_transformer", "model_ref": "x", "label": "st", "chunking": value}
        ingest_run_dir = Path(config["paths"]["processed"]) / "attack_cwe" / "20260101T000000Z-aaaaaaaa"
        _write_ingest_run(ingest_run_dir, texts={"attack:1": "hello", "attack:2": "", "attack:3": "text"})
        _run_embed(config, tmp_path / str(value))
        chunked[value] = _FakeBatchEmbedder.kwargs["chunking"]

    assert chunked == {"fixed": False, False: False, True: True}


def test_embed_pipeline_sentence_transformer_falls_back_to_per_node_when_a_batch_fails(tmp_path, monkeypatch):
    ctx = _run_sentence_transformer(tmp_path, monkeypatch, fail_batches=True, batch_size=2)

    assert ctx.manifest.status == "success"
    assert read_json(ctx.run_dir / "stats.json")["n_ok"] == 2  # per-node embed() still succeeded
