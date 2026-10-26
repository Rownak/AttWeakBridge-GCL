"""Tests for EmbeddingStore resume semantics and materialization."""

from __future__ import annotations

from cga.data.embedding_store import EmbeddingStore


def test_pending_nodes_returns_all_when_no_status(tmp_path):
    store = EmbeddingStore(tmp_path)
    assert store.pending_nodes(["a", "b"]) == ["a", "b"]


def test_ok_rows_untouched_by_resume(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.upsert_status("a", "ok")
    store.upsert_status("b", "failed", error="boom")

    pending = store.pending_nodes(["a", "b"])
    assert pending == ["b"]


def test_failed_rows_rerun_and_attempts_increment(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.upsert_status("a", "failed", error="first failure")
    store.upsert_status("a", "ok")

    status = store._read_status()
    idx = status.column("node_id").to_pylist().index("a")
    assert status.column("status")[idx].as_py() == "ok"
    assert status.column("attempts")[idx].as_py() == 2

    assert store.pending_nodes(["a"]) == []


def test_materialize_excludes_non_ok_rows(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.append_shard("a", [1.0, 2.0])
    store.upsert_status("a", "ok", input_sha256="hash-a")

    store.append_shard("b", [3.0, 4.0])
    store.upsert_status("b", "failed", error="boom")

    matrix = store.materialize()
    assert matrix.node_ids == ["a"]
    assert matrix.vectors.shape == (1, 2)


def test_materialize_keeps_index_aligned_with_vectors(tmp_path):
    store = EmbeddingStore(tmp_path)
    for node_id, vec in [("a", [1.0]), ("b", [2.0]), ("c", [3.0])]:
        store.append_shard(node_id, vec)
        store.upsert_status(node_id, "ok")
    store.upsert_status("c", "failed", error="later marked failed")

    matrix = store.materialize()
    assert matrix.node_ids == ["a", "b"]
    assert matrix.vectors.shape[0] == len(matrix.node_ids)
