"""Tests for ``cga.pipelines.embed.text_embedders``."""

from __future__ import annotations

from cga.pipelines.embed.text_embedders import chunk_tokens


def test_chunk_tokens_splits_long_sequence_into_fixed_size_chunks():
    token_ids = list(range(1000))
    chunks = chunk_tokens(token_ids, chunk_size=512)

    assert len(chunks) == 2
    assert chunks[0] == list(range(512))
    assert chunks[1] == list(range(512, 1000))


def test_chunk_tokens_short_sequence_is_one_chunk():
    token_ids = list(range(10))
    chunks = chunk_tokens(token_ids, chunk_size=512)
    assert chunks == [token_ids]


def test_chunk_tokens_empty_sequence_returns_one_empty_chunk():
    assert chunk_tokens([], chunk_size=512) == [[]]
