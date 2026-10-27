"""Tests for ``cga.pipelines.embed.text_embedders``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cga.pipelines.embed.text_embedders import SentenceTransformerEmbedder, chunk_tokens


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


class _StubSentenceTransformer:
    """Stand-in for ``sentence_transformers.SentenceTransformer``: no download, norm-3 vectors."""

    max_seq_length = 384
    encoded: list[list[str]] = []

    def __init__(self, model_ref, device="cpu"):
        self.model_ref, self.device = model_ref, device

    def eval(self):
        return self

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
        assert convert_to_numpy and not normalize_embeddings
        type(self).encoded.append(list(texts))
        return np.array([[float(len(t)), 0.0, 0.0] for t in texts])


def _stub_embedder(monkeypatch, **kwargs):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _StubSentenceTransformer)
    _StubSentenceTransformer.encoded = []
    return SentenceTransformerEmbedder(model_ref="stub/model", **kwargs)


def test_sentence_transformer_embedder_returns_unnormalised_vector(monkeypatch):
    embedder = _stub_embedder(monkeypatch, batch_size=8)

    vector, n_chunks = embedder.embed("abcd")

    assert vector == [4.0, 0.0, 0.0]  # norm 4, so not L2-normalised
    assert n_chunks == 1
    assert embedder.hidden_size == 3
    assert embedder.max_seq_length == 384


def test_sentence_transformer_embedder_empty_text_is_zero_vector_and_not_encoded(monkeypatch):
    embedder = _stub_embedder(monkeypatch)

    results = embedder.embed_batch(["ab", "", "abc"])

    assert results == [([2.0, 0.0, 0.0], 1), ([0.0, 0.0, 0.0], 0), ([3.0, 0.0, 0.0], 1)]
    assert _StubSentenceTransformer.encoded == [["ab", "abc"]]  # input order kept, empty text skipped


class _StubTokenizer:
    """One token per word (id = word length + 10); special tokens 1 and 2 wrap a sequence."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False, verbose=False):
        ids = [len(w) + 10 for w in text.split()]
        return {"input_ids": [1, *ids, 2] if add_special_tokens else ids}


class _StubChunkModel(_StubSentenceTransformer):
    """Stub whose sentence vector is ``[mean token id, 0, 0]`` over the unmasked tokens."""

    max_seq_length = 6  # 4 content tokens per chunk once the two special tokens are added
    tokenizer = _StubTokenizer()

    def __call__(self, features):
        ids, mask = features["input_ids"].float(), features["attention_mask"].float()
        mean = (ids * mask).sum(1) / mask.sum(1)
        return {"sentence_embedding": torch.stack([mean, torch.zeros_like(mean), torch.zeros_like(mean)], dim=1)}

    def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
        ids = [self.tokenizer(t)["input_ids"][: self.max_seq_length] for t in texts]
        return np.array([[sum(i) / len(i), 0.0, 0.0] for i in ids])


def _chunking_embedder(monkeypatch, chunking):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _StubChunkModel)
    return SentenceTransformerEmbedder(model_ref="stub/model", batch_size=2, chunking=chunking)


def test_chunked_embedding_of_a_text_that_fits_one_chunk_equals_the_unchunked_vector(monkeypatch):
    chunked = _chunking_embedder(monkeypatch, chunking=True).embed("aa bbb cccc")
    plain = _chunking_embedder(monkeypatch, chunking=False).embed("aa bbb cccc")

    assert chunked[1] == 1
    assert chunked[0] == pytest.approx(plain[0])


def test_chunked_embedding_of_a_long_text_is_the_token_weighted_mean_of_its_chunks(monkeypatch):
    text = "a bb ccc dddd e ff"  # token ids 11 12 13 14 11 12 -> chunks of 4 and 2 tokens
    vector, n_chunks = _chunking_embedder(monkeypatch, chunking=True).embed(text)

    first = (1 + 11 + 12 + 13 + 14 + 2) / 6  # chunk vector = mean over the special tokens too
    second = (1 + 11 + 12 + 2) / 4
    assert n_chunks == 2
    assert vector[0] == pytest.approx((4 * first + 2 * second) / 6)
    truncated = _chunking_embedder(monkeypatch, chunking=False).embed(text)[0]
    assert vector[0] != pytest.approx(truncated[0])


def test_chunked_embedding_keeps_empty_text_as_a_zero_vector(monkeypatch):
    results = _chunking_embedder(monkeypatch, chunking=True).embed_batch(["", "aa bbb"])

    assert results[0] == ([0.0, 0.0, 0.0], 0)
    assert results[1][1] == 1
