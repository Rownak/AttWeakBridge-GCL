"""Text embedders: ``hf_mean_pool`` (chunked token mean-pooling) and ``sentence_transformer``.

``hf_mean_pool``: tokenize, fixed-size chunk, mean-pool last hidden layer.
``sentence_transformer`` (``SentenceTransformerEmbedder``): one sentence-level vector per text from a
sentence-transformers model, truncated at the model's ``max_seq_length`` instead of chunked.

Model-agnostic by design (target_architecture.md §7.5): ``model_ref`` is any hub id
or local path loadable via ``AutoTokenizer``/``AutoModel``, so the default
(``jackaduma/SecRoBERTa``, an MLM-trained encoder) and any alternative encoder
share this one code path. Vanilla decoder-only models (GPT-2 family) are not
special-cased here and are not the default — their raw hidden states are known
to be highly anisotropic and a poor fit for similarity/clustering use, per
target_architecture.md and legacy experimentation (`legacy/LLM_embeddings/`).
Long text is split into fixed-size token chunks so no input is silently
truncated; each chunk is mean-pooled over the sequence dimension, then chunk
vectors are averaged into one per-node vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def chunk_tokens(token_ids: list[int], chunk_size: int) -> list[list[int]]:
    """Split ``token_ids`` into consecutive chunks of at most ``chunk_size``.

    Args:
        token_ids: Full sequence of token ids for one text.
        chunk_size: Maximum tokens per chunk.

    Returns:
        A list of one or more chunks; a single chunk (possibly empty) if
        ``token_ids`` is shorter than ``chunk_size``.
    """
    if not token_ids:
        return [[]]
    return [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


@dataclass
class HfMeanPoolEmbedder:
    """Loads one HF encoder once and embeds text via chunked mean-pooling.

    Attributes:
        model_ref: Hub id or local path passed to ``AutoModel``/``AutoTokenizer``.
        max_length: Chunk size in tokens.
        add_special_tokens: Whether chunks include the tokenizer's special tokens.
        device: Torch device string to run the forward pass on.
    """

    model_ref: str
    max_length: int = 512
    add_special_tokens: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ref)
        self.model = AutoModel.from_pretrained(self.model_ref)
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size: int = self.model.config.hidden_size

    def embed(self, text: str) -> tuple[list[float], int]:
        """Embed one text: chunk, forward pass, mean-pool per chunk, mean across chunks.

        Args:
            text: Raw text to embed. Empty string yields a zero vector.

        Returns:
            A ``(vector, n_chunks)`` pair: the pooled embedding as a plain
            ``list[float]`` of length ``hidden_size``, and how many chunks the
            text was split into.
        """
        if not text:
            return [0.0] * self.hidden_size, 0

        token_ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
        chunks = chunk_tokens(token_ids, self.max_length)

        chunk_vectors = []
        with torch.no_grad():
            for chunk in chunks:
                if not chunk:
                    continue
                input_ids = torch.tensor([chunk], device=self.device)
                outputs = self.model(input_ids, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                chunk_vectors.append(last_hidden.mean(dim=1).squeeze(0))

        if not chunk_vectors:
            return [0.0] * self.hidden_size, 0

        pooled = torch.stack(chunk_vectors, dim=0).mean(dim=0)
        return pooled.cpu().tolist(), len(chunk_vectors)


class SentenceTransformerEmbedder:
    """Loads one sentence-transformers model and embeds each text, truncated or chunked.

    Sentence encoders already pool their token states into one vector. By default
    (``chunking=False``) a text longer than the model's ``max_seq_length`` is truncated by the
    model's own tokenizer. With ``chunking=True`` the whole text is embedded: its tokens are
    split into chunks that fit the model, each chunk is encoded, and the chunk vectors are
    averaged weighted by their token count. That weighted mean is the mean over all tokens of
    the text, and a text that fits one chunk gives the same vector as ``chunking=False``
    (up to padding noise). ``HfMeanPoolEmbedder`` instead averages chunks without weights. Vectors are returned as
    the model emits them (not L2-normalised); the evaluators normalise before scoring.
    ``multi-qa-*-dot-*`` models are trained for dot-product scoring, so the raw norm carries
    information the cosine evaluators discard.

    Attributes:
        model_ref: Hub id or local path passed to ``SentenceTransformer``.
        batch_size: Texts encoded per forward pass in ``embed_batch``.
        device: Torch device string the model runs on.
        chunking: Embed the whole text as token-weighted chunk vectors instead of truncating.
    """

    def __init__(self, model_ref: str, batch_size: int = 32, device: str = "cpu", chunking: bool = False) -> None:
        # Imported here so the heavy dependency is only needed when this kind is used.
        from sentence_transformers import SentenceTransformer

        self.model_ref = model_ref
        self.batch_size = batch_size
        self.device = device
        self.chunking = chunking
        self.model = SentenceTransformer(model_ref, device=device)
        self.model.eval()
        # ``get_sentence_embedding_dimension`` was renamed in sentence-transformers 6; accept both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.hidden_size: int = get_dim()
        self.max_seq_length: int = self.model.max_seq_length

    def embed(self, text: str) -> tuple[list[float], int]:
        """Embed one text.

        Args:
            text: Raw text. Empty string yields a zero vector.

        Returns:
            A ``(vector, n_chunks)`` pair with ``n_chunks`` always 1 (0 for empty text).
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[tuple[list[float], int]]:
        """Embed several texts in batches of ``batch_size``.

        Args:
            texts: Raw texts; empty strings yield zero vectors without being encoded.

        Returns:
            One ``(vector, n_chunks)`` pair per input, in input order. ``n_chunks`` is 1 for a
            truncated text and the number of token chunks when ``chunking`` is on.
        """
        non_empty = [i for i, t in enumerate(texts) if t]
        out: list[tuple[list[float], int]] = [([0.0] * self.hidden_size, 0) for _ in texts]
        if non_empty and self.chunking:
            for i, result in zip(non_empty, self._embed_chunked([texts[i] for i in non_empty]), strict=True):
                out[i] = result
        elif non_empty:
            vectors = self.model.encode(
                [texts[i] for i in non_empty],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            for i, vec in zip(non_empty, vectors, strict=True):
                out[i] = (vec.astype("float32").tolist(), 1)
        return out

    def _embed_chunked(self, texts: list[str]) -> list[tuple[list[float], int]]:
        """Embed non-empty texts as token-weighted means of chunk vectors.

        Chunks are fed to the model as token ids (special tokens added), not decoded back to
        text, so a chunk never exceeds ``max_seq_length`` after re-tokenisation.

        Args:
            texts: Non-empty raw texts.

        Returns:
            One ``(vector, n_chunks)`` pair per text, in input order.
        """
        import torch

        tok = self.model.tokenizer
        # Special-token wrapper read off the tokenizer itself (``build_inputs_with_special_tokens`` is gone in
        # transformers 5): encode one token with and without special tokens and split around it.
        probe = tok("a", add_special_tokens=False, verbose=False)["input_ids"]
        wrapped = tok("a", add_special_tokens=True, verbose=False)["input_ids"]
        at = wrapped.index(probe[0])
        prefix, suffix = wrapped[:at], wrapped[at + len(probe) :]
        size = self.max_seq_length - len(prefix) - len(suffix)
        chunks: list[list[int]] = []
        owner: list[int] = []
        for i, text in enumerate(texts):
            ids = tok(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            for start in range(0, max(len(ids), 1), size):
                chunks.append(ids[start : start + size])
                owner.append(i)

        vectors = np.zeros((len(chunks), self.hidden_size), dtype=np.float64)
        order = sorted(range(len(chunks)), key=lambda j: len(chunks[j]))  # similar lengths per batch: less padding
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            seqs = [prefix + chunks[j] + suffix for j in batch]
            width = max(len(q) for q in seqs)
            ids_t = torch.tensor([q + [tok.pad_token_id] * (width - len(q)) for q in seqs], device=self.device)
            mask = torch.tensor([[1] * len(q) + [0] * (width - len(q)) for q in seqs], device=self.device)
            with torch.no_grad():
                emb = self.model({"input_ids": ids_t, "attention_mask": mask})["sentence_embedding"]
            vectors[batch] = emb.float().cpu().numpy()

        sums = np.zeros((len(texts), self.hidden_size))
        weight = np.zeros(len(texts))
        count = np.zeros(len(texts), dtype=int)
        for j, i in enumerate(owner):
            w = max(len(chunks[j]), 1)
            sums[i] += w * vectors[j]
            weight[i] += w
            count[i] += 1
        return [((sums[i] / weight[i]).astype("float32").tolist(), int(count[i])) for i in range(len(texts))]
