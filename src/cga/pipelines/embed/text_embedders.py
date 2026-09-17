"""``hf_mean_pool`` text embedder: tokenize, fixed-size chunk, mean-pool last hidden layer.

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
