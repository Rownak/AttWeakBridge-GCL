"""Full-batch training loop for the ``DualEncoder`` (target_architecture.md §5.4 step 2).

One forward pass per epoch computes fused embeddings for **every** node in the
universe (transductive message passing needs held-out nodes' structure/text
features, and ``evaluate`` later needs their embeddings), but the loss —
``alpha * L_text + (1 - alpha) * L_graph`` — only indexes rows for the
train-split examples in ``examples.parquet``; held-out nodes are never an
anchor/positive/negative there (enforced upstream by ``training_data``'s
leakage check), so they receive no direct gradient signal, only the indirect
effect of participating in neighbors' message passing — standard transductive
GNN practice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cga.core.io import write_json
from cga.core.registry import losses as loss_registry
from cga.pipelines.gcl_train.losses import FullSoftmaxData
from cga.pipelines.gcl_train.models import DualEncoder, ResidualAdapterEncoder


@dataclass
class ExampleTensors:
    """Row-index tensors into the universe-ordered embedding matrix, one row per training example.

    Attributes:
        anchor: Anchor row indices.
        text_pos: Text-positive row indices (cross-graph).
        text_neg: Text-negative row indices (cross-graph).
        graph_pos: Graph-positive row indices (within-graph).
        graph_neg: Graph-negative row indices (within-graph).
    """

    anchor: torch.Tensor
    text_pos: torch.Tensor
    text_neg: torch.Tensor
    graph_pos: torch.Tensor
    graph_neg: torch.Tensor

    def __len__(self) -> int:
        return self.anchor.shape[0]


def train_model(
    *,
    model: DualEncoder | ResidualAdapterEncoder,
    text_features: torch.Tensor,
    struct_features: torch.Tensor,
    edge_index: torch.Tensor,
    train_examples: ExampleTensors,
    loss_name: str,
    loss_kwargs: dict[str, float],
    alpha: float,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    checkpoint_every: int,
    checkpoint_dir: Path,
    universe_sha256: str,
    metrics_path: Path,
    logger: Any,
    log_every: int = 10,
    edge_type: torch.Tensor | None = None,
    full_data: FullSoftmaxData | None = None,
) -> torch.Tensor:
    """Run the full-batch training loop, writing per-epoch metrics and periodic checkpoints.

    Args:
        model: The ``DualEncoder`` to train, already on the target device.
        text_features: ``(n_universe, text_dim)`` text embeddings, row-aligned
            to the universe.
        struct_features: ``(n_universe, struct_dim)`` structural embeddings,
            row-aligned to the universe.
        edge_index: ``(2, n_edges)`` edge index remapped to universe rows.
        train_examples: Row indices for train-split examples only.
        loss_name: Registered loss name (``"triplet"`` or ``"infonce"``).
        loss_kwargs: Extra kwargs forwarded to the loss (``margin`` or
            ``temperature``).
        alpha: Text-view loss weight; graph view gets ``1 - alpha``.
        optimizer: Optimizer already bound to ``model.parameters()``.
        epochs: Number of full-batch epochs.
        checkpoint_every: Write a checkpoint every this many epochs (0 disables
            periodic checkpoints; the final epoch is always checkpointed).
        checkpoint_dir: Directory checkpoints are written into.
        universe_sha256: Hash of the universe this run trains on, embedded in
            every checkpoint so a mismatched universe is refused on load.
        metrics_path: Path to append one JSON object per epoch to
            (``metrics.jsonl``).
        logger: Logger for per-``log_every``-epoch progress lines.
        log_every: How often (in epochs) to log progress.
        edge_type: Optional ``(n_edges,)`` edge types forwarded to the model.
        full_data: Required when ``loss_name == "infonce_full"``. That loss replaces both
            views (``alpha`` is ignored); ``metrics.jsonl`` then records the forward term as
            ``loss_text`` and the reverse term as ``loss_graph``.

    Returns:
        The final fused embeddings for every universe row, detached.
    """
    loss_fn = loss_registry.get(loss_name)
    if loss_name == "infonce_full" and full_data is None:
        raise ValueError("loss 'infonce_full' needs full_data")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, epochs + 1):
            start = time.time()
            model.train()
            optimizer.zero_grad()

            embeddings = model(text_features, struct_features, edge_index, edge_type)

            if loss_name == "infonce_full":
                loss, loss_text, loss_graph = loss_fn(embeddings, full_data, **loss_kwargs)
            else:
                loss_text = loss_fn(
                    embeddings[train_examples.anchor],
                    embeddings[train_examples.text_pos],
                    embeddings[train_examples.text_neg],
                    **loss_kwargs,
                )
                loss_graph = loss_fn(
                    embeddings[train_examples.anchor],
                    embeddings[train_examples.graph_pos],
                    embeddings[train_examples.graph_neg],
                    **loss_kwargs,
                )
                loss = alpha * loss_text + (1 - alpha) * loss_graph

            loss.backward()
            optimizer.step()

            elapsed_s = time.time() - start
            record = {
                "epoch": epoch,
                "loss": loss.item(),
                "loss_text": loss_text.item(),
                "loss_graph": loss_graph.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": elapsed_s,
            }
            metrics_file.write(_json_line(record))
            metrics_file.flush()

            if epoch % log_every == 0 or epoch == epochs:
                logger.info(
                    "epoch %d/%d: loss=%.4f loss_text=%.4f loss_graph=%.4f",
                    epoch,
                    epochs,
                    loss.item(),
                    loss_text.item(),
                    loss_graph.item(),
                )

            is_last = epoch == epochs
            if is_last or (checkpoint_every > 0 and epoch % checkpoint_every == 0):
                _write_checkpoint(model, optimizer, epoch, universe_sha256, checkpoint_dir)

    model.eval()
    with torch.no_grad():
        final_embeddings = model(text_features, struct_features, edge_index, edge_type)
    return final_embeddings.detach()


def _json_line(record: dict[str, Any]) -> str:
    import json

    return json.dumps(record, sort_keys=True) + "\n"


def _write_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, universe_sha256: str, checkpoint_dir: Path
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "universe_sha256": universe_sha256,
        },
        checkpoint_dir / f"epoch_{epoch:04d}.pt",
    )


def load_checkpoint(path: Path, *, universe_sha256: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint, refusing it if it was trained against a different universe.

    Raises:
        ValueError: If the checkpoint's recorded ``universe_sha256`` does not
            match ``universe_sha256``.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("universe_sha256") != universe_sha256:
        raise ValueError(
            f"Checkpoint '{path}' was trained against universe_sha256="
            f"{checkpoint.get('universe_sha256')!r}, but this run's universe is "
            f"{universe_sha256!r}"
        )
    return checkpoint


def sha256_of_universe(node_ids: list[str]) -> str:
    """Return the sha256 hex digest of the universe's ordered node id list."""
    import hashlib

    payload = "\n".join(node_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
