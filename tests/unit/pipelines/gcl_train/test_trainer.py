"""Checkpoint universe_sha256 guard and basic training-loop behavior."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from cga.pipelines.gcl_train.models import build_dual_encoder
from cga.pipelines.gcl_train.trainer import ExampleTensors, load_checkpoint, sha256_of_universe, train_model


def test_sha256_of_universe_deterministic_and_order_sensitive() -> None:
    a = sha256_of_universe(["x:1", "x:2", "x:3"])
    b = sha256_of_universe(["x:1", "x:2", "x:3"])
    c = sha256_of_universe(["x:2", "x:1", "x:3"])

    assert a == b
    assert a != c


def test_load_checkpoint_refuses_mismatched_universe(tmp_path: Path) -> None:
    n_nodes = 4
    model = build_dual_encoder(text_dim=3, struct_dim=3, encoder="gcn", hidden_dim=4, out_dim=4)
    text_features = torch.randn(n_nodes, 3)
    struct_features = torch.randn(n_nodes, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    examples = ExampleTensors(
        anchor=torch.tensor([0]),
        text_pos=torch.tensor([1]),
        text_neg=torch.tensor([2]),
        graph_pos=torch.tensor([1]),
        graph_neg=torch.tensor([3]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger = logging.getLogger("test")

    train_model(
        model=model,
        text_features=text_features,
        struct_features=struct_features,
        edge_index=edge_index,
        train_examples=examples,
        loss_name="triplet",
        loss_kwargs={"margin": 1.0},
        alpha=0.5,
        optimizer=optimizer,
        epochs=1,
        checkpoint_every=1,
        checkpoint_dir=tmp_path / "checkpoints",
        universe_sha256="abc123",
        metrics_path=tmp_path / "metrics.jsonl",
        logger=logger,
        log_every=1,
    )

    checkpoint_path = tmp_path / "checkpoints" / "epoch_0001.pt"
    assert checkpoint_path.is_file()

    load_checkpoint(checkpoint_path, universe_sha256="abc123")  # matching hash: no raise

    with pytest.raises(ValueError):
        load_checkpoint(checkpoint_path, universe_sha256="different-hash")


def test_train_model_writes_one_metrics_line_per_epoch(tmp_path: Path) -> None:
    n_nodes = 4
    model = build_dual_encoder(text_dim=3, struct_dim=3, encoder="gcn", hidden_dim=4, out_dim=4)
    text_features = torch.randn(n_nodes, 3)
    struct_features = torch.randn(n_nodes, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    examples = ExampleTensors(
        anchor=torch.tensor([0]),
        text_pos=torch.tensor([1]),
        text_neg=torch.tensor([2]),
        graph_pos=torch.tensor([1]),
        graph_neg=torch.tensor([3]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger = logging.getLogger("test")
    metrics_path = tmp_path / "metrics.jsonl"

    final_embeddings = train_model(
        model=model,
        text_features=text_features,
        struct_features=struct_features,
        edge_index=edge_index,
        train_examples=examples,
        loss_name="triplet",
        loss_kwargs={"margin": 1.0},
        alpha=0.5,
        optimizer=optimizer,
        epochs=5,
        checkpoint_every=0,
        checkpoint_dir=tmp_path / "checkpoints",
        universe_sha256="abc123",
        metrics_path=metrics_path,
        logger=logger,
        log_every=100,
    )

    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    assert final_embeddings.shape == (n_nodes, 4)
    # checkpoint_every=0 still checkpoints the final epoch.
    assert (tmp_path / "checkpoints" / "epoch_0005.pt").is_file()
    assert not (tmp_path / "checkpoints" / "epoch_0001.pt").is_file()
