"""Checkpoint universe_sha256 guard and basic training-loop behavior."""

from __future__ import annotations

import logging
import math
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


def test_train_model_infonce_full_decreases_loss_and_requires_full_data(tmp_path: Path) -> None:
    import json

    from cga.pipelines.gcl_train.losses import build_full_softmax_data
    from cga.pipelines.gcl_train.models import build_residual_adapter_encoder

    torch.manual_seed(0)
    n_src, n_tgt = 6, 8
    text = torch.randn(n_src + n_tgt, 8)
    roles = ["g1"] * n_src + ["g2"] * n_tgt
    pairs = [(i, n_src + i) for i in range(n_src)]
    data = build_full_softmax_data(pairs, graph_roles=roles, held_out=[False] * len(roles), gold_pairs=pairs)
    examples = ExampleTensors(*(torch.zeros(1, dtype=torch.long) for _ in range(5)))
    model = build_residual_adapter_encoder(adapter="mlp", lam=4.0, adapter_dim=16, text_dim=8, adapter_dropout=0.0)
    kwargs = dict(
        model=model,
        text_features=text,
        struct_features=text,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        train_examples=examples,
        loss_kwargs={"temperature": 0.1, "reverse_weight": 0.5},
        alpha=0.5,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        epochs=40,
        checkpoint_every=0,
        checkpoint_dir=tmp_path / "ck",
        universe_sha256="x",
        metrics_path=tmp_path / "metrics.jsonl",
        logger=logging.getLogger("test"),
        log_every=100,
    )
    with pytest.raises(ValueError):
        train_model(loss_name="infonce_full", **kwargs)

    out = train_model(loss_name="infonce_full", full_data=data, **kwargs)

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 40
    assert all(math.isfinite(r["loss"]) for r in rows)
    assert rows[-1]["loss"] < rows[0]["loss"]
    assert out.shape == (n_src + n_tgt, 8 + 16)


def _run_inbatch(tmp_path: Path, text_neg, graph_neg) -> tuple[list[float], torch.Tensor]:
    import json

    torch.manual_seed(0)
    model = build_dual_encoder(text_dim=3, struct_dim=3, encoder="gcn", hidden_dim=4, out_dim=4)
    text_features, struct_features = torch.randn(6, 3), torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]], dtype=torch.long)
    examples = ExampleTensors(
        anchor=torch.tensor([0, 0, 1, 4]),
        text_pos=torch.tensor([3, 4, 5, 1]),
        text_neg=text_neg,
        graph_pos=torch.tensor([1, 2, 0, 5]),
        graph_neg=graph_neg,
    )
    final = train_model(
        model=model,
        text_features=text_features,
        struct_features=struct_features,
        edge_index=edge_index,
        train_examples=examples,
        loss_name="infonce_inbatch",
        loss_kwargs={"temperature": 0.1},
        alpha=0.5,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        epochs=3,
        checkpoint_every=0,
        checkpoint_dir=tmp_path / "ckpt",
        universe_sha256="x",
        metrics_path=tmp_path / "metrics.jsonl",
        logger=logging.getLogger("test"),
        node_roles=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    losses = [json.loads(line)["loss"] for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    return losses, final


def test_infonce_inbatch_training_never_reads_the_negative_columns(tmp_path: Path) -> None:
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(), b_dir.mkdir()
    losses_a, emb_a = _run_inbatch(a_dir, torch.tensor([5, 3, 3, 0]), torch.tensor([2, 1, 2, 3]))
    losses_b, emb_b = _run_inbatch(b_dir, torch.tensor([4, 4, 4, 4]), torch.tensor([0, 0, 0, 0]))
    assert losses_a == losses_b and all(math.isfinite(x) for x in losses_a)
    assert torch.equal(emb_a, emb_b)
    # and it is reproducible
    c_dir = tmp_path / "c"
    c_dir.mkdir()
    losses_c, emb_c = _run_inbatch(c_dir, torch.tensor([5, 3, 3, 0]), torch.tensor([2, 1, 2, 3]))
    assert losses_c == losses_a and torch.equal(emb_c, emb_a)


def test_infonce_inbatch_requires_node_roles(tmp_path: Path) -> None:
    model = build_dual_encoder(text_dim=3, struct_dim=3, encoder="gcn", hidden_dim=4, out_dim=4)
    examples = ExampleTensors(*(torch.tensor([0]) for _ in range(5)))
    with pytest.raises(ValueError, match="node_roles"):
        train_model(
            model=model,
            text_features=torch.randn(2, 3),
            struct_features=torch.randn(2, 3),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            train_examples=examples,
            loss_name="infonce_inbatch",
            loss_kwargs={"temperature": 0.1},
            alpha=0.5,
            optimizer=torch.optim.Adam(model.parameters()),
            epochs=1,
            checkpoint_every=0,
            checkpoint_dir=tmp_path / "ckpt",
            universe_sha256="x",
            metrics_path=tmp_path / "metrics.jsonl",
            logger=logging.getLogger("test"),
        )
