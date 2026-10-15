"""Shape and construction checks for DualEncoder / GCN / GAT towers."""

from __future__ import annotations

import pytest
import torch

from cga.pipelines.gcl_train.models import build_dual_encoder


@pytest.mark.parametrize("encoder", ["gcn", "gat"])
def test_dual_encoder_output_shape(encoder: str) -> None:
    n_nodes = 6
    text_dim, struct_dim = 8, 4
    hidden_dim, out_dim = 16, 16

    model = build_dual_encoder(
        text_dim=text_dim, struct_dim=struct_dim, encoder=encoder, hidden_dim=hidden_dim, out_dim=out_dim
    )

    text_features = torch.randn(n_nodes, text_dim)
    struct_features = torch.randn(n_nodes, struct_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

    output = model(text_features, struct_features, edge_index)

    assert output.shape == (n_nodes, hidden_dim)


def test_build_dual_encoder_rejects_unknown_encoder() -> None:
    with pytest.raises(ValueError):
        build_dual_encoder(text_dim=4, struct_dim=4, encoder="mlp", hidden_dim=8, out_dim=8)


def test_dual_encoder_handles_no_edges() -> None:
    n_nodes = 3
    model = build_dual_encoder(text_dim=4, struct_dim=4, encoder="gcn", hidden_dim=8, out_dim=8)
    text_features = torch.randn(n_nodes, 4)
    struct_features = torch.randn(n_nodes, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    output = model(text_features, struct_features, edge_index)

    assert output.shape == (n_nodes, 8)


# ---- ResidualAdapterEncoder (SRGA) -------------------------------------------------------------

from cga.pipelines.gcl_train.models import build_residual_adapter_encoder  # noqa: E402


def _chain_fixture():
    """3 nodes: 0 = child, 1 = middle, 2 = parent (stored edges child->parent: 0->1, 1->2)."""
    text = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    stored = [(0, 1), (1, 2)]
    pairs, types = [], []
    for s, d in stored:
        pairs += [(s, d), (d, s)]
        types += [0, 1]
    return text, torch.tensor(pairs).t().contiguous(), torch.tensor(types)


def test_residual_adapter_lambda_zero_keeps_unit_text_block_and_zero_residual():
    text, edge_index, edge_type = _chain_fixture()
    model = build_residual_adapter_encoder(lam=0.0)
    z = model(text * 3.0, text, edge_index, edge_type)
    assert z.shape == (3, 3 + 3)
    assert torch.allclose(z[:, :3], text)  # normalized, unchanged direction
    assert torch.count_nonzero(z[:, 3:]) == 0


def test_residual_adapter_output_has_text_plus_h_columns():
    text, edge_index, edge_type = _chain_fixture()
    undirected = build_residual_adapter_encoder(lam=0.25)(text, text, edge_index, edge_type)
    directional = build_residual_adapter_encoder(lam=0.25, directional=True)(text, text, edge_index, edge_type)
    assert undirected.shape[1] == 3 + 3
    assert directional.shape[1] == 3 + 6


def test_residual_adapter_undirected_h_is_neighbour_mean():
    text, edge_index, edge_type = _chain_fixture()
    z = build_residual_adapter_encoder(lam=1.0)(text, text, edge_index, edge_type)
    # Node 1 has neighbours 0 and 2 -> mean (0.5, 0, 0.5) -> unit (1, 0, 1)/sqrt2.
    assert torch.allclose(z[1, 3:], torch.tensor([1.0, 0.0, 1.0]) / 2**0.5)
    # Node 0's only neighbour is node 1.
    assert torch.allclose(z[0, 3:], torch.tensor([0.0, 1.0, 0.0]))


def test_residual_adapter_directional_uses_parent_and_child_sides_correctly():
    text, edge_index, edge_type = _chain_fixture()
    z = build_residual_adapter_encoder(lam=1.0, directional=True)(text, text, edge_index, edge_type)
    h = z[:, 3:]  # [mean parents (3) ; mean children (3)], unit-normalized as a whole
    # Node 1: parent = node 2, child = node 0.
    assert torch.allclose(h[1], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) / 2**0.5)
    # Node 0 (leaf): parent = node 1, no children.
    assert torch.allclose(h[0], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    # Node 2 (root): no parents, child = node 1.
    assert torch.allclose(h[2], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_residual_adapter_isolated_node_gets_zero_residual_and_duplicate_edges_collapse():
    text = torch.eye(3)
    edge_index = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])  # pair 0-1 stored twice
    edge_type = torch.tensor([0, 1, 0, 1])
    z = build_residual_adapter_encoder(lam=1.0)(text, text, edge_index, edge_type)
    assert torch.count_nonzero(z[2, 3:]) == 0
    assert torch.allclose(z[0, 3:], torch.tensor([0.0, 1.0, 0.0]))


def test_residual_adapter_has_no_trainable_params_and_rejects_bad_config():
    model = build_residual_adapter_encoder(lam=0.5)
    assert sum(p.numel() for p in model.parameters()) == 0
    with pytest.raises(ValueError):
        build_residual_adapter_encoder(adapter="relational")
    with pytest.raises(ValueError):
        build_residual_adapter_encoder(adapter="nope")
    with pytest.raises(ValueError):
        build_residual_adapter_encoder(lam=-1.0)
    with pytest.raises(ValueError):
        build_residual_adapter_encoder(lam=0.5, directional=True)(
            torch.eye(2), torch.eye(2), torch.empty((2, 0), dtype=torch.long)
        )


def test_mlp_adapter_shapes_params_and_small_last_layer_init():
    model = build_residual_adapter_encoder(adapter="mlp", lam=0.25, adapter_dim=16, text_dim=8)
    x = torch.randn(5, 8)
    z = model(x, x, torch.empty((2, 0), dtype=torch.long))
    assert z.shape == (5, 8 + 16)
    assert torch.allclose(z[:, :8].norm(dim=1), torch.ones(5), atol=1e-5)  # SapBERT block untouched (unit)
    assert torch.allclose(z[:, 8:].norm(dim=1), torch.full((5,), 0.25**0.5), atol=1e-5)
    last = model.mlp[-1]
    assert last.weight.std().item() < 5e-3 and torch.count_nonzero(last.bias) == 0
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0


def test_mlp_adapter_ignores_edges_and_dropout_only_in_train_mode():
    model = build_residual_adapter_encoder(adapter="mlp", lam=1.0, adapter_dim=16, text_dim=8, adapter_dropout=0.5)
    x = torch.randn(6, 8)
    no_edges = torch.empty((2, 0), dtype=torch.long)
    some_edges = torch.tensor([[0, 1], [1, 0]])
    model.eval()
    assert torch.equal(model(x, x, no_edges), model(x, x, some_edges, torch.tensor([0, 1])))
    assert torch.equal(model(x, x, no_edges), model(x, x, no_edges))  # deterministic in eval


def test_mlp_adapter_lambda_zero_gives_zero_gradient_on_adapter():
    model = build_residual_adapter_encoder(adapter="mlp", lam=0.0, adapter_dim=16, text_dim=8)
    x = torch.randn(6, 8)
    z = model(x, x, torch.empty((2, 0), dtype=torch.long))
    z.pow(2).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    assert torch.isfinite(z).all()
