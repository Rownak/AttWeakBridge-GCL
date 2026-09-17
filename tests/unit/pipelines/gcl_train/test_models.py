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
