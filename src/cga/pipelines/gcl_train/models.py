"""``DualEncoder`` model: two GNN towers over text and structural node features.

Ported from ``legacy/GCL/models/GNN_dual_loss2.py`` (``GCN``, ``GAT``,
``DualEncoder``), generalized to be dataset-agnostic: nothing here references
ATT&CK/CWE or any dataset-specific dimension. Both towers run full-batch,
transductive message passing over the single ``edge_index`` built from
``edges.parquet`` (intra-graph edges only, so ``graph_1`` and ``graph_2`` are
disjoint connected components and a GCN/GAT layer never mixes information
across the pair) — this covers every node in the universe, including
``held_out`` ones, because message passing needs their structure/text
features and ``evaluate`` needs their embeddings; only the loss (in
:mod:`cga.pipelines.gcl_train.losses`) restricts itself to train-split
example rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv


class GCN(nn.Module):
    """Two-layer GCN tower: ``in_channels -> hid_dim -> out_channels``."""

    def __init__(self, in_channels: int, out_channels: int, hid_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    """Two-layer GAT tower: multi-head attention then a single-head projection.

    Args:
        heads: ``(first_layer_heads, second_layer_heads)``. The first layer's
            output width is ``hid_dim * heads[0]`` (concatenated heads); the
            second layer collapses back to ``out_channels`` with ``heads[1]``
            heads (1 by default, matching legacy behavior).
    """

    def __init__(
        self, in_channels: int, out_channels: int, hid_dim: int, heads: tuple[int, int] = (8, 1)
    ) -> None:
        super().__init__()
        first_heads, second_heads = heads
        self.gat1 = GATConv(in_channels, hid_dim, heads=first_heads)
        self.gat2 = GATConv(hid_dim * first_heads, out_channels, heads=second_heads, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x


class DualEncoder(nn.Module):
    """Fuses a text-feature GNN tower and a structural-feature GNN tower into one embedding.

    Args:
        graph_model1: Tower consuming text embeddings as node features.
        graph_model2: Tower consuming structural embeddings as node features.
        text_dim: Input text-embedding dimension.
        struct_dim: Input structural-embedding dimension.
        out_channels: Width of the projected text branch (``fc1``) and of
            ``graph_model1``'s output.
        hidden_dim: Final fused embedding dimension.
    """

    def __init__(
        self,
        graph_model1: nn.Module,
        graph_model2: nn.Module,
        *,
        text_dim: int,
        struct_dim: int,
        out_channels: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.graph_model1 = graph_model1
        self.graph_model2 = graph_model2
        self.fc1 = nn.Linear(text_dim, out_channels)
        self.fc2 = nn.Linear(struct_dim, struct_dim)
        self.fc3 = nn.Linear(2 * out_channels + 2 * struct_dim, hidden_dim)

    def forward(
        self, text_emb: torch.Tensor, struct_emb: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        graph_emb1 = self.graph_model1(text_emb, edge_index)
        graph_emb2 = self.graph_model2(struct_emb, edge_index)
        text_proj = self.fc1(text_emb)
        struct_proj = self.fc2(struct_emb)
        combined = torch.cat([text_proj, graph_emb1, struct_proj, graph_emb2], dim=1)
        return self.fc3(combined)


def build_dual_encoder(
    *,
    text_dim: int,
    struct_dim: int,
    encoder: str,
    hidden_dim: int,
    out_dim: int,
    gat_heads: tuple[int, int] = (8, 1),
) -> DualEncoder:
    """Construct a :class:`DualEncoder` with matching GCN or GAT towers.

    Args:
        text_dim: Input text-embedding dimension.
        struct_dim: Input structural-embedding dimension.
        encoder: ``"gcn"`` or ``"gat"``.
        hidden_dim: Hidden width for each tower and the final fused embedding
            dimension.
        out_dim: Output dimension of the text-branch tower and the ``fc1``
            projection; the structural branch keeps its native ``struct_dim``
            (mirroring the legacy ``feature_2`` tower, which projects
            ``struct_dim -> struct_dim`` rather than to a shared ``out_dim``).
        gat_heads: Attention heads for the GAT towers; ignored for GCN.

    Raises:
        ValueError: If ``encoder`` is not ``"gcn"`` or ``"gat"``.
    """
    if encoder == "gcn":
        graph_model1: nn.Module = GCN(in_channels=text_dim, out_channels=out_dim, hid_dim=hidden_dim)
        graph_model2: nn.Module = GCN(in_channels=struct_dim, out_channels=struct_dim, hid_dim=struct_dim)
    elif encoder == "gat":
        graph_model1 = GAT(in_channels=text_dim, out_channels=out_dim, hid_dim=hidden_dim, heads=gat_heads)
        graph_model2 = GAT(in_channels=struct_dim, out_channels=struct_dim, hid_dim=struct_dim, heads=gat_heads)
    else:
        raise ValueError(f"Unknown encoder '{encoder}'; expected 'gcn' or 'gat'")

    return DualEncoder(
        graph_model1,
        graph_model2,
        text_dim=text_dim,
        struct_dim=struct_dim,
        out_channels=out_dim,
        hidden_dim=hidden_dim,
    )
