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
        self,
        text_emb: torch.Tensor,
        struct_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # edge_type is accepted for interface parity with ResidualAdapterEncoder and ignored here.
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


_ADAPTERS = ("neighbor_mean", "mlp", "relational")
_PARENT_EDGE_TYPE = 1  # reversed copy: message flows parent -> child, so the target's neighbour is a parent
_CHILD_EDGE_TYPE = 0  # stored direction child -> parent: the target (parent) receives its children


def _unit(x: torch.Tensor) -> torch.Tensor:
    """Row-normalize ``x`` to unit L2 norm; all-zero rows stay zero."""
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _mean_over_incoming(
    x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None, wanted_type: int | None
) -> torch.Tensor:
    """Mean of ``x`` over each node's incoming neighbours (zeros when it has none).

    Thin wrapper over :func:`_mean_and_count_over_incoming` that drops the counts.
    """
    return _mean_and_count_over_incoming(x, edge_index, edge_type, wanted_type)[0]


def _mean_and_count_over_incoming(
    x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None, wanted_type: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of ``x`` over each node's incoming neighbours, plus the neighbour counts.

    Args:
        x: ``(n, d)`` source features.
        edge_index: ``(2, e)`` message edges, row 0 = source, row 1 = target.
        edge_type: ``(e,)`` edge types, or ``None`` when ``wanted_type`` is ``None``.
        wanted_type: Keep only edges of this type; ``None`` keeps all edges.

    Returns:
        ``(mean, count)``: ``(n, d)`` neighbour means (zeros without neighbours) and ``(n,)``
        unique-neighbour counts. Duplicate edges are collapsed first so a pair stored under
        two ``edge_sources`` is not counted twice.
    """
    src, dst = edge_index
    if wanted_type is not None:
        keep = edge_type == wanted_type
        src, dst = src[keep], dst[keep]
    n = x.shape[0]
    unique = torch.unique(src * n + dst)
    src, dst = unique // n, unique % n
    total = torch.zeros_like(x).index_add_(0, dst, x[src])
    count = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(
        0, dst, torch.ones(dst.shape[0], device=x.device, dtype=x.dtype)
    )
    return total / count.clamp_min(1.0).unsqueeze(1), count


class ResidualAdapterEncoder(nn.Module):
    """SapBERT-residual graph adapter (SRGA): ``z = [x̂ ; sqrt(lambda) * normalize(h)]``.

    Keeps the frozen text vector as an untouched block and adds a graph correction
    beside it, so ``cos(z_a, z_d)`` ranks by ``cos(x̂_a, x̂_d) + lambda * cos(h_a, h_d)``
    up to per-node norms, and ``lambda = 0`` ranks exactly like the text vector
    (design_docs/sapbert_residual_graph_adapter.md §5.1). It sits in ``gcl_train``
    beside :class:`DualEncoder`, selected by ``gcl_train.model.encoder: residual_adapter``;
    ``gold_alignment`` scores the exported ``z`` unchanged.

    Adapters:
        ``neighbor_mean``: parameter-free, ``h`` = mean of ``x̂`` over graph neighbours (Phase 5.1).
        ``mlp``: trainable, **no graph input**, ``h = MLP(x̂)`` with
        ``Linear(d, a) -> ReLU -> LayerNorm(a) -> Dropout -> Linear(a, a)`` (Phase 5.2). The last
        layer starts at std 1e-3 / zero bias, so ``h`` is a small, smooth function of ``x̂`` at
        initialization (``normalize(h)`` is still unit length, so ``z`` is SapBERT plus a
        ``sqrt(lambda)``-scaled, input-dependent block, not a small perturbation).
        ``relational``: trainable directional graph adapter (Phase 5.3), see below.

    Relational adapter (``directional=True``; ``a`` = ``adapter_dim``, ``d`` = text width)::

        a_n = [ x̂_n ; p_n ; c_n ; x̂_n ⊙ p_n ; log(1+n_children) ; log(1+n_parents) ]   (4d + 2)
              p_n = mean of x̂ over parents,  c_n = mean of x̂ over children
        u_n = W_in a_n + b_in                                                            (a)
        r_n = W_par · mean_{parents(n)} u  +  W_chi · mean_{children(n)} u               (a)
        h_n = MLP_2( LayerNorm( u_n + r_n ) )                       MLP_2 = Linear-ReLU-Dropout-Linear

    ``W_par`` and ``W_chi`` are the separate parent/child weights of the one relational layer. The
    input projection ``W_in`` comes first so the residual ``u_n + r_n`` has matching width (a raw
    ``4d + 2`` residual would need ~9M extra weights). With ``directional=False`` the control A6
    collapses it to one undirected relation: ``a_n = [x̂_n ; m_n ; x̂_n ⊙ m_n ; log(1+deg)]``
    (``3d + 1``, ``m_n`` = mean over all neighbours) and one weight ``W_nbr``. The last ``MLP_2``
    layer starts at std 1e-3 like the Phase 5.2 adapter.

    Args:
        adapter: ``neighbor_mean``, ``mlp`` or ``relational``.
        lam: Fixed global residual weight ``lambda`` (``>= 0``).
        adapter_dim: Hidden/output width ``a`` of the trainable adapters (unused by ``neighbor_mean``).
        directional: When true, ``neighbor_mean`` uses ``h = [mean parents ; mean children]`` and
            ``relational`` uses separate parent/child weights; false collapses both to one undirected relation.
        text_dim: Width of the text input ``x̂`` (needed to size the trainable adapters).
        adapter_dropout: Dropout inside the trainable adapters (train mode only).

    Raises:
        ValueError: On an unknown adapter, a not-yet-implemented adapter, or negative ``lam``.
    """

    def __init__(
        self,
        *,
        adapter: str,
        lam: float,
        adapter_dim: int = 256,
        directional: bool = False,
        text_dim: int = 768,
        adapter_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if adapter not in _ADAPTERS:
            raise ValueError(f"Unknown adapter '{adapter}'; expected one of {list(_ADAPTERS)}")
        if lam < 0:
            raise ValueError(f"lambda must be >= 0, got {lam}")
        self.adapter = adapter
        self.lam = float(lam)
        self.adapter_dim = adapter_dim
        self.directional = directional
        if adapter == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(text_dim, adapter_dim),
                nn.ReLU(),
                nn.LayerNorm(adapter_dim),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_dim, adapter_dim),
            )
            nn.init.normal_(self.mlp[-1].weight, std=1e-3)
            nn.init.zeros_(self.mlp[-1].bias)
        elif adapter == "relational":
            in_dim = 4 * text_dim + 2 if directional else 3 * text_dim + 1
            self.proj = nn.Linear(in_dim, adapter_dim)
            if directional:
                self.w_parent = nn.Linear(adapter_dim, adapter_dim, bias=False)
                self.w_child = nn.Linear(adapter_dim, adapter_dim, bias=False)
            else:
                self.w_neighbor = nn.Linear(adapter_dim, adapter_dim, bias=False)
            self.norm = nn.LayerNorm(adapter_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(adapter_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_dim, adapter_dim),
            )
            nn.init.normal_(self.mlp2[-1].weight, std=1e-3)
            nn.init.zeros_(self.mlp2[-1].bias)

    def _relational_h(self, x_hat: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None):
        """Compute ``h`` for the ``relational`` adapter (equations in the class docstring)."""
        if self.directional:
            if edge_type is None:
                raise ValueError("directional relational adapter needs edge_type")
            parents, n_parents = _mean_and_count_over_incoming(x_hat, edge_index, edge_type, _PARENT_EDGE_TYPE)
            children, n_children = _mean_and_count_over_incoming(x_hat, edge_index, edge_type, _CHILD_EDGE_TYPE)
            a = torch.cat(
                [x_hat, parents, children, x_hat * parents, torch.log1p(n_children)[:, None], torch.log1p(n_parents)[:, None]],
                dim=1,
            )
            u = self.proj(a)
            mean_par = _mean_over_incoming(u, edge_index, edge_type, _PARENT_EDGE_TYPE)
            mean_chi = _mean_over_incoming(u, edge_index, edge_type, _CHILD_EDGE_TYPE)
            r = self.w_parent(mean_par) + self.w_child(mean_chi)
        else:
            neighbors, degree = _mean_and_count_over_incoming(x_hat, edge_index, None, None)
            a = torch.cat([x_hat, neighbors, x_hat * neighbors, torch.log1p(degree)[:, None]], dim=1)
            u = self.proj(a)
            r = self.w_neighbor(_mean_over_incoming(u, edge_index, None, None))
        return self.mlp2(self.norm(u + r))

    def forward(
        self,
        text_emb: torch.Tensor,
        struct_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``z`` for every node.

        Args:
            text_emb: ``(n, 768)`` text embeddings (used frozen).
            struct_emb: Unused; kept so the signature matches :class:`DualEncoder`
                (DeepWalk is not cross-graph comparable and is not an SRGA input).
            edge_index: ``(2, e)`` symmetrized message edges.
            edge_type: ``(e,)`` per-edge type (0 = stored child->parent direction,
                1 = reversed). Required when ``directional``.

        Returns:
            ``(n, d + d_h)`` with ``d_h`` = ``adapter_dim`` (``mlp``, ``relational``), ``d`` (undirected
            ``neighbor_mean``) or ``2d`` (directional); ``d`` is the text width.

        Raises:
            ValueError: If ``directional`` and ``edge_type`` is ``None`` (``neighbor_mean``, ``relational``).
        """
        x_hat = _unit(text_emb.detach())
        if self.adapter == "mlp":
            return torch.cat([x_hat, (self.lam**0.5) * _unit(self.mlp(x_hat))], dim=1)
        if self.adapter == "relational":
            return torch.cat([x_hat, (self.lam**0.5) * _unit(self._relational_h(x_hat, edge_index, edge_type))], dim=1)
        if self.directional:
            if edge_type is None:
                raise ValueError("directional neighbor_mean needs edge_type")
            parents = _mean_over_incoming(x_hat, edge_index, edge_type, _PARENT_EDGE_TYPE)
            children = _mean_over_incoming(x_hat, edge_index, edge_type, _CHILD_EDGE_TYPE)
            h = torch.cat([parents, children], dim=1)
        else:
            h = _mean_over_incoming(x_hat, edge_index, None, None)
        return torch.cat([x_hat, (self.lam**0.5) * _unit(h)], dim=1)


def build_residual_adapter_encoder(
    *,
    adapter: str = "neighbor_mean",
    lam: float = 0.25,
    adapter_dim: int = 256,
    directional: bool = False,
    text_dim: int = 768,
    adapter_dropout: float = 0.2,
) -> ResidualAdapterEncoder:
    """Construct a :class:`ResidualAdapterEncoder` (see its docstring for the arguments)."""
    return ResidualAdapterEncoder(
        adapter=adapter,
        lam=lam,
        adapter_dim=adapter_dim,
        directional=directional,
        text_dim=text_dim,
        adapter_dropout=adapter_dropout,
    )
