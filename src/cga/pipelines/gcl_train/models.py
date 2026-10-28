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
import torch.nn.functional as F
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


_BRANCHES = ("both", "text", "graph")


class DualEncoder(nn.Module):
    """Fuses a text-feature GNN tower and a structural-feature GNN tower into one embedding.

    The default (``branch="both"``, message passing on) is the paper model. Sprint 11.12 ablations
    drop a whole branch (``branch="text"`` / ``"graph"``) or the graph convolutions
    (``message_passing=False``, leaving only the ``fc1`` / ``fc2`` skip projections). Unused
    modules are not created, so the default model has the same parameters in the same creation
    order (and therefore the same seeded initialisation) as before the options existed.

    Args:
        graph_model1: Tower consuming text embeddings as node features (``None`` when unused).
        graph_model2: Tower consuming structural embeddings as node features (``None`` when unused).
        text_dim: Input text-embedding dimension.
        struct_dim: Input structural-embedding dimension.
        out_channels: Width of the projected text branch (``fc1``) and of
            ``graph_model1``'s output.
        hidden_dim: Final fused embedding dimension.
        branch: ``"both"``, ``"text"`` (the structural input is ignored) or ``"graph"`` (the text
            input is ignored).
        message_passing: When false the GNN towers are absent and ``edge_index`` is ignored.

    Raises:
        ValueError: If ``branch`` is unknown.
    """

    def __init__(
        self,
        graph_model1: nn.Module | None,
        graph_model2: nn.Module | None,
        *,
        text_dim: int,
        struct_dim: int,
        out_channels: int,
        hidden_dim: int,
        branch: str = "both",
        message_passing: bool = True,
    ) -> None:
        super().__init__()
        if branch not in _BRANCHES:
            raise ValueError(f"Unknown branch '{branch}'; expected one of {_BRANCHES}")
        self.branch = branch
        self.message_passing = message_passing
        use_text, use_graph = branch in ("both", "text"), branch in ("both", "graph")
        # Registration order (towers, fc1, fc2, fc3) is the original one; keep it for state_dict parity.
        if use_text and message_passing:
            self.graph_model1 = graph_model1
        if use_graph and message_passing:
            self.graph_model2 = graph_model2
        width = 0
        if use_text:
            self.fc1 = nn.Linear(text_dim, out_channels)
            width += out_channels * (2 if message_passing else 1)
        if use_graph:
            self.fc2 = nn.Linear(struct_dim, struct_dim)
            width += struct_dim * (2 if message_passing else 1)
        self.fc3 = nn.Linear(width, hidden_dim)

    def forward(
        self,
        text_emb: torch.Tensor,
        struct_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # edge_type is accepted for interface parity with ResidualAdapterEncoder and ignored here.
        parts: list[torch.Tensor] = []
        if self.branch in ("both", "text"):
            parts.append(self.fc1(text_emb))
            if self.message_passing:
                parts.append(self.graph_model1(text_emb, edge_index))
        if self.branch in ("both", "graph"):
            parts.append(self.fc2(struct_emb))
            if self.message_passing:
                parts.append(self.graph_model2(struct_emb, edge_index))
        return self.fc3(torch.cat(parts, dim=1))


def build_dual_encoder(
    *,
    text_dim: int,
    struct_dim: int,
    encoder: str,
    hidden_dim: int,
    out_dim: int,
    gat_heads: tuple[int, int] = (8, 1),
    branch: str = "both",
    message_passing: bool = True,
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
        branch: ``"both"`` (default), ``"text"`` or ``"graph"``; see :class:`DualEncoder`.
        message_passing: ``False`` builds no GNN towers (Sprint 11.12 controls).

    Raises:
        ValueError: If ``encoder`` is not ``"gcn"`` or ``"gat"``, or ``branch`` is unknown.
    """
    if encoder not in ("gcn", "gat"):
        raise ValueError(f"Unknown encoder '{encoder}'; expected 'gcn' or 'gat'")
    if branch not in _BRANCHES:
        raise ValueError(f"Unknown branch '{branch}'; expected one of {_BRANCHES}")

    # Only the towers that are used are built, so an ablation does not consume random numbers for
    # modules it drops; the default ("both", message passing) builds both in the original order.
    graph_model1: nn.Module | None = None
    graph_model2: nn.Module | None = None
    if message_passing:
        if encoder == "gcn":
            if branch in ("both", "text"):
                graph_model1 = GCN(in_channels=text_dim, out_channels=out_dim, hid_dim=hidden_dim)
            if branch in ("both", "graph"):
                graph_model2 = GCN(in_channels=struct_dim, out_channels=struct_dim, hid_dim=struct_dim)
        else:
            if branch in ("both", "text"):
                graph_model1 = GAT(in_channels=text_dim, out_channels=out_dim, hid_dim=hidden_dim, heads=gat_heads)
            if branch in ("both", "graph"):
                graph_model2 = GAT(
                    in_channels=struct_dim, out_channels=struct_dim, hid_dim=struct_dim, heads=gat_heads
                )

    return DualEncoder(
        graph_model1,
        graph_model2,
        text_dim=text_dim,
        struct_dim=struct_dim,
        out_channels=out_dim,
        hidden_dim=hidden_dim,
        branch=branch,
        message_passing=message_passing,
    )


_ADAPTERS = ("neighbor_mean", "mlp", "relational", "gcn", "gat", "none")
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
        ``gcn``: the paper's text branch with DeepWalk removed (Phase 5.5), see below.
        ``gat``: the same branch with the paper's GAT tower in place of the GCN one (Phase 5.6).

    GCN adapter (``a`` = ``adapter_dim``): the text half of the paper's :class:`DualEncoder`, undirected::

        t_n = fc1(x̂_n)                              Linear(d, a)
        g_n = GCN(x̂, edge_index)_n                  the unchanged 2-layer GCN tower, d -> a -> a
        h_n = fc3( Dropout([t_n ; g_n]) )           Linear(2a, a), last layer std 1e-3

    With ``residual_floor=False`` (the A12 control) the SapBERT block is dropped and ``z = h``: the
    paper's architecture trained alone with the supervised loss, so it separates the effect of the
    SapBERT floor from the effect of the loss.

    GAT adapter (Phase 5.6): identical to the GCN adapter except for the graph tower, which is the unchanged
    two-layer :class:`GAT`: ``GATConv(d, a/H, heads=H)`` (heads concatenated to ``a``), ReLU,
    ``GATConv(a, a, heads=1)``, where ``H = gat_heads[0]``. Attention weights replace the GCN's fixed
    symmetric average, and the per-head width ``a/H`` keeps the tower the same size as the GCN's. ``residual_floor``
    works as for ``gcn`` (A15 is the no-floor control).

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

    Graph branch (Sprint 9.5, ``graph_branch=True``): a second residual block from a GCN over the *graph* (DeepWalk)
    features ``s``, built like the ``gcn`` adapter (``fc(s)`` and a 2-layer GCN over ``s``, fused by a last layer at
    std 1e-3) and appended as ``sqrt(lambda) * normalize(h_graph)``, so ``z = [x̂ ; text residual ; graph residual]``.
    ``adapter: none`` drops the text residual (``z = [x̂ ; graph residual]``) and needs the branch.

    Args:
        adapter: ``neighbor_mean``, ``mlp``, ``relational``, ``gcn``, ``gat`` or ``none``.
        lam: Fixed global residual weight ``lambda`` (``>= 0``).
        adapter_dim: Hidden/output width ``a`` of the trainable adapters (unused by ``neighbor_mean``).
        directional: When true, ``neighbor_mean`` uses ``h = [mean parents ; mean children]`` and
            ``relational`` uses separate parent/child weights; false collapses both to one undirected relation.
        text_dim: Width of the text input ``x̂`` (needed to size the trainable adapters).
        adapter_dropout: Dropout inside the trainable adapters (train mode only).
        residual_floor: When false, export ``z = h`` with no SapBERT block (``gcn``/``gat`` controls A12/A15).
        gat_heads: ``(first_layer_heads, second_layer_heads)`` for the ``gat`` adapter; ``adapter_dim`` must be
            divisible by ``gat_heads[0]``.
        graph_branch: Add the DeepWalk GCN residual block (default off: earlier runs are unchanged).
        graph_dim: Width of the graph features ``s`` (sizes the branch).

    Raises:
        ValueError: On an unknown adapter, negative ``lam``, ``adapter: none`` without the branch, or the branch
            with ``residual_floor=False``.
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
        residual_floor: bool = True,
        gat_heads: tuple[int, int] = (8, 1),
        graph_branch: bool = False,
        graph_dim: int = 128,
    ) -> None:
        super().__init__()
        if adapter == "none" and not graph_branch:
            raise ValueError("adapter 'none' needs graph_branch=True")
        if graph_branch and not residual_floor:
            raise ValueError("graph_branch needs residual_floor=True")
        if adapter not in _ADAPTERS:
            raise ValueError(f"Unknown adapter '{adapter}'; expected one of {list(_ADAPTERS)}")
        if lam < 0:
            raise ValueError(f"lambda must be >= 0, got {lam}")
        self.adapter = adapter
        self.lam = float(lam)
        self.adapter_dim = adapter_dim
        self.directional = directional
        self.residual_floor = residual_floor
        self.graph_branch = graph_branch
        if graph_branch:
            self.g_fc = nn.Linear(graph_dim, adapter_dim)
            self.g_gcn = GCN(in_channels=graph_dim, out_channels=adapter_dim, hid_dim=adapter_dim)
            self.g_dropout = nn.Dropout(adapter_dropout)
            self.g_fuse = nn.Linear(2 * adapter_dim, adapter_dim)
            nn.init.normal_(self.g_fuse.weight, std=1e-3)
            nn.init.zeros_(self.g_fuse.bias)
        if adapter in ("gcn", "gat"):
            if adapter == "gat" and adapter_dim % gat_heads[0] != 0:
                raise ValueError(f"adapter_dim {adapter_dim} must be divisible by gat_heads[0]={gat_heads[0]}")
            self.fc_text = nn.Linear(text_dim, adapter_dim)
            if adapter == "gcn":
                self.gcn = GCN(in_channels=text_dim, out_channels=adapter_dim, hid_dim=adapter_dim)
            else:
                self.gat = GAT(
                    in_channels=text_dim, out_channels=adapter_dim, hid_dim=adapter_dim // gat_heads[0], heads=gat_heads
                )
            self.fuse_dropout = nn.Dropout(adapter_dropout)
            self.fc_fuse = nn.Linear(2 * adapter_dim, adapter_dim)
            nn.init.normal_(self.fc_fuse.weight, std=1e-3)
            nn.init.zeros_(self.fc_fuse.bias)
        elif adapter == "mlp":
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
            struct_emb: Graph (DeepWalk) features; used only by the 9.5 ``graph_branch`` (DeepWalk is not
                cross-graph comparable, so it is not an SRGA text input).
            edge_index: ``(2, e)`` symmetrized message edges.
            edge_type: ``(e,)`` per-edge type (0 = stored child->parent direction,
                1 = reversed). Required when ``directional``.

        Returns:
            ``(n, d + d_h)`` with ``d_h`` = ``adapter_dim`` (``mlp``, ``relational``, ``gcn``, ``gat``), ``d`` (undirected
            ``neighbor_mean``) or ``2d`` (directional); ``d`` is the text width. With
            ``residual_floor=False`` only the ``adapter_dim`` columns of ``h`` are returned.

        Raises:
            ValueError: If ``directional`` and ``edge_type`` is ``None`` (``neighbor_mean``, ``relational``).
        """
        z = self._text_forward(text_emb, edge_index, edge_type)
        if not self.graph_branch:
            return z
        s = struct_emb.detach()
        h_g = self.g_fuse(self.g_dropout(torch.cat([self.g_fc(s), self.g_gcn(s, edge_index)], dim=1)))
        return torch.cat([z, (self.lam**0.5) * _unit(h_g)], dim=1)

    def _text_forward(
        self, text_emb: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor | None
    ) -> torch.Tensor:
        """``z`` without the graph branch (the pre-9.5 forward, unchanged)."""
        x_hat = _unit(text_emb.detach())
        if self.adapter == "none":
            return x_hat
        if self.adapter in ("gcn", "gat"):
            tower = self.gcn if self.adapter == "gcn" else self.gat
            h = self.fc_fuse(self.fuse_dropout(torch.cat([self.fc_text(x_hat), tower(x_hat, edge_index)], dim=1)))
            if not self.residual_floor:
                return h
            return torch.cat([x_hat, (self.lam**0.5) * _unit(h)], dim=1)
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
    residual_floor: bool = True,
    gat_heads: tuple[int, int] = (8, 1),
    graph_branch: bool = False,
    graph_dim: int = 128,
) -> ResidualAdapterEncoder:
    """Construct a :class:`ResidualAdapterEncoder` (see its docstring for the arguments)."""
    return ResidualAdapterEncoder(
        adapter=adapter,
        lam=lam,
        adapter_dim=adapter_dim,
        directional=directional,
        text_dim=text_dim,
        adapter_dropout=adapter_dropout,
        residual_floor=residual_floor,
        gat_heads=gat_heads,
        graph_branch=graph_branch,
        graph_dim=graph_dim,
    )


_GATES = ("learned", "none", "fixed_zero")
_NORMS = ("layernorm", "none")


def _propagate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute ``Â x`` with the GCN normalisation ``D^-1/2 (A + I) D^-1/2`` (self-loops added).

    Args:
        x: ``(n, d)`` node features.
        edge_index: ``(2, e)`` message edges, row 0 = source, row 1 = target (already symmetrised).

    Returns:
        ``(n, d)`` propagated features. An isolated node keeps its own features (self-loop only).
    """
    n = x.shape[0]
    loops = torch.arange(n, device=x.device)
    src = torch.cat([edge_index[0], loops])
    dst = torch.cat([edge_index[1], loops])
    degree = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(
        0, dst, torch.ones(dst.shape[0], device=x.device, dtype=x.dtype)
    )
    inv_sqrt = degree.pow(-0.5)
    weight = inv_sqrt[src] * inv_sqrt[dst]
    return torch.zeros_like(x).index_add_(0, dst, weight.unsqueeze(1) * x[src])


class GatedResidualEncoder(nn.Module):
    """Gated residual graph adapter over frozen text: ``z = LN(x + g ⊙ f(Â X))``.

    The frozen text vector ``x`` is kept as the base and a small graph transform ``f`` of the propagated features is
    added through a learned per-node, per-dimension gate ``g = sigmoid(W_g [x ‖ Â X] + b_g)`` whose bias starts at
    ``gate_bias_init`` (gate about 0), so the untrained model is ``LN(x)`` and structure enters only where it lowers
    the loss (``design_docs/gcl_method_update.md`` section 3). It sits in ``gcl_train`` beside :class:`DualEncoder`,
    selected by ``gcl_train.model.encoder: residual_gated``; ``evaluate`` scores the exported ``z`` unchanged.

    Text variant (``graph_branch=False``)::

        z_i = LN( x_i + g_i ⊙ f_t(Â X)_i ),   f_t = Linear(d, r) - ReLU - Linear(r, d)

    Dual variant (``graph_branch=True``, ``s`` = DeepWalk features): the gates arbitrate between the text and the
    graph signal instead of concatenating them::

        z_i = LN( x_i + g^t_i ⊙ f_t(Â X)_i + g^s_i ⊙ f_s(Â S)_i ),   g^s_i = sigmoid(W_s [x_i ‖ (Â S)_i] + b_s)

    The last layer of every ``f`` starts at std 1e-3 / zero bias and every gate weight at std 1e-3, so the initial gate
    is ``sigmoid(gate_bias_init)`` for all nodes. ``x`` and ``s`` are detached (frozen inputs).

    Args:
        text_dim: Width ``d`` of the text input (and of ``z``).
        graph_dim: Width of the structural input ``s`` (used only with ``graph_branch``).
        adapter_dim: Bottleneck width ``r`` of each ``f``.
        gate: ``"learned"`` (gate above), ``"none"`` (``g = 1``: ``z = LN(x + f(ÂX))``) or ``"fixed_zero"``
            (``g = 0``: ``z = LN(x)``, no trainable parameters).
        graph_branch: Add the DeepWalk branch.
        gate_bias_init: Initial gate bias (default ``-4``, gate about 0.018).
        norm: ``"layernorm"`` (parameter-free ``LayerNorm``) or ``"none"`` (control that isolates the effect of LN).

    Raises:
        ValueError: On an unknown ``gate`` or ``norm``.
    """

    def __init__(
        self,
        *,
        text_dim: int = 768,
        graph_dim: int = 128,
        adapter_dim: int = 128,
        gate: str = "learned",
        graph_branch: bool = False,
        gate_bias_init: float = -4.0,
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        if gate not in _GATES:
            raise ValueError(f"Unknown gate '{gate}'; expected one of {_GATES}")
        if norm not in _NORMS:
            raise ValueError(f"Unknown norm '{norm}'; expected one of {_NORMS}")
        self.text_dim = text_dim
        self.gate = gate
        self.graph_branch = graph_branch
        self.norm = norm
        if gate == "fixed_zero":
            return  # nothing trainable: z = LN(x)
        self.f_text = self._transform(text_dim, adapter_dim, text_dim)
        if gate == "learned":
            self.w_gate_text = self._gate_layer(2 * text_dim, text_dim, gate_bias_init)
        if graph_branch:
            self.f_graph = self._transform(graph_dim, adapter_dim, text_dim)
            if gate == "learned":
                self.w_gate_graph = self._gate_layer(text_dim + graph_dim, text_dim, gate_bias_init)

    @staticmethod
    def _transform(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
        """``Linear - ReLU - Linear`` whose last layer starts at std 1e-3 / zero bias."""
        net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        nn.init.normal_(net[-1].weight, std=1e-3)
        nn.init.zeros_(net[-1].bias)
        return net

    @staticmethod
    def _gate_layer(in_dim: int, out_dim: int, bias_init: float) -> nn.Linear:
        """Gate projection with weights at std 1e-3 and the bias at ``bias_init``."""
        layer = nn.Linear(in_dim, out_dim)
        nn.init.normal_(layer.weight, std=1e-3)
        nn.init.constant_(layer.bias, bias_init)
        return layer

    def gate_parameters(self) -> list[nn.Parameter]:
        """Return the gate projection parameters (the group that gets its own weight decay)."""
        return [p for name, p in self.named_parameters() if name.startswith("w_gate_")]

    def _terms(
        self, text_emb: torch.Tensor, struct_emb: torch.Tensor, edge_index: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return ``(transform output, gate)`` per branch; the gate is ``1`` for ``gate: none``."""
        x = text_emb.detach()
        ax = _propagate(x, edge_index)
        h = self.f_text(ax)
        g = torch.sigmoid(self.w_gate_text(torch.cat([x, ax], dim=1))) if self.gate == "learned" else torch.ones_like(h)
        terms = [(h, g)]
        if self.graph_branch:
            a_s = _propagate(struct_emb.detach(), edge_index)
            h_s = self.f_graph(a_s)
            g_s = (
                torch.sigmoid(self.w_gate_graph(torch.cat([x, a_s], dim=1)))
                if self.gate == "learned"
                else torch.ones_like(h_s)
            )
            terms.append((h_s, g_s))
        return terms

    def forward(
        self,
        text_emb: torch.Tensor,
        struct_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``z`` for every node.

        Args:
            text_emb: ``(n, d)`` frozen text embeddings.
            struct_emb: ``(n, graph_dim)`` structural embeddings (used only with ``graph_branch``).
            edge_index: ``(2, e)`` symmetrised message edges.
            edge_type: Accepted for interface parity with the other encoders and ignored.

        Returns:
            ``(n, d)`` embeddings.
        """
        h = text_emb.detach()
        if self.gate != "fixed_zero":
            for transformed, g in self._terms(text_emb, struct_emb, edge_index):
                h = h + g * transformed
        return F.layer_norm(h, (self.text_dim,)) if self.norm == "layernorm" else h

    @torch.no_grad()
    def gate_values(
        self, text_emb: torch.Tensor, struct_emb: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor | None:
        """Return the per-node trust score: the gate averaged over the ``d`` dimensions.

        Args:
            text_emb: ``(n, d)`` frozen text embeddings.
            struct_emb: ``(n, graph_dim)`` structural embeddings.
            edge_index: ``(2, e)`` symmetrised message edges.

        Returns:
            ``(n,)`` mean gate (text variant) or ``(n, 2)`` with the text and graph gates (dual variant);
            ``None`` unless ``gate == "learned"``.
        """
        if self.gate != "learned":
            return None
        scores = [g.mean(dim=1) for _, g in self._terms(text_emb, struct_emb, edge_index)]
        return torch.stack(scores, dim=1) if self.graph_branch else scores[0]


def build_gated_residual_encoder(
    *,
    text_dim: int = 768,
    graph_dim: int = 128,
    adapter_dim: int = 128,
    gate: str = "learned",
    graph_branch: bool = False,
    gate_bias_init: float = -4.0,
    norm: str = "layernorm",
) -> GatedResidualEncoder:
    """Construct a :class:`GatedResidualEncoder` (see its docstring for the arguments)."""
    return GatedResidualEncoder(
        text_dim=text_dim,
        graph_dim=graph_dim,
        adapter_dim=adapter_dim,
        gate=gate,
        graph_branch=graph_branch,
        gate_bias_init=gate_bias_init,
        norm=norm,
    )
