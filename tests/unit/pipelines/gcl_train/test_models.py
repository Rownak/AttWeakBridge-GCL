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


# ---- relational adapter (A4 / A6) ---------------------------------------------------------------


def _bare_relational(directional: bool, d: int = 3, a: int = 4):
    """Relational adapter with LayerNorm/MLP_2 removed, so h = u + r and the relational term is visible."""
    model = build_residual_adapter_encoder(
        adapter="relational", lam=1.0, adapter_dim=a, text_dim=d, directional=directional, adapter_dropout=0.0
    )
    model.norm = torch.nn.Identity()
    model.mlp2 = torch.nn.Identity()
    model.eval()
    return model


def test_relational_export_layout_is_text_block_then_scaled_unit_residual():
    text, edge_index, edge_type = _chain_fixture()
    model = build_residual_adapter_encoder(adapter="relational", lam=0.25, adapter_dim=5, text_dim=3)
    model.eval()
    z = model(text * 2.0, text, edge_index, edge_type)
    assert z.shape == (3, 3 + 5)
    assert torch.allclose(z[:, :3], text)
    assert torch.allclose(z[:, 3:].norm(dim=1), torch.full((3,), 0.5), atol=1e-5)  # sqrt(0.25) * unit


def test_relational_layer_uses_separate_weights_for_parents_and_children():
    text, edge_index, edge_type = _chain_fixture()  # node 0 = leaf (parent 1), node 2 = root (child 1)
    model = _bare_relational(directional=True)

    def h_with(w_parent_on: bool, w_child_on: bool) -> torch.Tensor:
        torch.manual_seed(0)
        with torch.no_grad():
            model.w_parent.weight.copy_(torch.eye(4) if w_parent_on else torch.zeros(4, 4))
            model.w_child.weight.copy_(torch.eye(4) if w_child_on else torch.zeros(4, 4))
        return model._relational_h(text, edge_index, edge_type)

    both, only_parent, only_child, neither = h_with(True, True), h_with(True, False), h_with(False, True), h_with(False, False)
    # Leaf (node 0) has a parent but no children: only W_par can matter.
    assert not torch.allclose(only_parent[0], neither[0]) and torch.allclose(only_child[0], neither[0])
    # Root (node 2) has a child but no parents: only W_chi can matter.
    assert not torch.allclose(only_child[2], neither[2]) and torch.allclose(only_parent[2], neither[2])
    # Middle node (1) has both, and the two sides contribute different amounts.
    assert not torch.allclose(only_parent[1], only_child[1])
    assert torch.allclose(both[1] - neither[1], (only_parent[1] - neither[1]) + (only_child[1] - neither[1]), atol=1e-6)


def test_relational_undirected_control_has_one_relation_and_fewer_parameters():
    directional = build_residual_adapter_encoder(adapter="relational", adapter_dim=8, text_dim=4, directional=True)
    undirected = build_residual_adapter_encoder(adapter="relational", adapter_dim=8, text_dim=4, directional=False)
    assert hasattr(directional, "w_parent") and hasattr(directional, "w_child")
    assert hasattr(undirected, "w_neighbor") and not hasattr(undirected, "w_parent")
    assert directional.proj.in_features == 4 * 4 + 2 and undirected.proj.in_features == 3 * 4 + 1
    text, edge_index, edge_type = _chain_fixture()
    small = build_residual_adapter_encoder(adapter="relational", adapter_dim=8, text_dim=3, directional=False).eval()
    assert small(text, text, edge_index).shape == (3, 3 + 8)  # undirected needs no edge_type


def test_relational_adapter_is_trainable_and_needs_edge_type_when_directional():
    model = build_residual_adapter_encoder(adapter="relational", adapter_dim=16, text_dim=3, directional=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    text, edge_index, _ = _chain_fixture()
    with pytest.raises(ValueError):
        model(text, text, edge_index)


def test_relational_adapter_lambda_zero_has_zero_gradient():
    text, edge_index, edge_type = _chain_fixture()
    model = build_residual_adapter_encoder(adapter="relational", lam=0.0, adapter_dim=6, text_dim=3)
    model(text, text, edge_index, edge_type).pow(2).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


# ---- gcn adapter (A11) and the no-floor control (A12) ------------------------------------------


def test_gcn_adapter_output_layout_and_text_block_is_the_unit_vector():
    text, edge_index, edge_type = _chain_fixture()
    model = build_residual_adapter_encoder(adapter="gcn", lam=0.25, adapter_dim=5, text_dim=3).eval()
    z = model(text * 4.0, text, edge_index, edge_type)
    assert z.shape == (3, 3 + 5)
    assert torch.allclose(z[:, :3], text)  # SapBERT block is the unit text vector, untouched
    assert torch.allclose(z[:, 3:].norm(dim=1), torch.full((3,), 0.5), atol=1e-5)  # sqrt(0.25) * unit


def test_gcn_adapter_depends_on_edges_unlike_mlp_and_has_paper_sized_parameters():
    text = torch.randn(4, 6)
    chain = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    none = torch.empty((2, 0), dtype=torch.long)

    def with_big_last_layer(adapter: str):
        torch.manual_seed(0)
        model = build_residual_adapter_encoder(adapter=adapter, lam=1.0, adapter_dim=8, text_dim=6, adapter_dropout=0.0)
        last = model.fc_fuse if adapter == "gcn" else model.mlp[-1]
        torch.nn.init.normal_(last.weight, std=1.0)  # the 1e-3 init would hide the difference
        return model.eval()

    gcn, mlp = with_big_last_layer("gcn"), with_big_last_layer("mlp")
    assert not torch.allclose(gcn(text, text, chain), gcn(text, text, none))
    assert torch.allclose(mlp(text, text, chain), mlp(text, text, none))
    full = build_residual_adapter_encoder(adapter="gcn", text_dim=768)
    assert sum(p.numel() for p in full.parameters()) == 590_848  # fc1 + 2 GCNConv + fc3


def test_gcn_adapter_lambda_zero_has_zero_gradient():
    text, edge_index, edge_type = _chain_fixture()
    model = build_residual_adapter_encoder(adapter="gcn", lam=0.0, adapter_dim=6, text_dim=3)
    model(text, text, edge_index, edge_type).pow(2).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_residual_floor_false_exports_h_only_and_defaults_to_the_floor():
    text, edge_index, edge_type = _chain_fixture()
    no_floor = build_residual_adapter_encoder(adapter="gcn", adapter_dim=5, text_dim=3, residual_floor=False).eval()
    z = no_floor(text, text, edge_index, edge_type)
    assert z.shape == (3, 5)  # no SapBERT block
    assert build_residual_adapter_encoder(adapter="gcn", adapter_dim=5, text_dim=3).eval()(
        text, text, edge_index, edge_type
    ).shape == (3, 3 + 5)


# ---- gat adapter (A14) and the no-floor control (A15) ------------------------------------------


def test_gat_adapter_output_layout_and_text_block_is_the_unit_vector():
    text, edge_index, edge_type = _chain_fixture()
    model = build_residual_adapter_encoder(
        adapter="gat", lam=0.25, adapter_dim=8, text_dim=3, gat_heads=(2, 1)
    ).eval()
    z = model(text * 4.0, text, edge_index, edge_type)
    assert z.shape == (3, 3 + 8)
    assert torch.allclose(z[:, :3], text)  # SapBERT block is the unit text vector, untouched
    assert torch.allclose(z[:, 3:].norm(dim=1), torch.full((3,), 0.5), atol=1e-5)  # sqrt(0.25) * unit


def test_gat_adapter_depends_on_edges_and_matches_gcn_adapter_size():
    text = torch.randn(4, 6)
    chain = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    none = torch.empty((2, 0), dtype=torch.long)
    torch.manual_seed(0)
    model = build_residual_adapter_encoder(
        adapter="gat", lam=1.0, adapter_dim=8, text_dim=6, adapter_dropout=0.0, gat_heads=(2, 1)
    )
    torch.nn.init.normal_(model.fc_fuse.weight, std=1.0)  # the 1e-3 init would hide the difference
    model.eval()
    assert not torch.allclose(model(text, text, chain), model(text, text, none))
    gat = build_residual_adapter_encoder(adapter="gat", text_dim=768)
    gcn = build_residual_adapter_encoder(adapter="gcn", text_dim=768)
    n_gat, n_gcn = (sum(p.numel() for p in m.parameters()) for m in (gat, gcn))
    assert abs(n_gat - n_gcn) / n_gcn < 0.01  # attention vs averaging at matched size
    assert hasattr(gat, "gat") and not hasattr(gat, "gcn")


def test_gat_adapter_lambda_zero_has_zero_gradient():
    text, edge_index, edge_type = _chain_fixture()
    model = build_residual_adapter_encoder(adapter="gat", lam=0.0, adapter_dim=6, text_dim=3, gat_heads=(2, 1))
    model(text, text, edge_index, edge_type).pow(2).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_gat_residual_floor_false_exports_h_only():
    text, edge_index, edge_type = _chain_fixture()
    z = build_residual_adapter_encoder(
        adapter="gat", adapter_dim=8, text_dim=3, gat_heads=(2, 1), residual_floor=False
    ).eval()(text, text, edge_index, edge_type)
    assert z.shape == (3, 8)


def test_gat_adapter_dim_must_be_divisible_by_heads():
    with pytest.raises(ValueError):
        build_residual_adapter_encoder(adapter="gat", adapter_dim=10, text_dim=3, gat_heads=(4, 1))


# ---- graph branch (Sprint 9.5) -----------------------------------------------------------


def _graph_fixture():
    text, edge_index, edge_type = _chain_fixture()
    return text, torch.randn(3, 4), edge_index, edge_type


def test_graph_branch_is_off_by_default_and_changes_nothing():
    text, struct, edge_index, edge_type = _graph_fixture()
    torch.manual_seed(0)
    plain = build_residual_adapter_encoder(adapter="gcn", lam=0.25, adapter_dim=5, text_dim=3).eval()
    torch.manual_seed(0)
    explicit = build_residual_adapter_encoder(
        adapter="gcn", lam=0.25, adapter_dim=5, text_dim=3, graph_branch=False, graph_dim=4
    ).eval()
    assert torch.equal(plain(text, struct, edge_index, edge_type), explicit(text, struct, edge_index, edge_type))


def test_graph_branch_appends_a_scaled_unit_block_and_none_drops_the_text_residual():
    text, struct, edge_index, edge_type = _graph_fixture()
    both = build_residual_adapter_encoder(
        adapter="gcn", lam=0.25, adapter_dim=5, text_dim=3, graph_branch=True, graph_dim=4
    ).eval()
    z = both(text, struct, edge_index, edge_type)
    assert z.shape == (3, 3 + 5 + 5)
    assert torch.allclose(z[:, :3], text)
    assert torch.allclose(z[:, 8:].norm(dim=1), torch.full((3,), 0.5), atol=1e-5)

    graph_only = build_residual_adapter_encoder(
        adapter="none", lam=0.25, adapter_dim=5, text_dim=3, graph_branch=True, graph_dim=4
    ).eval()
    assert graph_only(text, struct, edge_index, edge_type).shape == (3, 3 + 5)
    # The graph branch reads the graph features; the text-only residual does not.
    assert not torch.allclose(
        graph_only(text, struct, edge_index, edge_type), graph_only(text, struct * 2 + 1, edge_index, edge_type)
    )


def test_graph_branch_rejects_inconsistent_options():
    with pytest.raises(ValueError, match="none"):
        build_residual_adapter_encoder(adapter="none", text_dim=3)
    with pytest.raises(ValueError, match="residual_floor"):
        build_residual_adapter_encoder(adapter="gcn", text_dim=3, graph_branch=True, residual_floor=False)
