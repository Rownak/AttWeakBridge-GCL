"""Tests for ``GatedResidualEncoder`` (Sprint 12.5) and its ``gcl_train`` wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cga.core.io import read_json
from cga.data.aligned import AlignedMatrix
from cga.pipelines.gcl_train.models import (
    GatedResidualEncoder,
    _propagate,
    build_dual_encoder,
    build_gated_residual_encoder,
)
from .test_pipeline import (
    ALL_IDS,
    _build_config,
    _run_gcl_train,
    _setup_training_data_run,
)

N, D, S = 6, 12, 5
EDGES = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # nodes 4 and 5 are isolated


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N, D, generator=generator) * 3.0, torch.randn(N, S, generator=generator)


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (D,))


def _model(**kwargs) -> GatedResidualEncoder:
    torch.manual_seed(0)
    return build_gated_residual_encoder(text_dim=D, graph_dim=S, adapter_dim=4, **kwargs)


def test_fixed_zero_equals_layernorm_and_has_no_parameters():
    x, s = _inputs()
    model = _model(gate="fixed_zero")
    assert sum(p.numel() for p in model.parameters()) == 0
    assert torch.equal(model(x, s, EDGES), _ln(x))
    assert model.gate_values(x, s, EDGES) is None


def test_untrained_learned_gate_starts_near_sigmoid_of_bias_and_output_near_layernorm():
    x, s = _inputs()
    model = _model(gate="learned")
    gate = model.gate_values(x, s, EDGES)
    assert gate.shape == (N,)
    assert gate.mean().item() == pytest.approx(torch.sigmoid(torch.tensor(-4.0)).item(), abs=5e-3)
    assert (model(x, s, EDGES) - _ln(x)).abs().max().item() < 1e-2


def test_gate_none_has_no_gate_parameters_and_adds_the_transform():
    x, s = _inputs()
    model = _model(gate="none")
    assert model.gate_parameters() == []
    assert model.gate_values(x, s, EDGES) is None
    with torch.no_grad():
        model.f_text[-1].weight.normal_(std=0.5)
    out = model(x, s, EDGES)
    expected = _ln(x + model.f_text(_propagate(x, EDGES)))
    assert torch.allclose(out, expected, atol=1e-6)


def test_dual_model_with_gates_forced_to_zero_equals_layernorm():
    x, s = _inputs()
    model = _model(gate="learned", graph_branch=True)
    with torch.no_grad():
        for layer in (model.w_gate_text, model.w_gate_graph):
            layer.weight.zero_()
            layer.bias.fill_(-1.0e4)
        model.f_text[-1].weight.normal_(std=1.0)
        model.f_graph[-1].weight.normal_(std=1.0)
    assert torch.allclose(model(x, s, EDGES), _ln(x), atol=1e-6)
    gate = model.gate_values(x, s, EDGES)
    assert gate.shape == (N, 2) and gate.max().item() < 1e-6


def test_propagate_keeps_isolated_nodes_and_is_finite():
    x, _ = _inputs()
    out = _propagate(x, EDGES)
    assert torch.allclose(out[4:], x[4:])  # self-loop only: weight 1/(1*1)
    assert torch.isfinite(out).all()
    # connected pair: degree 2 each, weights 1/2 on self-loop and on the neighbour
    assert torch.allclose(out[0], 0.5 * (x[0] + x[1]))


def test_gate_bias_init_and_gate_parameter_group():
    model = _model(gate="learned", graph_branch=True, gate_bias_init=-2.5)
    assert torch.allclose(model.w_gate_text.bias, torch.full((D,), -2.5))
    assert model.w_gate_text.weight.std().item() < 5e-3
    ids = {id(p) for p in model.gate_parameters()}
    assert ids == {id(p) for layer in (model.w_gate_text, model.w_gate_graph) for p in layer.parameters()}


def test_output_shape_finite_and_gradients_reach_the_gate_only_for_learned():
    x, s = _inputs()
    for gate, expect_gate_grad in (("learned", True), ("none", False)):
        model = _model(gate=gate)
        model(x, s, EDGES).pow(2).sum().backward()
        assert model(x, s, EDGES).shape == (N, D)
        assert torch.isfinite(model(x, s, EDGES)).all()
        grads = [p.grad for p in model.gate_parameters()]
        assert (len(grads) > 0) == expect_gate_grad
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert model.f_text[0].weight.grad is not None


def test_same_seed_gives_identical_output_and_inputs_are_detached():
    x, s = _inputs()
    x.requires_grad_(True)
    out_a, out_b = _model(gate="learned", graph_branch=True)(x, s, EDGES), _model(gate="learned", graph_branch=True)(x, s, EDGES)
    assert torch.equal(out_a, out_b)
    assert not out_a.requires_grad or out_a.grad_fn is not None  # parameters require grad
    out_a.sum().backward()
    assert x.grad is None  # x is frozen


def test_norm_none_returns_the_raw_sum_and_unknown_options_are_rejected():
    x, s = _inputs()
    assert torch.equal(_model(gate="fixed_zero", norm="none")(x, s, EDGES), x)
    with pytest.raises(ValueError):
        build_gated_residual_encoder(gate="bogus")
    with pytest.raises(ValueError):
        build_gated_residual_encoder(norm="bogus")


def test_existing_encoders_are_unchanged():
    torch.manual_seed(0)
    default = build_dual_encoder(text_dim=8, struct_dim=4, encoder="gcn", hidden_dim=16, out_dim=16)
    assert sum(p.numel() for p in default.parameters() if p.requires_grad) > 0
    assert not hasattr(default, "gate_values")


def _gated_config(tmp_path, **model):
    config = _build_config(tmp_path, epochs=3)
    config["gcl_train"]["loss"]["name"] = "infonce"
    config["gcl_train"]["model"] = {"encoder": "residual_gated", "adapter_dim": 4, "model_id": "gated-test", **model}
    return config


def test_pipeline_trains_gated_model_and_exports_gate(tmp_path):
    config = _gated_config(tmp_path, gate="learned", gate_weight_decay=1.0e-3)
    _setup_training_data_run(
        config,
        example_tuples=[
            ("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3"),
            ("attack:2", "cwe:2", "cwe:3", "attack:1", "attack:4"),
        ],
    )
    ctx = _run_gcl_train(config)

    matrix = AlignedMatrix.load(ctx.run_dir / "node_embeddings")
    assert matrix.vectors.shape == (len(ALL_IDS), 6)  # text_dim 6 in the fixture
    assert np.isfinite(matrix.vectors).all()
    gate = np.load(ctx.run_dir / "node_embeddings" / "gate.npy")
    assert gate.shape == (len(ALL_IDS),)
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert "node_embeddings_gate" in [o["name"] for o in manifest["outputs"]]
    assert manifest["summary"]["n_trainable_params"] > 0
    assert len((ctx.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 3


def test_pipeline_dual_gate_exports_two_columns_and_fixed_zero_skips_training(tmp_path):
    config = _gated_config(tmp_path, gate="learned", graph_branch=True)
    _setup_training_data_run(config, example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3")])
    ctx = _run_gcl_train(config)
    assert np.load(ctx.run_dir / "node_embeddings" / "gate.npy").shape == (len(ALL_IDS), 2)

    config = _gated_config(tmp_path / "fz", gate="fixed_zero")
    _setup_training_data_run(config, example_tuples=[("attack:1", "cwe:1", "cwe:2", "attack:2", "attack:3")])
    ctx = _run_gcl_train(config)
    assert not (ctx.run_dir / "node_embeddings" / "gate.npy").exists()
    assert read_json(ctx.run_dir / "manifest.json")["summary"]["n_trainable_params"] == 0
