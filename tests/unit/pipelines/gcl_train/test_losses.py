"""Hand-computed checks for the registered contrastive losses."""

from __future__ import annotations

import math

import pytest
import torch

from cga.pipelines.gcl_train.losses import infonce_loss, triplet_loss


def test_triplet_loss_zero_when_negative_beyond_margin() -> None:
    # anchor == positive (distance 0); negative orthogonal (cosine distance 1).
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0]])

    loss = triplet_loss(anchor, positive, negative, margin=0.5)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_triplet_loss_hand_computed() -> None:
    # anchor=(1,0), positive=(0,1) -> cos_sim=0 -> pos_dist=1
    # anchor=(1,0), negative=(1,0) -> cos_sim=1 -> neg_dist=0
    # loss = relu(1 - 0 + margin) = 1 + margin
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[0.0, 1.0]])
    negative = torch.tensor([[1.0, 0.0]])
    margin = 0.3

    loss = triplet_loss(anchor, positive, negative, margin=margin)

    assert loss.item() == pytest.approx(1.0 + margin, abs=1e-6)


def test_triplet_loss_batch_mean() -> None:
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negative = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    margin = 0.3
    # row0: pos_dist=0, neg_dist=0 -> relu(0-0+0.3)=0.3
    # row1: pos_dist=1, neg_dist=0 -> relu(1-0+0.3)=1.3
    expected = (0.3 + 1.3) / 2

    loss = triplet_loss(anchor, positive, negative, margin=margin)

    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_infonce_loss_low_when_positive_dominates() -> None:
    # anchor perfectly aligned with positive, orthogonal to negative -> near-zero CE loss.
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0]])

    loss = infonce_loss(anchor, positive, negative, temperature=0.1)

    assert loss.item() < 1e-3


def test_infonce_loss_hand_computed_single_row() -> None:
    # anchor=(1,0), positive=(1,0) -> logit_pos = 1/T
    # anchor=(1,0), negative=(0,1) -> logit_neg = 0/T
    # cross_entropy([logit_pos, logit_neg], label=0)
    #   = -log( exp(logit_pos) / (exp(logit_pos) + exp(logit_neg)) )
    temperature = 0.5
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0]])

    logit_pos = 1.0 / temperature
    logit_neg = 0.0 / temperature
    expected = -math.log(math.exp(logit_pos) / (math.exp(logit_pos) + math.exp(logit_neg)))

    loss = infonce_loss(anchor, positive, negative, temperature=temperature)

    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_infonce_uses_in_batch_negatives() -> None:
    # anchor/positive for both rows are trivially aligned (cos_sim=1), so any
    # difference in loss comes only from the negative logits matrix, which
    # includes every row's own negative *and* every other row's negative
    # in-batch. Row 1's own negative is orthogonal to row 0's anchor
    # (contributes nothing extra) in `negative_orthogonal`, but is made
    # identical to row 0's anchor in `negative_aligned_with_row0` -- if
    # in-batch negatives are used, that extra logit must raise row 0's loss.
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    negative_orthogonal = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    negative_aligned_with_row0 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    loss_orthogonal = infonce_loss(anchor, positive, negative_orthogonal, temperature=0.1)
    loss_aligned = infonce_loss(anchor, positive, negative_aligned_with_row0, temperature=0.1)

    assert loss_aligned.item() > loss_orthogonal.item()
