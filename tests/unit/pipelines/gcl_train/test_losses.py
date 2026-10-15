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


# ---- infonce_full ------------------------------------------------------------------------------

import math  # noqa: E402

from cga.pipelines.gcl_train.losses import build_full_softmax_data, infonce_full_loss  # noqa: E402

_R2 = 2**-0.5
# rows 0,1 = source anchors; rows 2..5 = target candidates.
_Z = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [_R2, _R2], [-1.0, 0.0]],
)
_ROLES = ["g1", "g1", "g2", "g2", "g2", "g2"]


def _data(**kw):
    args = dict(pairs=[(0, 2), (1, 3)], graph_roles=_ROLES, held_out=[False] * 6, gold_pairs=[(0, 2), (1, 3)])
    args.update(kw)
    return build_full_softmax_data(args.pop("pairs"), **args)


def _nll(logits: list[float], target: int, skip: tuple[int, ...] = ()) -> float:
    keep = [x for i, x in enumerate(logits) if i not in skip]
    kept_target = target - sum(1 for s in skip if s < target)
    return -(keep[kept_target] - math.log(sum(math.exp(x) for x in keep)))


def test_infonce_full_matches_hand_computed_value():
    total, fwd, rev = infonce_full_loss(_Z, _data(), temperature=1.0, reverse_weight=0.5)
    # forward: 2 anchors x 4 candidates (rows 2..5). anchor 0=(1,0): cos [1, 0, R2, -1]; anchor 1=(0,1): [0, 1, R2, 0].
    expected_fwd = (_nll([1, 0, _R2, -1], 0) + _nll([0, 1, _R2, 0], 1)) / 2
    # reverse: positives rows 2=(1,0), 3=(0,1) vs 2 NCIT candidates (rows 0,1): cos [1,0] and [0,1].
    expected_rev = (_nll([1, 0], 0) + _nll([0, 1], 1)) / 2
    assert fwd.item() == pytest.approx(expected_fwd, abs=1e-6)
    assert rev.item() == pytest.approx(expected_rev, abs=1e-6)
    assert total.item() == pytest.approx(expected_fwd + 0.5 * expected_rev, abs=1e-6)


def test_infonce_full_masks_other_gold_targets_of_the_anchor():
    # Anchor 0 also has gold target row 4 (not its positive): it must drop out of anchor 0's softmax.
    data = _data(gold_pairs=[(0, 2), (1, 3), (0, 4)])
    _, fwd, _ = infonce_full_loss(_Z, data, temperature=1.0)
    expected = (_nll([1, 0, _R2, -1], 0, skip=(2,)) + _nll([0, 1, _R2, 0], 1)) / 2
    assert fwd.item() == pytest.approx(expected, abs=1e-6)
    assert data.mask_fwd.tolist() == [[0], [2]]  # (pair 0, candidate column of row 4)


def test_masked_gold_target_contributes_no_gradient():
    z2 = _Z.clone().requires_grad_(True)
    data = _data(pairs=[(0, 2)], gold_pairs=[(0, 2), (0, 4)])
    _, fwd2, _ = infonce_full_loss(z2, data, temperature=1.0, reverse_weight=0.0)
    fwd2.backward()
    assert torch.count_nonzero(z2.grad[4]) == 0  # only anchor 0 is queried, and row 4 is masked for it


def test_held_out_nodes_are_never_candidates_and_pair_endpoints_must_be_trainable():
    held = [False, False, False, False, True, False]
    data = _data(held_out=held)
    assert 4 not in data.cand_fwd.tolist() and data.cand_fwd.tolist() == [2, 3, 5]
    assert data.cand_rev.tolist() == [0, 1]
    with pytest.raises(ValueError):
        _data(held_out=[False, False, True, False, False, False])  # positive of pair (0, 2) is held out
    with pytest.raises(ValueError):
        _data(pairs=[])


def test_infonce_full_reverse_weight_is_respected_and_pairs_are_deduplicated():
    data = _data(pairs=[(0, 2), (1, 3), (0, 2)])
    assert len(data.anchor) == 2
    t0, fwd, rev = infonce_full_loss(_Z, data, temperature=1.0, reverse_weight=0.0)
    t1, _, _ = infonce_full_loss(_Z, data, temperature=1.0, reverse_weight=2.0)
    assert t0.item() == pytest.approx(fwd.item())
    assert t1.item() == pytest.approx(fwd.item() + 2.0 * rev.item())
