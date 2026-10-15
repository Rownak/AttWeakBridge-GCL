"""Registered contrastive losses: ``triplet``, ``infonce`` and ``infonce_full`` (target_architecture.md §5.4, §7.8).

Both operate on the final fused node embeddings (not raw input features), take
one row per ``(anchor, positive, negative)`` index into those embeddings, and
are used identically for the text view (cross-graph pairs) and the structural
view (within-graph pairs) — the view only changes which ``examples.parquet``
columns supply the indices, not the loss math.

``jsd`` (Jensen-Shannon divergence over softmax'd embeddings) from the legacy
``GNN_dual_loss2.py`` is deliberately not ported: treating an L2-normalized
embedding's softmax as a probability distribution has no principled
interpretation, and the developer confirmed dropping it rather than carrying
the legacy approximation forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from cga.core.registry import losses


class ContrastiveLoss(Protocol):
    """A registered loss: anchor/positive/negative embeddings -> scalar loss."""

    def __call__(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, **kwargs: float
    ) -> torch.Tensor: ...


@losses.register("triplet")
def triplet_loss(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, *, margin: float = 1.0
) -> torch.Tensor:
    """Cosine-distance triplet margin loss (legacy ``ContrastiveLoss``).

    ``relu(dist(anchor, positive) - dist(anchor, negative) + margin)``, mean
    over the batch, where distance is ``1 - cosine_similarity``. Zero when the
    negative is already at least ``margin`` farther from the anchor than the
    positive is.
    """
    anchor_n = F.normalize(anchor, p=2, dim=1)
    positive_n = F.normalize(positive, p=2, dim=1)
    negative_n = F.normalize(negative, p=2, dim=1)

    pos_dist = 1 - F.cosine_similarity(anchor_n, positive_n, dim=1)
    neg_dist = 1 - F.cosine_similarity(anchor_n, negative_n, dim=1)

    return torch.relu(pos_dist - neg_dist + margin).mean()


@losses.register("infonce")
def infonce_loss(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, *, temperature: float = 0.1
) -> torch.Tensor:
    """In-batch InfoNCE: each row's positive against its own negative plus every other row's negative.

    Every other example's negative in the batch is reused as an additional
    negative for this row (in-batch negatives), which is standard InfoNCE
    practice and strictly more informative than one negative per anchor
    without introducing any new positive/negative source beyond
    ``examples.parquet``'s own per-row negative column.
    """
    anchor_n = F.normalize(anchor, p=2, dim=1)
    positive_n = F.normalize(positive, p=2, dim=1)
    negative_n = F.normalize(negative, p=2, dim=1)

    positive_logit = torch.sum(anchor_n * positive_n, dim=1, keepdim=True) / temperature
    negative_logits = (anchor_n @ negative_n.t()) / temperature

    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


@dataclass
class FullSoftmaxData:
    """Index tensors for :func:`infonce_full_loss` (rows are universe-row indices).

    Attributes:
        anchor: ``(P,)`` source-graph (NCIT) row of each positive pair.
        positive: ``(P,)`` target-graph (DOID) row of each positive pair.
        cand_fwd: ``(C2,)`` non-held-out target-graph rows (forward candidates).
        cand_rev: ``(C1,)`` non-held-out source-graph rows (reverse candidates).
        target_fwd: ``(P,)`` position of ``positive`` inside ``cand_fwd``.
        target_rev: ``(P,)`` position of ``anchor`` inside ``cand_rev``.
        mask_fwd: ``(2, M)`` (pair index, ``cand_fwd`` position) logits to exclude.
        mask_rev: ``(2, M)`` (pair index, ``cand_rev`` position) logits to exclude.
    """

    anchor: torch.Tensor
    positive: torch.Tensor
    cand_fwd: torch.Tensor
    cand_rev: torch.Tensor
    target_fwd: torch.Tensor
    target_rev: torch.Tensor
    mask_fwd: torch.Tensor
    mask_rev: torch.Tensor


def build_full_softmax_data(
    pairs: list[tuple[int, int]],
    *,
    graph_roles: list[str],
    held_out: list[bool],
    gold_pairs: list[tuple[int, int]],
    device: str | torch.device = "cpu",
) -> FullSoftmaxData:
    """Build the candidate sets and gold masks for the full-softmax loss.

    Args:
        pairs: Positive ``(anchor_row, positive_row)`` pairs (duplicates are dropped).
        graph_roles: ``graph_role`` of every universe row.
        held_out: ``held_out`` flag of every universe row; held-out rows are never candidates.
        gold_pairs: Every gold ``(source_row, target_row)`` alignment of any split, already
            restricted to the universe. Other gold targets of an anchor (and other gold sources
            of a positive) are masked so a valid alternative is never scored as a negative.
        device: Device for the returned tensors.

    Returns:
        A :class:`FullSoftmaxData`.

    Raises:
        ValueError: If ``pairs`` is empty or a pair endpoint is a held-out row (it would be
            missing from the candidates, i.e. a leak).
    """
    pairs = list(dict.fromkeys(pairs))
    if not pairs:
        raise ValueError("infonce_full needs at least one positive pair")
    source_role = graph_roles[pairs[0][0]]
    target_role = graph_roles[pairs[0][1]]
    cand_fwd = [i for i, (r, h) in enumerate(zip(graph_roles, held_out)) if r == target_role and not h]
    cand_rev = [i for i, (r, h) in enumerate(zip(graph_roles, held_out)) if r == source_role and not h]
    col_fwd = {row: j for j, row in enumerate(cand_fwd)}
    col_rev = {row: j for j, row in enumerate(cand_rev)}

    targets_of: dict[int, list[int]] = {}
    sources_of: dict[int, list[int]] = {}
    for s, t in gold_pairs:
        targets_of.setdefault(s, []).append(t)
        sources_of.setdefault(t, []).append(s)

    target_fwd, target_rev = [], []
    fwd_rows, fwd_cols, rev_rows, rev_cols = [], [], [], []
    for i, (a, p) in enumerate(pairs):
        if p not in col_fwd or a not in col_rev:
            raise ValueError(f"pair ({a}, {p}) has a held-out or wrong-role endpoint")
        target_fwd.append(col_fwd[p])
        target_rev.append(col_rev[a])
        for t in targets_of.get(a, []):
            if t != p and t in col_fwd:
                fwd_rows.append(i)
                fwd_cols.append(col_fwd[t])
        for s in sources_of.get(p, []):
            if s != a and s in col_rev:
                rev_rows.append(i)
                rev_cols.append(col_rev[s])

    def _t(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    return FullSoftmaxData(
        anchor=_t([a for a, _ in pairs]),
        positive=_t([p for _, p in pairs]),
        cand_fwd=_t(cand_fwd),
        cand_rev=_t(cand_rev),
        target_fwd=_t(target_fwd),
        target_rev=_t(target_rev),
        mask_fwd=torch.stack([_t(fwd_rows), _t(fwd_cols)]),
        mask_rev=torch.stack([_t(rev_rows), _t(rev_cols)]),
    )


def _masked_cross_entropy(
    query: torch.Tensor, candidates: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = (query @ candidates.t()) / temperature
    if mask.shape[1] > 0:
        blocked = torch.zeros_like(logits, dtype=torch.bool)
        blocked[mask[0], mask[1]] = True
        logits = logits.masked_fill(blocked, float("-inf"))
    return F.cross_entropy(logits, target)


@losses.register("infonce_full")
def infonce_full_loss(
    embeddings: torch.Tensor, data: FullSoftmaxData, *, temperature: float = 0.05, reverse_weight: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-softmax InfoNCE over every non-held-out candidate (design §6).

    Unlike the other registered losses this takes the whole embedding matrix and a
    :class:`FullSoftmaxData`, not ``(anchor, positive, negative)`` rows, so the trainer
    branches on its name. Forward term: each source anchor against all target-graph
    candidates; reverse term (weight ``reverse_weight``): each target positive against all
    source-graph candidates. Scores are ``cos / temperature``; masked gold alternatives are
    removed from the softmax.

    Args:
        embeddings: ``(n_universe, d)`` node embeddings (any scale; rows are L2-normalized here).
        data: Indices and masks from :func:`build_full_softmax_data`.
        temperature: Softmax temperature.
        reverse_weight: Weight of the reverse term.

    Returns:
        ``(total, forward, reverse)`` scalar losses with ``total = forward + reverse_weight * reverse``.
    """
    z = F.normalize(embeddings, p=2, dim=1)
    fwd = _masked_cross_entropy(z[data.anchor], z[data.cand_fwd], data.target_fwd, data.mask_fwd, temperature)
    rev = _masked_cross_entropy(z[data.positive], z[data.cand_rev], data.target_rev, data.mask_rev, temperature)
    return fwd + reverse_weight * rev, fwd, rev
