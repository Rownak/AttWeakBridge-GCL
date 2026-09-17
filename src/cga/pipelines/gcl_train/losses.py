"""Registered contrastive losses: ``triplet`` and ``infonce`` (target_architecture.md §5.4, §7.8).

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
