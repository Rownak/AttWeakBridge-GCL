"""Per-node train/val/test partition (target_architecture.md §5.1.1).

Proportional per ``graph_role``, identical policy for every dataset: each
graph is shuffled independently with a seeded RNG and cut into contiguous
``train``/``val``/``test`` blocks by ``fractions``. This is a node-level
partition, not derived from any alignment pairs a dataset happens to ship —
``ncit_doid``'s own ``refs_equiv/{train,test}.tsv`` split stays recorded
separately in ``alignments.parquet.split`` and is combined with this one only
downstream, in ``training_data.build_universe`` (§5.3 step 1).
"""

from __future__ import annotations

import random

from cga.core.errors import ConfigError
from cga.data.schemas import GraphRole, NodeSplit

_SPLIT_NAMES: tuple[NodeSplit, NodeSplit, NodeSplit] = ("train", "val", "test")


def validate_fractions(fractions: list[float]) -> None:
    """Raise if ``fractions`` is not exactly three non-negative values summing to 1.0.

    Raises:
        ConfigError: If the length or sum is wrong, or any fraction is negative.
    """
    if len(fractions) != 3:
        raise ConfigError(f"split.fractions must have exactly 3 values (train, val, test), got {fractions}")
    if any(f < 0 for f in fractions):
        raise ConfigError(f"split.fractions must be non-negative, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ConfigError(f"split.fractions must sum to 1.0, got {fractions} (sum={sum(fractions)})")


def split_nodes(
    node_ids_by_role: dict[GraphRole, list[str]],
    *,
    fractions: list[float],
    seed: int,
) -> dict[str, NodeSplit]:
    """Return ``{node_id: split}`` for every node, partitioned independently per graph_role.

    Each graph's node ids are shuffled with ``random.Random(seed)`` (a fresh
    instance per call, independent of global RNG state) and cut into
    contiguous ``train``/``val``/``test`` blocks sized by ``fractions``, so
    both graphs contribute the same proportions regardless of their relative
    size. Deterministic: the same ``node_ids_by_role``, ``fractions``, and
    ``seed`` always produce the same assignment.

    Args:
        node_ids_by_role: ``{"graph_1": [...], "graph_2": [...]}``, each list
            of node ids for that graph (order irrelevant — reshuffled here).
        fractions: ``[train, val, test]`` proportions, validated by
            :func:`validate_fractions`.
        seed: RNG seed.

    Returns:
        A mapping from every input node id to its assigned split.
    """
    validate_fractions(fractions)

    result: dict[str, NodeSplit] = {}
    for role in sorted(node_ids_by_role):
        ids = list(node_ids_by_role[role])
        rng = random.Random(f"{seed}:{role}")
        rng.shuffle(ids)

        n = len(ids)
        n_train = round(n * fractions[0])
        n_val = round(n * fractions[1])
        # test gets the remainder, so rounding never drops or duplicates a node.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        boundaries = [0, n_train, n_train + n_val, n]
        for split_name, start, end in zip(_SPLIT_NAMES, boundaries[:-1], boundaries[1:]):
            for node_id in ids[start:end]:
                result[node_id] = split_name

    return result
