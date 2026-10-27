"""Shared target-side candidate pool for the gold evaluators (Sprint 10.6).

A pool file is a JSON object ``{"node_ids": [...], "source_run": "...", "sha256": "..."}``
written by ``scripts/make_target_pool.py``. The ``gold_alignment`` and ``gold_alignment_prf``
evaluators accept its path as ``target_pool`` and rank / predict only among those target nodes.
The pool is derived from gold, so a pool view is optimistic by construction; it is meant to be
read next to a random baseline on the same pool.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cga.core.errors import ConfigError


def pool_sha256(node_ids: list[str]) -> str:
    """Return the sha256 of the sorted, newline-joined node ids (the pool's identity)."""
    return hashlib.sha256("\n".join(sorted(node_ids)).encode("utf-8")).hexdigest()


def load_target_pool(path: str | Path, *, evaluation_id: str) -> tuple[set[str], str]:
    """Read a pool file and verify its recorded hash.

    Args:
        path: Path of the JSON pool file.
        evaluation_id: Evaluation instance name, used in error messages.

    Returns:
        ``(node_ids, sha256)``.

    Raises:
        ConfigError: If the file is missing, malformed, empty, or its hash does not match its ids.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"evaluation '{evaluation_id}': target_pool file not found: {path}")
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        ids = list(payload["node_ids"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"evaluation '{evaluation_id}': malformed target_pool file {path}: {exc}") from exc
    if not ids:
        raise ConfigError(f"evaluation '{evaluation_id}': target_pool {path} is empty")
    digest = pool_sha256(ids)
    if payload.get("sha256") not in (None, digest):
        raise ConfigError(f"evaluation '{evaluation_id}': target_pool {path} does not match its recorded sha256")
    return set(ids), digest


def check_targets_in_pool(targets: set[str], pool: set[str], *, evaluation_id: str) -> None:
    """Raise if a gold target is not in the pool (a pool must cover the gold it is used with).

    Raises:
        ConfigError: Listing up to five uncovered targets.
    """
    missing = sorted(targets - pool)
    if missing:
        raise ConfigError(
            f"evaluation '{evaluation_id}': {len(missing)} gold target(s) are outside target_pool, e.g. {missing[:5]}"
        )
