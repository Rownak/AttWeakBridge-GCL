"""Canonical node id construction and parsing.

A node_id is ``"{graph_key}:{source_id}"``, assigned once by the adapter from
the source record's own identifier (never a counter or loop index). The
``graph_key`` must not itself contain a colon, since parsing splits on the
first colon only — everything after it, including further colons, belongs to
``source_id`` (STIX ids and IRIs routinely contain colons).
"""

from __future__ import annotations

from cga.core.errors import ConfigError


def make_node_id(graph_key: str, source_id: str) -> str:
    """Build a canonical node_id from a graph key and the source's own id.

    Raises:
        ConfigError: If ``graph_key`` contains a colon or either part is empty.
    """
    if not graph_key:
        raise ConfigError("graph_key must be non-empty")
    if not source_id:
        raise ConfigError("source_id must be non-empty")
    if ":" in graph_key:
        raise ConfigError(f"graph_key must not contain ':': {graph_key!r}")
    return f"{graph_key}:{source_id}"


def parse_node_id(node_id: str) -> tuple[str, str]:
    """Split a node_id into ``(graph_key, source_id)``.

    Splits on the first colon only, so a ``source_id`` containing colons
    (STIX ids, IRIs) round-trips correctly.

    Raises:
        ConfigError: If ``node_id`` has no colon separator.
    """
    if ":" not in node_id:
        raise ConfigError(f"Invalid node_id (missing ':' separator): {node_id!r}")
    graph_key, source_id = node_id.split(":", 1)
    if not graph_key or not source_id:
        raise ConfigError(f"Invalid node_id (empty graph_key or source_id): {node_id!r}")
    return graph_key, source_id
