"""Training-protocol guard: the ``protocol: unsupervised`` top-level config key (Sprint 9).

Under the unsupervised protocol no gold alignment may influence training: not as a positive,
not as a softmax mask, not as a held-out marker, and not as an in-loop monitor. The guard lives
here, outside any one pipeline, so ``training_data`` and ``gcl_train`` refuse the same way and a
new refused option is added in one place. ``evaluate`` is deliberately not guarded: reading the
gold test pairs to score finished embeddings is the point of the protocol.
"""

from __future__ import annotations

from typing import Any

from cga.core.errors import ConfigError

PROTOCOLS = ("supervised", "unsupervised")
DEFAULT_PROTOCOL = "supervised"


def get_protocol(config: dict[str, Any]) -> str:
    """Return the config's ``protocol`` (``supervised`` when the key is absent).

    Raises:
        ConfigError: If the value is not one of :data:`PROTOCOLS`.
    """
    protocol = config.get("protocol", DEFAULT_PROTOCOL)
    if protocol not in PROTOCOLS:
        raise ConfigError(f"Unknown protocol '{protocol}'; expected one of {list(PROTOCOLS)}")
    return protocol


def is_unsupervised(config: dict[str, Any]) -> bool:
    """Return True when the config declares ``protocol: unsupervised``."""
    return get_protocol(config) == "unsupervised"


def refuse(pipeline: str, option: str, reason: str) -> None:
    """Raise the guard's error for one refused option.

    Args:
        pipeline: Pipeline that refuses (``training_data`` or ``gcl_train``).
        option: The offending config key and value, e.g. ``loss.name=infonce_full``.
        reason: Why the option reads gold alignments.

    Raises:
        ConfigError: Always.
    """
    raise ConfigError(f"protocol 'unsupervised': {pipeline} refuses {option} ({reason})")
