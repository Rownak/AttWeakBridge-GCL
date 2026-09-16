"""Configuration layering: YAML deep-merge, ``${}`` interpolation, and hashing.

Layers are merged in a fixed order (base -> dataset -> pipeline -> experiment ->
``--config`` files -> ``--set`` overrides), each layer deep-merging over the
previous one with later layers winning and lists replaced wholesale (not
concatenated). This is deliberately Hydra's composition model minus its runtime:
one pure function, no plugin system, easy to reason about and test.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from cga.core.errors import ConfigError

_INTERP_PATTERN = re.compile(r"\$\{(paths\.[a-zA-Z0-9_.]+|env:[A-Za-z_][A-Za-z0-9_]*)\}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Nested dicts are merged key-by-key; any other type (including lists) in
    ``override`` replaces the corresponding value in ``base`` wholesale.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dict.

    Raises:
        ConfigError: If the file is missing or does not parse to a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping at top level: {path}")
    return data


def set_by_dotted_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``value`` at ``dotted_key`` (e.g. ``"gcl_train.train.epochs"``) in place.

    Intermediate mappings are created as needed, matching ``--set key.path=value``
    CLI semantics.
    """
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_set_value(raw: str) -> Any:
    """Parse a ``--set`` CLI value string as YAML scalar (int/float/bool/str)."""
    return yaml.safe_load(raw)


def interpolate(value: Any, resolved: dict[str, Any]) -> Any:
    """Recursively resolve ``${paths.x}`` and ``${env:VAR}`` references in ``value``.

    Raises:
        ConfigError: If a referenced path key or env var is undefined.
    """
    if isinstance(value, dict):
        return {k: interpolate(v, resolved) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, resolved) for v in value]
    if isinstance(value, str):
        return _interpolate_string(value, resolved)
    return value


def _interpolate_string(value: str, resolved: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        ref = match.group(1)
        if ref.startswith("env:"):
            var_name = ref[len("env:") :]
            if var_name not in os.environ:
                raise ConfigError(f"Environment variable '{var_name}' referenced by ${{{ref}}} is not set")
            return os.environ[var_name]
        # "paths.<key...>"
        parts = ref.split(".")
        node: Any = resolved
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Config interpolation '${{{ref}}}' does not resolve to a value")
            node = node[part]
        return str(node)

    return _INTERP_PATTERN.sub(replace, value)


def config_hash(resolved: dict[str, Any]) -> str:
    """Return the sha256 hex digest of a fully-resolved config.

    Uses a canonical JSON encoding (sorted keys, no extra whitespace) so that
    semantically identical configs always hash identically regardless of
    dict insertion order.
    """
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_config(
    *,
    base_path: str | Path,
    dataset_path: str | Path | None = None,
    pipeline_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    extra_config_paths: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Layer and resolve config files in the order the architecture spec defines.

    Order: base -> dataset -> pipeline -> experiment -> each extra config file,
    in order -> ``overrides`` (dotted-path -> value, applied last). Interpolation
    of ``${paths.*}``/``${env:*}`` runs once, after all layers are merged.

    Returns:
        The fully-resolved config dict (interpolated, not yet pydantic-validated).
    """
    merged: dict[str, Any] = {}
    for path in (base_path, dataset_path, pipeline_path, experiment_path):
        if path is not None:
            merged = deep_merge(merged, load_yaml(path))
    for path in extra_config_paths or []:
        merged = deep_merge(merged, load_yaml(path))

    resolved = interpolate(merged, merged)

    for dotted_key, value in (overrides or {}).items():
        set_by_dotted_path(resolved, dotted_key, value)

    return resolved
