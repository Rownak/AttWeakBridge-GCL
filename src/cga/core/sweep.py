"""``cga sweep`` expansion: array-valued spec -> per-combination override dicts.

A sweep spec is a YAML file with a ``pipeline``, a ``dataset``, and a
``sweep:`` map from dotted config path to an array of values (Sprint 2
``dev_docs/sweep_experiment_plan.md`` §2). Keys are crossed (cartesian
product) by default; an optional ``zip:`` list of keys marks those keys as
paired by index instead -- each zipped key's array must be the same length,
and the zipped keys are expanded together as a single axis of the cross
product with the other (non-zipped) keys.

An optional top-level ``fixed:`` map (Sprint 2 Phase 2.2) holds single-valued
overrides that every point needs (e.g. pinning an upstream run's id, or an
epoch count) but that aren't part of the grid being compared. ``fixed:``
values are applied to every ``SweepPoint.overrides`` but excluded from
``sweep_point_id`` -- putting them in ``sweep:`` instead would work
functionally (a one-element array crossed against everything else), but bloats
the id with values that never vary, e.g. turning ``alpha0`` into
``alpha0-training_data_sampler_idrank_topk-k5-epochs100``.

This module only expands a spec into ``SweepPoint``s; it does not run
anything -- ``cga sweep``'s CLI command drives each point through the
existing ``build_config -> RunContext.create -> pipeline.run`` path (no
change to that path itself, per the plan's §2 "no change to core/config.py").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from cga.core.config import load_yaml
from cga.core.errors import ConfigError

_ZIP_KEY = "zip"


@dataclass(frozen=True)
class SweepSpec:
    """A parsed sweep spec file.

    Attributes:
        pipeline: Registered pipeline name to run for every combination.
        dataset: Dataset id to run against.
        axes: Dotted-path -> value-array map (excludes the ``zip:`` key itself).
        zipped_keys: Dotted paths paired by index instead of crossed, or empty.
        fixed: Dotted-path -> value map applied to every point's overrides,
            excluded from ``sweep_point_id`` (Sprint 2 Phase 2.2). Empty by
            default -- older spec files with no ``fixed:`` block are unaffected.
    """

    pipeline: str
    dataset: str
    axes: dict[str, list[Any]]
    zipped_keys: tuple[str, ...] = ()
    fixed: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "SweepSpec":
        """Load and validate a sweep spec YAML file.

        Raises:
            ConfigError: If ``pipeline``/``dataset``/``sweep`` are missing, a
                zipped key's array length disagrees with its peers, a zipped
                key is not itself present in ``sweep:``, or a ``fixed:`` key
                also appears in ``sweep:`` (ambiguous which one wins).
        """
        raw = load_yaml(path)
        for required in ("pipeline", "dataset", "sweep"):
            if required not in raw:
                raise ConfigError(f"Sweep spec {path} is missing required key '{required}'")

        sweep_block = dict(raw["sweep"])
        zipped_keys = tuple(sweep_block.pop(_ZIP_KEY, ()) or ())
        fixed = dict(raw.get("fixed") or {})

        for key in zipped_keys:
            if key not in sweep_block:
                raise ConfigError(
                    f"Sweep spec {path}: zip: names '{key}' but it has no matching sweep.{key} entry"
                )

        if zipped_keys:
            lengths = {key: len(sweep_block[key]) for key in zipped_keys}
            if len(set(lengths.values())) > 1:
                raise ConfigError(
                    f"Sweep spec {path}: zipped keys must have equal-length value arrays, got {lengths}"
                )

        overlap = set(fixed) & set(sweep_block)
        if overlap:
            raise ConfigError(f"Sweep spec {path}: key(s) {sorted(overlap)} appear in both 'fixed:' and 'sweep:'")

        return cls(
            pipeline=raw["pipeline"],
            dataset=raw["dataset"],
            axes=sweep_block,
            zipped_keys=zipped_keys,
            fixed=fixed,
        )


@dataclass(frozen=True)
class SweepPoint:
    """One expanded combination: the ``--set`` overrides for a single ``cga run`` call.

    Attributes:
        sweep_point_id: Short deterministic id computed from ``overrides``
            (e.g. ``rank_topk-k20``), recorded in the run's manifest.
        overrides: Dotted-path -> value map to apply on top of the base config.
    """

    sweep_point_id: str
    overrides: dict[str, Any] = field(default_factory=dict)


def _short_value(value: Any) -> str:
    """Render a swept value compactly for use in a ``sweep_point_id``."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _combination_id(combo: dict[str, Any], *, key_order: list[str]) -> str:
    """Return a short, deterministic id for one expanded combination.

    Built from each key's last dotted segment (e.g. ``training_data.sampler.k``
    -> ``k``) so a spec sweeping ``sampler.name`` and ``sampler.k`` together
    produces ids like ``rank_topk-k20`` rather than repeating the full path. A
    ``name``-leaf key (e.g. ``sampler.name``) sorts first and contributes only
    its bare value (``rank_topk``, not ``namerank_topk``), matching the plan's
    example ids; every other leaf follows in ``key_order`` -- the order the
    spec file itself declared those keys in ``sweep:`` -- rather than
    alphabetically, since alphabetical order can silently disagree with
    ``build_sampler_id``/``build_model_id``'s own param order (e.g.
    ``percentile``'s ``top_pct`` before ``bottom_pct``) and produce a
    ``sweep_point_id`` that looks plausible but never matches the artifact
    directory a downstream ``gcl_train``/``evaluate`` config needs to
    reference. Same combo -> same id across runs and Python versions, which
    is what lets ``sweep_point_id`` double as the stable label the comparison
    table sorts and joins on.
    """
    name_keys = [k for k in key_order if k.rsplit(".", 1)[-1] == "name"]
    other_keys = [k for k in key_order if k.rsplit(".", 1)[-1] != "name"]

    parts = [_short_value(combo[key]) for key in name_keys]
    for key in other_keys:
        leaf = key.rsplit(".", 1)[-1]
        parts.append(f"{leaf}{_short_value(combo[key])}")
    return "-".join(parts)


def expand_sweep(spec: SweepSpec) -> list[SweepPoint]:
    """Expand ``spec`` into one ``SweepPoint`` per combination.

    Zipped keys are treated as a single paired axis (index ``i`` of every
    zipped key's array together); every other key is crossed against every
    other axis and against the zipped axis, in cartesian-product order.
    ``spec.fixed`` (if any) is merged into every point's ``overrides`` after
    ``sweep_point_id`` is computed from the varying keys alone, so a pinned
    value (e.g. an upstream run id) reaches ``cga run`` without lengthening
    the id every point is labeled and compared by.
    """
    zipped = list(spec.zipped_keys)
    crossed_keys = [k for k in spec.axes if k not in zipped]
    key_order = zipped + crossed_keys

    axis_names: list[str] = []
    axis_values: list[list[Any]] = []

    if zipped:
        axis_names.append(_ZIP_KEY)
        paired = list(zip(*(spec.axes[k] for k in zipped)))
        axis_values.append(paired)

    for key in crossed_keys:
        axis_names.append(key)
        axis_values.append(list(spec.axes[key]))

    points: list[SweepPoint] = []
    for combo_values in product(*axis_values) if axis_values else [()]:
        combo: dict[str, Any] = {}
        for name, value in zip(axis_names, combo_values):
            if name == _ZIP_KEY:
                for key, v in zip(zipped, value):
                    combo[key] = v
            else:
                combo[name] = value
        sweep_point_id = _combination_id(combo, key_order=key_order)
        if spec.fixed:
            combo = {**combo, **spec.fixed}
        points.append(SweepPoint(sweep_point_id=sweep_point_id, overrides=combo))

    _assert_unique_ids(points)
    return points


def _assert_unique_ids(points: list[SweepPoint]) -> None:
    seen: dict[str, dict[str, Any]] = {}
    for point in points:
        if point.sweep_point_id in seen and seen[point.sweep_point_id] != point.overrides:
            raise ConfigError(
                f"Sweep produced duplicate sweep_point_id '{point.sweep_point_id}' for two different "
                f"combinations: {seen[point.sweep_point_id]} and {point.overrides}"
            )
        seen[point.sweep_point_id] = point.overrides


def sweep_point_hash(overrides: dict[str, Any]) -> str:
    """Return a short stable hash of ``overrides``, for disambiguating logs only.

    Not used as ``sweep_point_id`` itself (that stays human-readable) --
    available for callers that want a guaranteed-unique suffix.
    """
    import json

    canonical = json.dumps(overrides, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
