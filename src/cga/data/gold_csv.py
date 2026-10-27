"""Generic gold-pair reader: one CSV of source/target external ids -> alignments + provenance.

Dataset-agnostic. A dataset adapter hands this module the id maps of its two
graphs (``external_id -> node_id``) and the selected CSV; the module resolves
each row, dedupes to distinct pairs, assigns a split from the YAML fractions,
and keeps one provenance record per kept pair. Unresolvable rows are dropped
and counted by reason, never guessed.

``alignments.parquet`` keeps its shared schema; provenance lives in the
sidecar ``alignment_provenance.parquet`` written by the ``ingest`` pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from cga.core.errors import ConfigError
from cga.core.io import sha256_file
from cga.data.schemas import AlignmentRecord

DROP_REASONS = ("source_not_in_graph", "target_not_in_graph", "duplicate")
_SPLIT_NAMES = ("train", "val", "test")


class GoldColumns(BaseModel):
    """CSV column names holding the source-side and target-side external ids."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


class GoldSplitConfig(BaseModel):
    """Split settings (``alignments.equivalence.split``).

    Attributes:
        fractions: ``[train, val, test]``. The default ``[0, 0, 1]`` makes the
            whole gold set ``test``: unsupervised training never reads gold,
            so nothing is held out.
        seed: Hash seed for non-default fractions.
        group_regex: Pattern whose group 1 is the split group (e.g. a
            technique's root id); ``None`` splits per source id.
    """

    model_config = ConfigDict(extra="forbid")

    fractions: list[float] = [0.0, 0.0, 1.0]
    seed: int = 42
    group_regex: str | None = None


class GoldAlignmentConfig(BaseModel):
    """``alignments.equivalence`` block of a dataset YAML for CSV-backed gold.

    Attributes:
        source: Key into ``sources`` selecting the gold file for this run.
        sources: ``{key: path relative to the dataset root}``. Each key is a
            separate gold set; exactly one is read per ingest.
        columns: Source/target id columns.
        split: Split settings.
        capec_column: CSV column with the intermediate CAPEC id.
        cited_column: CSV column with the technique id as cited by the source.
        edge_source_column: CSV column naming where the path came from.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    sources: dict[str, str]
    columns: GoldColumns
    split: GoldSplitConfig = GoldSplitConfig()
    capec_column: str = "capec_id"
    cited_column: str = "attack_id_cited"
    edge_source_column: str = "edge_source"

    def selected_path(self) -> str:
        """Return the relative path of the selected gold file.

        Raises:
            ConfigError: If ``source`` is not a key of ``sources``.
        """
        if self.source not in self.sources:
            raise ConfigError(
                f"alignments.equivalence.source={self.source!r} is not one of {sorted(self.sources)}"
            )
        return self.sources[self.source]


@dataclass
class GoldResult:
    """Output of :func:`read_gold_csv`.

    Attributes:
        alignments: One record per kept distinct pair, in first-seen order.
        provenance: One dict per kept pair, keyed like the provenance schema.
        report: ``gold_source``, file sha256, row/pair counts and drops by reason.
    """

    alignments: list[AlignmentRecord] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _validate_fractions(fractions: list[float]) -> None:
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ConfigError(f"split.fractions must be 3 non-negative values summing to 1.0, got {fractions}")


def group_id(external_id: str, group_regex: str | None) -> str:
    """Return the split group of ``external_id``: the regex's first group, or the id itself.

    Args:
        external_id: Source-side external id (e.g. ``T1003.001``).
        group_regex: Pattern whose group 1 (or whole match) is the group key
            (e.g. ``^(T\\d+)`` keeps a technique's root id). ``None`` = no grouping.

    Returns:
        The group key; ``external_id`` if the pattern does not match.
    """
    if not group_regex:
        return external_id
    m = re.match(group_regex, external_id)
    if not m:
        return external_id
    return m.group(1) if m.groups() else m.group(0)


def assign_grouped_split(
    group_ids: Iterable[str], *, fractions: list[float], seed: int
) -> dict[str, str]:
    """Assign each group to train/val/test by a seeded hash of its id.

    Hash-based (not shuffle-based) so a group's split depends only on
    ``(seed, group id)``, not on row order or on which other groups exist.
    With ``fractions=[0, 0, 1]`` every group is ``test``.

    Args:
        group_ids: Group keys (duplicates allowed).
        fractions: ``[train, val, test]`` proportions summing to 1.
        seed: Hash seed.

    Returns:
        ``{group_id: split}``.

    Raises:
        ConfigError: If ``fractions`` is malformed.
    """
    _validate_fractions(fractions)
    edges = [fractions[0], fractions[0] + fractions[1]]
    out: dict[str, str] = {}
    for g in set(group_ids):
        digest = hashlib.sha256(f"{seed}:{g}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        out[g] = "train" if u < edges[0] else "val" if u < edges[1] else "test"
    return out


def read_gold_csv(
    path: str | Path,
    *,
    cfg: GoldAlignmentConfig,
    source_ids: Mapping[str, str],
    target_ids: Mapping[str, str],
    root: str | Path,
) -> GoldResult:
    """Read one gold CSV into split alignments with provenance.

    Args:
        path: Absolute path of the selected CSV.
        cfg: Gold config (columns, split, provenance column names).
        source_ids: ``external_id -> node_id`` for the source graph.
        target_ids: ``external_id -> node_id`` for the target graph.
        root: Dataset root; ``source_file`` in provenance is relative to it.

    Returns:
        A :class:`GoldResult`. Rows whose source or target id is unresolved are
        dropped (counted as ``source_not_in_graph`` / ``target_not_in_graph``);
        rows repeating a kept pair only add to that pair's provenance and are
        counted as ``duplicate``.

    Raises:
        ConfigError: If a configured column is missing from the CSV.
    """
    path = Path(path)
    sha = sha256_file(path)
    rel = path.relative_to(Path(root)).as_posix() if Path(root) in path.parents else path.name

    drops: Counter[str] = Counter({r: 0 for r in DROP_REASONS})
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    n_rows = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in (cfg.columns.source, cfg.columns.target):
            if col not in (reader.fieldnames or []):
                raise ConfigError(f"Gold CSV {path} has no column {col!r}")
        for row in reader:
            n_rows += 1
            src_ext = row[cfg.columns.source].strip()
            tgt_ext = row[cfg.columns.target].strip()
            if src_ext not in source_ids:
                drops["source_not_in_graph"] += 1
                continue
            if tgt_ext not in target_ids:
                drops["target_not_in_graph"] += 1
                continue
            key = (src_ext, tgt_ext)
            if key in pairs:
                drops["duplicate"] += 1
            entry = pairs.setdefault(key, {"capec": [], "cited": [], "edge": [], "n": 0})
            entry["n"] += 1
            for k, col in (
                ("capec", cfg.capec_column),
                ("cited", cfg.cited_column),
                ("edge", cfg.edge_source_column),
            ):
                v = (row.get(col) or "").strip()
                if v and v not in entry[k]:
                    entry[k].append(v)

    group_of = {s: group_id(s, cfg.split.group_regex) for s, _ in pairs}
    group_split = assign_grouped_split(
        group_of.values(), fractions=cfg.split.fractions, seed=cfg.split.seed
    )

    result = GoldResult()
    for (src_ext, tgt_ext), entry in pairs.items():
        n1, n2 = source_ids[src_ext], target_ids[tgt_ext]
        aid = hashlib.sha1(f"{n1}|{n2}|equivalence".encode("utf-8")).hexdigest()
        result.alignments.append(
            AlignmentRecord(
                alignment_id=aid,
                node_1_id=n1,
                node_2_id=n2,
                relation="equivalence",
                score=1.0,
                split=group_split[group_of[src_ext]],
            )
        )
        result.provenance.append(
            {
                "alignment_id": aid,
                "gold_source": cfg.source,
                "source_file": rel,
                "source_sha256": sha,
                "edge_source": "|".join(sorted(entry["edge"])),
                "capec_ids": sorted(entry["capec"]),
                "attack_id_cited": "|".join(sorted(entry["cited"])),
                "n_csv_rows": entry["n"],
            }
        )

    by_split = Counter(a.split for a in result.alignments)
    result.report = {
        "gold_source": cfg.source,
        "source_file": rel,
        "source_sha256": sha,
        "n_csv_rows": n_rows,
        "n_pairs_kept": len(result.alignments),
        "n_techniques": len({a.node_1_id for a in result.alignments}),
        "dropped": {r: drops[r] for r in DROP_REASONS},
        "n_pairs_by_split": {s: by_split.get(s, 0) for s in _SPLIT_NAMES},
    }
    return result
