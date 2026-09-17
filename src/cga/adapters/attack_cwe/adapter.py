"""``attack_cwe`` dataset adapter: MITRE ATT&CK (STIX) + MITRE CWE (XML) -> canonical records.

No alignment ground truth is available for this dataset (target_architecture.md
§4.2); evaluation is unsupervised only until a mapping is supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict

from cga.adapters.attack_cwe.cwe_parser import (
    iter_cwe_edges,
    iter_cwe_nodes,
    iter_cwe_texts,
    parse_cwe_catalog,
)
from cga.adapters.attack_cwe.stix_parser import (
    iter_stix_edges,
    iter_stix_nodes,
    iter_stix_texts,
    parse_stix_bundle,
)
from cga.adapters.base import DatasetAdapter, GraphSpec, register_adapter
from cga.core.errors import MissingInputError
from cga.core.io import sha256_file
from cga.data.schemas import AlignmentRecord, EdgeRecord, NodeRecord, TextRecord


class AttackConfig(BaseModel):
    """Adapter config for the ATT&CK (``graph_1``) side."""

    model_config = ConfigDict(extra="forbid")

    file: str
    edge_from_relationships: bool = True
    include_types: list[str] | None = None


class CweConfig(BaseModel):
    """Adapter config for the CWE (``graph_2``) side."""

    model_config = ConfigDict(extra="forbid")

    file: str
    version: str
    edge_relations: list[str] = ["ChildOf"]


class AttackCweAdapterConfig(BaseModel):
    """Top-level ``adapter_config`` block for ``attack_cwe`` (configs/datasets/attack_cwe.yaml §7.3)."""

    model_config = ConfigDict(extra="forbid")

    attack: AttackConfig
    cwe: CweConfig


@register_adapter("attack_cwe")
class AttackCweAdapter(DatasetAdapter):
    """Reads an ATT&CK STIX bundle and a CWE XML catalog into canonical records."""

    ConfigModel = AttackCweAdapterConfig
    graphs = (
        GraphSpec(key="attack", role="graph_1", display_name="MITRE ATT&CK"),
        GraphSpec(key="cwe", role="graph_2", display_name="MITRE CWE"),
    )

    def __init__(self, config: AttackCweAdapterConfig, *, root: str | Path, text_variants: dict) -> None:
        """Build the adapter from its validated config block.

        Args:
            config: Validated ``adapter_config`` (``attack`` + ``cwe`` blocks).
            root: Directory containing the dataset's raw source files
                (``dataset.root`` in the dataset YAML).
            text_variants: ``{variant_name: {attack: [...], cwe: [...]}}`` from
                the dataset YAML's ``text_variants`` block.
        """
        self._config = config
        self._root = Path(root)
        self._text_variants = text_variants
        self._attack_path = self._root / config.attack.file
        self._cwe_path = self._root / config.cwe.file

    def validate_sources(self) -> None:
        """Raise ``MissingInputError`` with actionable paths if a source file is absent."""
        for path in (self._attack_path, self._cwe_path):
            if not path.is_file():
                raise MissingInputError(f"attack_cwe adapter: required source file not found: {path}")

    def iter_nodes(self) -> Iterator[NodeRecord]:
        stix = parse_stix_bundle(self._attack_path)
        yield from iter_stix_nodes(
            stix, graph_key="attack", graph_role="graph_1", source_version=""
        )
        cwe = parse_cwe_catalog(self._cwe_path)
        yield from iter_cwe_nodes(cwe, graph_key="cwe", graph_role="graph_2")

    def iter_edges(self) -> Iterator[EdgeRecord]:
        stix = parse_stix_bundle(self._attack_path)
        yield from iter_stix_edges(stix, graph_key="attack", graph_role="graph_1")
        cwe = parse_cwe_catalog(self._cwe_path)
        yield from iter_cwe_edges(
            cwe,
            graph_key="cwe",
            graph_role="graph_2",
            edge_relations=self._config.cwe.edge_relations,
        )

    def iter_texts(self) -> Iterator[TextRecord]:
        attack_variants = {
            variant: fields["attack"]
            for variant, fields in self._text_variants.items()
            if "attack" in fields
        }
        stix = parse_stix_bundle(self._attack_path)
        yield from iter_stix_texts(stix, graph_key="attack", text_variants=attack_variants)

        cwe_variants = {
            variant: fields["cwe"] for variant, fields in self._text_variants.items() if "cwe" in fields
        }
        cwe = parse_cwe_catalog(self._cwe_path)
        yield from iter_cwe_texts(cwe, graph_key="cwe", text_variants=cwe_variants)

    def iter_alignments(self) -> Iterator[AlignmentRecord]:
        return iter(())

    def source_fingerprint(self) -> dict[str, str]:
        return {
            str(self._attack_path.relative_to(self._root)): sha256_file(self._attack_path),
            str(self._cwe_path.relative_to(self._root)): sha256_file(self._cwe_path),
        }
