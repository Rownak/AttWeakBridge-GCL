"""``ncit_doid`` dataset adapter: NCI Thesaurus + Disease Ontology (both OWL/XML) -> canonical records.

Ground truth (``refs_equiv/{train,test}.tsv`` + ``test.cands.tsv``) is available
for this dataset, unlike ``attack_cwe`` (target_architecture.md §4.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict

from cga.adapters.base import DatasetAdapter, GraphSpec, register_adapter
from cga.adapters.ncit_doid.owl_parser import (
    iter_owl_edges,
    iter_owl_nodes,
    iter_owl_texts,
    parse_owl_ontology,
)
from cga.adapters.ncit_doid.refs_parser import iter_refs_alignments
from cga.core.errors import MissingInputError
from cga.core.io import sha256_file
from cga.data.schemas import AlignmentRecord, EdgeRecord, NodeRecord, TextRecord


class OntologyConfig(BaseModel):
    """Adapter config for one ontology side (NCIT or DOID)."""

    model_config = ConfigDict(extra="forbid")

    file: str
    id_style: str = "fragment"
    edge_predicates: list[str] = ["rdfs:subClassOf"]
    # Which parent pointers become edges (see owl_parser.iter_owl_edges). NCIT needs both:
    # about half its classes state their parents only via equivalentClass intersections.
    edge_sources: list[str] = ["subClassOf"]
    predicate_map: dict[str, list[str]]


class PruningConfig(BaseModel):
    """Ontology pruning config; only ``none`` is implemented (§12 backlog)."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "none"


class NcitDoidAdapterConfig(BaseModel):
    """Top-level ``adapter_config`` block for ``ncit_doid`` (configs/datasets/ncit_doid.yaml §7.4)."""

    model_config = ConfigDict(extra="forbid")

    ncit: OntologyConfig
    doid: OntologyConfig
    pruning: PruningConfig = PruningConfig()


class AlignmentSourceConfig(BaseModel):
    """``alignments.equivalence`` block: paths relative to the dataset root."""

    model_config = ConfigDict(extra="forbid")

    train: str | None = None
    test: str | None = None
    candidates: str | None = None


@register_adapter("ncit_doid")
class NcitDoidAdapter(DatasetAdapter):
    """Reads the NCIT and DOID OWL ontologies plus their gold alignment TSVs into canonical records."""

    ConfigModel = NcitDoidAdapterConfig
    graphs = (
        GraphSpec(key="ncit", role="graph_1", display_name="NCI Thesaurus"),
        GraphSpec(key="doid", role="graph_2", display_name="Disease Ontology"),
    )

    def __init__(
        self,
        config: NcitDoidAdapterConfig,
        *,
        root: str | Path,
        text_variants: dict,
        alignments: dict | None = None,
    ) -> None:
        """Build the adapter from its validated config block.

        Args:
            config: Validated ``adapter_config`` (``ncit`` + ``doid`` + ``pruning`` blocks).
            root: Directory containing the dataset's raw source files.
            text_variants: ``{variant_name: [role, ...]}`` from the dataset
                YAML's ``text_variants`` block — shared across both ontologies
                since roles (``label``, ``synonym``, ``definition``) are generic.
            alignments: The dataset YAML's ``alignments.equivalence`` block
                (``{train, test, candidates}`` relative paths), or ``None`` if absent.
        """
        self._config = config
        self._root = Path(root)
        self._text_variants = text_variants
        self._ncit_path = self._root / config.ncit.file
        self._doid_path = self._root / config.doid.file
        self._alignments = AlignmentSourceConfig(**alignments) if alignments else None

    def validate_sources(self) -> None:
        """Raise ``MissingInputError`` with actionable paths if a source file is absent."""
        for path in (self._ncit_path, self._doid_path):
            if not path.is_file():
                raise MissingInputError(f"ncit_doid adapter: required source file not found: {path}")
        if self._alignments:
            for rel in (self._alignments.train, self._alignments.test, self._alignments.candidates):
                if rel and not (self._root / rel).is_file():
                    raise MissingInputError(
                        f"ncit_doid adapter: required alignment file not found: {self._root / rel}"
                    )

    def iter_nodes(self) -> Iterator[NodeRecord]:
        ncit = parse_owl_ontology(self._ncit_path)
        yield from iter_owl_nodes(ncit, graph_key="ncit", graph_role="graph_1")
        doid = parse_owl_ontology(self._doid_path)
        yield from iter_owl_nodes(doid, graph_key="doid", graph_role="graph_2")

    def iter_edges(self) -> Iterator[EdgeRecord]:
        ncit = parse_owl_ontology(self._ncit_path)
        yield from iter_owl_edges(
            ncit, graph_key="ncit", graph_role="graph_1", edge_sources=self._config.ncit.edge_sources
        )
        doid = parse_owl_ontology(self._doid_path)
        yield from iter_owl_edges(
            doid, graph_key="doid", graph_role="graph_2", edge_sources=self._config.doid.edge_sources
        )

    def iter_texts(self) -> Iterator[TextRecord]:
        ncit = parse_owl_ontology(self._ncit_path)
        yield from iter_owl_texts(
            ncit,
            graph_key="ncit",
            predicate_map=self._config.ncit.predicate_map,
            text_variants=self._text_variants,
        )
        doid = parse_owl_ontology(self._doid_path)
        yield from iter_owl_texts(
            doid,
            graph_key="doid",
            predicate_map=self._config.doid.predicate_map,
            text_variants=self._text_variants,
        )

    def iter_alignments(self) -> Iterator[AlignmentRecord]:
        if not self._alignments:
            return iter(())
        train_path = self._root / self._alignments.train if self._alignments.train else None
        test_path = self._root / self._alignments.test if self._alignments.test else None
        candidates_path = (
            self._root / self._alignments.candidates if self._alignments.candidates else None
        )
        return iter_refs_alignments(
            graph_1_key="ncit",
            graph_2_key="doid",
            train_path=train_path,
            test_path=test_path,
            candidates_path=candidates_path,
        )

    def source_fingerprint(self) -> dict[str, str]:
        fingerprint = {
            str(self._ncit_path.relative_to(self._root)): sha256_file(self._ncit_path),
            str(self._doid_path.relative_to(self._root)): sha256_file(self._doid_path),
        }
        if self._alignments:
            for rel in (self._alignments.train, self._alignments.test, self._alignments.candidates):
                if rel:
                    fingerprint[rel] = sha256_file(self._root / rel)
        return fingerprint
