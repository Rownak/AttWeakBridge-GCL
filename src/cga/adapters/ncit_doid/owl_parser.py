"""OWL/XML ontology -> canonical records, shared by NCIT and DOID.

Streaming parse (``xml.etree.ElementTree.iterparse``) of ``owl:Class`` elements,
reading each class's ``rdf:about`` IRI, its direct ``rdfs:subClassOf
rdf:resource`` pointers, and the *named* members of an ``owl:equivalentClass``
-> ``owl:intersectionOf`` definition. A named intersection member is an entailed
direct superclass, so it is equivalent to a ``subClassOf`` pointer; about half
of NCIT's classes record their parents only this way (see
dev_docs/ncit_edge_extraction_analysis.md). ``owl:Restriction`` members (typed,
non-is-a relations) are still not interpreted (target_architecture.md §4.3,
backlog §12). Text extraction is driven entirely by a per-ontology
**predicate map** passed in from the dataset config, since NCIT and DOID use
different annotation-property vocabularies for the same text roles (NCIT's
``P90``/``P97``/``P325`` vs. DOID's ``oboInOwl:hasExactSynonym``/
``IAO_0000115``) — this module has no ontology-specific knowledge itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

from cga.data.ids import make_node_id
from cga.data.schemas import EdgeRecord, NodeRecord, TextRecord

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET

_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_OWL_NS = "http://www.w3.org/2002/07/owl#"
_RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
_OBO_IN_OWL_NS = "http://www.geneontology.org/formats/oboInOwl#"
_OBO_NS = "http://purl.obolibrary.org/obo/"

_ABOUT = f"{{{_RDF_NS}}}about"
_RESOURCE = f"{{{_RDF_NS}}}resource"
_OWL_CLASS = f"{{{_OWL_NS}}}Class"
_RDFS_SUBCLASSOF = f"{{{_RDFS_NS}}}subClassOf"
_OWL_EQUIVALENT_CLASS = f"{{{_OWL_NS}}}equivalentClass"
_OWL_INTERSECTION_OF = f"{{{_OWL_NS}}}intersectionOf"
_RDF_DESCRIPTION = f"{{{_RDF_NS}}}Description"

EDGE_SOURCE_SUBCLASSOF = "subClassOf"
EDGE_SOURCE_EQUIVALENT_INTERSECTION = "equivalent_intersection"
_KNOWN_EDGE_SOURCES = (EDGE_SOURCE_SUBCLASSOF, EDGE_SOURCE_EQUIVALENT_INTERSECTION)

_WELL_KNOWN_PREFIXES = {
    "rdf": _RDF_NS,
    "rdfs": _RDFS_NS,
    "owl": _OWL_NS,
    "oboInOwl": _OBO_IN_OWL_NS,
    "obo": _OBO_NS,
}


def _resolve_predicate_tag(predicate: str, *, default_ns: str) -> str:
    """Resolve a ``prefix:localname`` or bare ``localname`` predicate to a Clark-notation tag.

    A predicate with a recognized prefix (``rdfs:label``, ``oboInOwl:hasExactSynonym``,
    ``obo:IAO_0000115``) resolves against that vocabulary's fixed namespace. A
    bare name (``P90``, ``P97``) is ontology-specific and resolves against
    ``default_ns`` — the ontology's own namespace, matching how NCIT declares
    its custom annotation properties in its own ``Thesaurus.owl#`` vocabulary.
    """
    if ":" in predicate:
        prefix, local = predicate.split(":", 1)
        ns = _WELL_KNOWN_PREFIXES.get(prefix, default_ns)
        return f"{{{ns}}}{local}"
    return f"{{{default_ns}}}{predicate}"


@dataclass
class OwlClassRecord:
    """One parsed ``owl:Class`` element.

    Attributes:
        iri: The class's ``rdf:about`` IRI.
        parent_iris: ``rdfs:subClassOf`` target IRIs (direct parents only).
        intersection_parent_iris: Named members of ``owl:equivalentClass`` ->
            ``owl:intersectionOf`` definitions (``owl:Restriction`` members
            excluded). Kept apart from ``parent_iris`` so edge extraction can
            be switched per ontology via ``edge_sources``.
        texts: ``{predicate_tag: [literal values]}`` for every annotation
            property this class carries, keyed by resolved Clark-notation tag.
        raw_sha256: Hash of a canonical serialization of this class's content,
            for change detection.
    """

    iri: str
    parent_iris: list[str] = field(default_factory=list)
    intersection_parent_iris: list[str] = field(default_factory=list)
    texts: dict[str, list[str]] = field(default_factory=dict)
    raw_sha256: str = ""


@dataclass
class OwlParseResult:
    """Parsed contents of one OWL/XML ontology file.

    Attributes:
        classes: Every ``owl:Class`` keyed by its IRI.
        default_ns: The ontology's own default namespace (used to resolve
            bare, ontology-specific predicate names like ``P90``).
    """

    classes: dict[str, OwlClassRecord]
    default_ns: str


def _default_namespace(path: str | Path) -> str:
    """Return the ``xmlns`` (no prefix) namespace declared on the document root."""
    for event, elem in _iterparse_events(path, events=("start-ns",)):
        prefix, uri = elem
        if prefix == "":
            return uri
    return ""


def _iterparse_events(path, events):
    import xml.etree.ElementTree as ET

    return ET.iterparse(str(path), events=events)


def parse_owl_ontology(path: str | Path) -> OwlParseResult:
    """Stream-parse an OWL/XML file into ``OwlClassRecord``s keyed by IRI.

    Uses ``iterparse`` so the full 50MB+ NCIT file is never held as one DOM
    tree; each ``owl:Class`` element is processed and then cleared.
    """
    import xml.etree.ElementTree as ET

    path = Path(path)
    default_ns = _default_namespace(path)

    classes: dict[str, OwlClassRecord] = {}
    context = ET.iterparse(str(path), events=("end",))
    for _, elem in context:
        if elem.tag != _OWL_CLASS:
            continue
        iri = elem.get(_ABOUT)
        if not iri:
            # Anonymous class (e.g. inside equivalentClass): leave it intact. Clearing it
            # here would erase the intersectionOf its named parent class still has to read.
            continue

        parent_iris: list[str] = []
        intersection_parent_iris: list[str] = []
        texts: dict[str, list[str]] = {}
        for child in elem:
            if child.tag == _RDFS_SUBCLASSOF:
                resource = child.get(_RESOURCE)
                if resource:
                    parent_iris.append(resource)
                continue
            if child.tag == _OWL_EQUIVALENT_CLASS:
                # A class may carry several equivalentClass axioms (C27691 has an
                # intersection and a bare Restriction); only intersections count.
                intersection_parent_iris.extend(_named_intersection_members(child))
                continue
            if child.text and child.text.strip():
                texts.setdefault(child.tag, []).append(child.text.strip())

        classes[iri] = OwlClassRecord(
            iri=iri,
            parent_iris=parent_iris,
            intersection_parent_iris=intersection_parent_iris,
            texts=texts,
            raw_sha256=_record_sha256(iri, parent_iris, texts, intersection_parent_iris),
        )
        elem.clear()

    return OwlParseResult(classes=classes, default_ns=default_ns)


def _named_intersection_members(equivalent_class: ET.Element) -> list[str]:
    """Return IRIs of named classes in an ``owl:equivalentClass`` intersection definition.

    Args:
        equivalent_class: An ``owl:equivalentClass`` element.

    Returns:
        ``rdf:about`` IRIs of the ``rdf:Description`` / ``owl:Class`` members of
        its ``owl:intersectionOf``. Empty if the axiom is not an intersection
        (e.g. a bare ``owl:Restriction``); ``owl:Restriction`` members are skipped.
    """
    members: list[str] = []
    for expression in equivalent_class:
        if expression.tag != _OWL_CLASS:
            continue
        for intersection in expression:
            if intersection.tag != _OWL_INTERSECTION_OF:
                continue
            for member in intersection:
                if member.tag in (_RDF_DESCRIPTION, _OWL_CLASS):
                    about = member.get(_ABOUT)
                    if about:
                        members.append(about)
    return members


def _record_sha256(
    iri: str,
    parent_iris: list[str],
    texts: dict[str, list[str]],
    intersection_parent_iris: list[str] | None = None,
) -> str:
    payload = {"iri": iri, "parents": sorted(parent_iris), "texts": {k: sorted(v) for k, v in texts.items()}}
    if intersection_parent_iris:
        # Added only when present so hashes of classes without it (all of DOID) stay unchanged.
        payload["intersection_parents"] = sorted(intersection_parent_iris)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iri_fragment(iri: str) -> str:
    """Return the fragment (after ``#``) or, if none, the final ``/`` segment of an IRI."""
    if "#" in iri:
        return iri.rsplit("#", 1)[1]
    return iri.rstrip("/").rsplit("/", 1)[1]


def iter_owl_nodes(
    result: OwlParseResult, *, graph_key: str, graph_role: str, label_predicate: str = "rdfs:label"
) -> Iterator[NodeRecord]:
    """Yield one ``NodeRecord`` per parsed class, keyed by the IRI fragment as ``source_id``."""
    label_tag = _resolve_predicate_tag(label_predicate, default_ns=result.default_ns)
    for iri, record in result.classes.items():
        labels = record.texts.get(label_tag) or []
        name = labels[0] if labels else ""
        yield NodeRecord(
            node_id=make_node_id(graph_key, iri_fragment(iri)),
            graph_key=graph_key,
            graph_role=graph_role,
            source_id=iri_fragment(iri),
            external_id=iri,
            name=name,
            node_type="owl:Class",
            metadata=json.dumps({}, sort_keys=True),
            has_text=bool(labels) or bool(record.texts),
            source_version="",
            raw_sha256=record.raw_sha256,
        )


def iter_owl_edges(
    result: OwlParseResult,
    *,
    graph_key: str,
    graph_role: str,
    edge_sources: Sequence[str] = (EDGE_SOURCE_SUBCLASSOF,),
) -> Iterator[EdgeRecord]:
    """Yield one ``EdgeRecord`` per parent pointer to a known class.

    Direction is child -> parent: ``src_id`` is the subclass, ``dst_id`` its
    parent, matching how the pointer reads in the source. Pointers to an IRI
    outside this ontology's own class set are skipped. Every edge carries
    ``relation="subClassOf"`` whatever its source, and a parent reachable
    through more than one source is emitted once.

    Args:
        result: Parsed ontology.
        graph_key: Graph key used to build node ids.
        graph_role: ``graph_1`` or ``graph_2``.
        edge_sources: Which parent pointers become edges: ``"subClassOf"``
            (direct ``rdfs:subClassOf``) and/or ``"equivalent_intersection"``
            (named members of an ``equivalentClass`` intersection).

    Raises:
        ValueError: If ``edge_sources`` names an unknown source.
    """
    unknown = set(edge_sources) - set(_KNOWN_EDGE_SOURCES)
    if unknown:
        raise ValueError(f"Unknown edge_sources {sorted(unknown)}; expected a subset of {list(_KNOWN_EDGE_SOURCES)}")

    for iri, record in result.classes.items():
        parents: list[str] = []
        if EDGE_SOURCE_SUBCLASSOF in edge_sources:
            parents.extend(record.parent_iris)
        if EDGE_SOURCE_EQUIVALENT_INTERSECTION in edge_sources:
            parents.extend(record.intersection_parent_iris)
        for parent_iri in dict.fromkeys(parents):
            if parent_iri not in result.classes:
                continue
            src_id = make_node_id(graph_key, iri_fragment(iri))
            dst_id = make_node_id(graph_key, iri_fragment(parent_iri))
            edge_id = hashlib.sha1(f"{src_id}|{dst_id}|subClassOf".encode("utf-8")).hexdigest()
            yield EdgeRecord(
                edge_id=edge_id,
                src_id=src_id,
                dst_id=dst_id,
                graph_role=graph_role,
                relation="subClassOf",
                directed=True,
            )


def iter_owl_texts(
    result: OwlParseResult,
    *,
    graph_key: str,
    predicate_map: dict[str, list[str]],
    text_variants: dict[str, list[str]],
) -> Iterator[TextRecord]:
    """Yield ``TextRecord`` rows per ``text_variants`` (variant -> ordered text roles).

    Args:
        predicate_map: ``{role: [predicate names]}`` (e.g. ``{"label":
            ["rdfs:label"], "synonym": ["P90"]}``) — the dataset config's
            per-ontology mapping from a generic text role to this ontology's
            own predicate name(s).
        text_variants: ``{variant: [role, ...]}`` — how roles compose into a
            named text variant (e.g. ``"label_definition": ["label",
            "definition"]``).
    """
    role_tags = {
        role: [_resolve_predicate_tag(p, default_ns=result.default_ns) for p in predicates]
        for role, predicates in predicate_map.items()
    }

    for iri, record in result.classes.items():
        node_id = make_node_id(graph_key, iri_fragment(iri))
        role_values: dict[str, list[str]] = {}
        for role, tags in role_tags.items():
            values: list[str] = []
            for tag in tags:
                values.extend(record.texts.get(tag, []))
            role_values[role] = values

        for variant, roles in text_variants.items():
            parts: list[str] = []
            for role in roles:
                parts.extend(role_values.get(role, []))
            text = "; ".join(parts)
            yield TextRecord(
                node_id=node_id,
                text_variant=variant,
                text=text,
                text_sha256=_sha256_text(text),
                n_chars=len(text),
            )
