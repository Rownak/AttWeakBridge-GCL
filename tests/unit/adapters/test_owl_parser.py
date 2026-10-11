"""Tests for the shared OWL/XML ontology parser against committed NCIT/DOID fixtures."""

from __future__ import annotations

from pathlib import Path

from cga.adapters.ncit_doid.owl_parser import (
    iri_fragment,
    iter_owl_edges,
    iter_owl_nodes,
    iter_owl_texts,
    parse_owl_ontology,
)

NCIT_FIXTURE = Path(__file__).parent / "fixtures" / "ncit_sample.owl"
DOID_FIXTURE = Path(__file__).parent / "fixtures" / "doid_sample.owl"


def test_iri_fragment_after_hash():
    assert iri_fragment("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3224") == "C3224"


def test_iri_fragment_after_last_slash_when_no_hash():
    assert iri_fragment("http://purl.obolibrary.org/obo/DOID_4362") == "DOID_4362"


def test_parse_reads_all_classes():
    result = parse_owl_ontology(NCIT_FIXTURE)
    assert len(result.classes) == 3
    assert "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3224" in result.classes


def test_parse_ignores_equivalent_class_logical_axioms():
    # C3224 has an owl:equivalentClass/intersectionOf block; it must not be
    # picked up as a subClassOf parent or a text field.
    result = parse_owl_ontology(NCIT_FIXTURE)
    c3224 = result.classes["http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3224"]
    assert c3224.parent_iris == ["http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C7058"]


def test_iter_nodes_uses_iri_fragment_as_source_id():
    result = parse_owl_ontology(NCIT_FIXTURE)
    nodes = {n.node_id: n for n in iter_owl_nodes(result, graph_key="ncit", graph_role="graph_1")}
    assert "ncit:C3224" in nodes
    assert nodes["ncit:C3224"].name == "Melanoma"
    assert nodes["ncit:C3224"].graph_role == "graph_1"
    assert nodes["ncit:C3224"].external_id == "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3224"


def test_iter_edges_drops_dangling_parent():
    result = parse_owl_ontology(NCIT_FIXTURE)
    edges = list(iter_owl_edges(result, graph_key="ncit", graph_role="graph_1"))
    assert len(edges) == 1
    assert edges[0].src_id == "ncit:C3224"
    assert edges[0].dst_id == "ncit:C7058"
    assert edges[0].relation == "subClassOf"


def test_iter_texts_ncit_predicate_map():
    result = parse_owl_ontology(NCIT_FIXTURE)
    predicate_map = {"label": ["rdfs:label"], "synonym": ["P90"], "definition": ["P97", "P325"]}
    text_variants = {"label_synonyms_definition": ["label", "synonym", "definition"]}
    texts = {t.node_id: t for t in iter_owl_texts(
        result, graph_key="ncit", predicate_map=predicate_map, text_variants=text_variants
    )}
    melanoma_text = texts["ncit:C3224"].text
    assert "Melanoma" in melanoma_text
    assert "Malignant Melanoma" in melanoma_text
    assert "A malignant tumor composed of atypical, neoplastic melanocytes." in melanoma_text
    assert "A malignant neoplasm composed of melanocytes." in melanoma_text


def test_iter_texts_doid_predicate_map_differs_from_ncit():
    result = parse_owl_ontology(DOID_FIXTURE)
    predicate_map = {
        "label": ["rdfs:label"],
        "synonym": ["oboInOwl:hasExactSynonym"],
        "definition": ["obo:IAO_0000115"],
    }
    text_variants = {"label_synonyms_definition": ["label", "synonym", "definition"]}
    texts = {t.node_id: t for t in iter_owl_texts(
        result, graph_key="doid", predicate_map=predicate_map, text_variants=text_variants
    )}
    cervical_text = texts["doid:DOID_4362"].text
    assert "cervical cancer" in cervical_text
    assert "cervix cancer" in cervical_text
    assert "uterine cervical neoplasm" in cervical_text
    assert "A cervical cancer that is located in the cervix." in cervical_text


def test_doid_edges_drop_dangling_parent():
    result = parse_owl_ontology(DOID_FIXTURE)
    edges = list(iter_owl_edges(result, graph_key="doid", graph_role="graph_2"))
    assert len(edges) == 1
    assert edges[0].src_id == "doid:DOID_4362"
    assert edges[0].dst_id == "doid:DOID_120"


INTERSECTION_FIXTURE = Path(__file__).parent / "fixtures" / "ncit_intersection_sample.owl"
_NCIT_NS = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
_BOTH_SOURCES = ("subClassOf", "equivalent_intersection")


def _edge_pairs(edge_sources=None):
    result = parse_owl_ontology(INTERSECTION_FIXTURE)
    kwargs = {} if edge_sources is None else {"edge_sources": edge_sources}
    edges = list(iter_owl_edges(result, graph_key="ncit", graph_role="graph_1", **kwargs))
    assert all(e.relation == "subClassOf" for e in edges)
    return sorted((e.src_id, e.dst_id) for e in edges)


def test_parse_collects_named_intersection_members_and_skips_restrictions():
    result = parse_owl_ontology(INTERSECTION_FIXTURE)
    assert result.classes[_NCIT_NS + "C10"].intersection_parent_iris == [_NCIT_NS + "C1"]
    assert result.classes[_NCIT_NS + "C10"].parent_iris == []
    # C13 has an intersection axiom and a bare Restriction axiom; only the former counts.
    assert result.classes[_NCIT_NS + "C13"].intersection_parent_iris == [_NCIT_NS + "C2", _NCIT_NS + "C3"]


def test_default_edge_sources_ignore_intersection_members():
    # Default keeps pre-change behaviour (DOID and old configs): no direct subClassOf here.
    assert _edge_pairs() == [("ncit:C12", "ncit:C1")]


def test_intersection_edges_one_named_plus_restriction_gives_one_edge():
    pairs = _edge_pairs(_BOTH_SOURCES)
    assert [p for p in pairs if p[0] == "ncit:C10"] == [("ncit:C10", "ncit:C1")]


def test_intersection_edges_three_named_members_give_three_edges():
    pairs = _edge_pairs(_BOTH_SOURCES)
    assert [p for p in pairs if p[0] == "ncit:C11"] == [
        ("ncit:C11", "ncit:C1"),
        ("ncit:C11", "ncit:C2"),
        ("ncit:C11", "ncit:C3"),
    ]


def test_intersection_edges_do_not_duplicate_direct_parent():
    pairs = _edge_pairs(_BOTH_SOURCES)
    assert pairs.count(("ncit:C12", "ncit:C1")) == 1


def test_intersection_edges_ignore_bare_restriction_axiom_and_drop_dangling():
    pairs = _edge_pairs(_BOTH_SOURCES)
    assert [p for p in pairs if p[0] == "ncit:C13"] == [("ncit:C13", "ncit:C2"), ("ncit:C13", "ncit:C3")]
    assert not [p for p in pairs if p[0] == "ncit:C14"]
    assert len(pairs) == 1 + 3 + 1 + 2


def test_unknown_edge_source_raises():
    import pytest

    result = parse_owl_ontology(INTERSECTION_FIXTURE)
    with pytest.raises(ValueError, match="Unknown edge_sources"):
        list(iter_owl_edges(result, graph_key="ncit", graph_role="graph_1", edge_sources=["bogus"]))


def test_raw_sha256_changes_only_when_intersection_parents_present():
    result = parse_owl_ontology(INTERSECTION_FIXTURE)
    ncit = parse_owl_ontology(NCIT_FIXTURE)
    # Fixture C3224 has both; its hash must differ from the no-intersection hash of the same content.
    from cga.adapters.ncit_doid.owl_parser import _record_sha256

    c = ncit.classes[_NCIT_NS + "C3224"]
    assert c.raw_sha256 == _record_sha256(c.iri, c.parent_iris, c.texts, c.intersection_parent_iris)
    assert c.raw_sha256 != _record_sha256(c.iri, c.parent_iris, c.texts)
    assert result.classes[_NCIT_NS + "C1"].raw_sha256 == _record_sha256(
        _NCIT_NS + "C1", [], result.classes[_NCIT_NS + "C1"].texts
    )
