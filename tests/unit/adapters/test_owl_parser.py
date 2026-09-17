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
