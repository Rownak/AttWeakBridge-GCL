"""Tests for the CWE XML catalog parser against a committed fixture."""

from __future__ import annotations

from pathlib import Path

from cga.adapters.attack_cwe.cwe_parser import (
    iter_cwe_edges,
    iter_cwe_nodes,
    iter_cwe_texts,
    parse_cwe_catalog,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cwe_sample.xml"


def test_parse_reads_version_and_weaknesses():
    result = parse_cwe_catalog(FIXTURE)
    assert result.version == "4.12"
    assert set(result.weaknesses) == {"1", "79", "80"}


def test_extended_description_joins_xhtml_paragraphs():
    result = parse_cwe_catalog(FIXTURE)
    assert result.weaknesses["1"].extended_description == "Extended paragraph one.Extended paragraph two."
    assert result.weaknesses["79"].extended_description == "Weakness 79 extended description."


def test_iter_nodes_assigns_ids_and_has_text():
    result = parse_cwe_catalog(FIXTURE)
    nodes = {n.node_id: n for n in iter_cwe_nodes(result, graph_key="cwe", graph_role="graph_2")}
    assert set(nodes) == {"cwe:1", "cwe:79", "cwe:80"}
    assert nodes["cwe:79"].external_id == "CWE-79"
    assert nodes["cwe:79"].graph_role == "graph_2"
    assert nodes["cwe:79"].has_text is True
    assert nodes["cwe:80"].has_text is False


def test_iter_edges_filters_by_nature():
    result = parse_cwe_catalog(FIXTURE)
    edges = list(iter_cwe_edges(result, graph_key="cwe", graph_role="graph_2", edge_relations=["ChildOf"]))
    pairs = {(e.src_id, e.dst_id, e.relation) for e in edges}
    assert pairs == {("cwe:1", "cwe:79", "ChildOf"), ("cwe:1", "cwe:80", "ChildOf")}


def test_iter_edges_drops_dangling_endpoint():
    # CWE 79's PeerOf points at CWE 200, which is not in the fixture catalog.
    result = parse_cwe_catalog(FIXTURE)
    edges = list(iter_cwe_edges(result, graph_key="cwe", graph_role="graph_2", edge_relations=["PeerOf"]))
    assert edges == []


def test_iter_texts_assembles_variants_and_skips_placeholder():
    result = parse_cwe_catalog(FIXTURE)
    text_variants = {"description": ["name", "description", "extended_description"]}
    texts = {t.node_id: t for t in iter_cwe_texts(result, graph_key="cwe", text_variants=text_variants)}
    assert texts["cwe:79"].text == "Improper Neutralization Weakness 79 description. Weakness 79 extended description."
    assert texts["cwe:80"].text == "No Description Weakness"
