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


_NESTED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7" xmlns:xhtml="http://www.w3.org/1999/xhtml" Version="4.20">
  <Weaknesses>
    <Weakness ID="284" Name="Plain Description" Abstraction="Pillar">
      <Description>The product does not restrict access.</Description>
      <Detection_Methods>
        <Detection_Method>
          <Method>Automated Static Analysis</Method>
          <Description><xhtml:p>Detected by static analysis tools.</xhtml:p></Description>
        </Detection_Method>
      </Detection_Methods>
      <Potential_Mitigations>
        <Mitigation>
          <Phase>Architecture and Design</Phase>
          <Description><xhtml:p>Compartmentalize the system.</xhtml:p></Description>
        </Mitigation>
      </Potential_Mitigations>
    </Weakness>
    <Weakness ID="120" Name="Paragraph Description">
      <Description><xhtml:p>First.</xhtml:p><xhtml:p>Second.</xhtml:p></Description>
    </Weakness>
    <Weakness ID="999" Name="No Description">
      <Potential_Mitigations>
        <Mitigation><Description><xhtml:p>Mitigation only.</xhtml:p></Description></Mitigation>
      </Potential_Mitigations>
    </Weakness>
  </Weaknesses>
</Weakness_Catalog>
"""


def test_description_ignores_nested_mitigation_and_detection_descriptions(tmp_path):
    path = tmp_path / "nested.xml"
    path.write_text(_NESTED_XML, encoding="utf-8")
    weaknesses = parse_cwe_catalog(path).weaknesses
    assert weaknesses["284"].description == "The product does not restrict access."
    # The nested fields still come from their own parsers.
    assert "Compartmentalize the system." in weaknesses["284"].potential_mitigations
    assert "Detected by static analysis tools." in weaknesses["284"].detection_methods


def test_description_with_xhtml_paragraphs_is_still_joined(tmp_path):
    path = tmp_path / "nested.xml"
    path.write_text(_NESTED_XML, encoding="utf-8")
    assert parse_cwe_catalog(path).weaknesses["120"].description == "First.Second."


def test_missing_description_is_na_not_nested_text(tmp_path):
    path = tmp_path / "nested.xml"
    path.write_text(_NESTED_XML, encoding="utf-8")
    weakness = parse_cwe_catalog(path).weaknesses["999"]
    assert weakness.description == "N/A"
    assert weakness.extended_description == "N/A"
