"""Tests for the ATT&CK STIX bundle parser against a committed fixture."""

from __future__ import annotations

from pathlib import Path

from cga.adapters.attack_cwe.stix_parser import (
    iter_stix_edges,
    iter_stix_nodes,
    iter_stix_texts,
    parse_stix_bundle,
)

FIXTURE = Path(__file__).parent / "fixtures" / "stix_sample.json"


def test_parse_counts_dangling_relationships():
    result = parse_stix_bundle(FIXTURE)
    assert result.n_dangling_relationships == 1


def test_iter_nodes_excludes_relationships_and_assigns_ids():
    result = parse_stix_bundle(FIXTURE)
    nodes = list(iter_stix_nodes(result, graph_key="attack", graph_role="graph_1", source_version="v15"))
    node_ids = {n.node_id for n in nodes}
    assert node_ids == {
        "attack:attack-pattern--aaaa",
        "attack:attack-pattern--bbbb",
        "attack:attack-pattern--cccc",
    }
    aaaa = next(n for n in nodes if n.node_id == "attack:attack-pattern--aaaa")
    assert aaaa.external_id == "T0800"
    assert aaaa.name == "Spoof Reporting Message"
    assert aaaa.has_text is True
    assert aaaa.graph_role == "graph_1"

    cccc = next(n for n in nodes if n.node_id == "attack:attack-pattern--cccc")
    assert cccc.has_text is False


def test_iter_edges_drops_dangling_relationships():
    result = parse_stix_bundle(FIXTURE)
    edges = list(iter_stix_edges(result, graph_key="attack", graph_role="graph_1"))
    assert len(edges) == 1
    edge = edges[0]
    assert edge.src_id == "attack:attack-pattern--aaaa"
    assert edge.dst_id == "attack:attack-pattern--bbbb"
    assert edge.relation == "uses"
    assert edge.graph_role == "graph_1"


def test_iter_texts_assembles_variants():
    result = parse_stix_bundle(FIXTURE)
    text_variants = {
        "description": ["description"],
        "full_metadata": ["description", "detection"],
    }
    texts = list(iter_stix_texts(result, graph_key="attack", text_variants=text_variants))

    desc = next(
        t for t in texts if t.node_id == "attack:attack-pattern--aaaa" and t.text_variant == "description"
    )
    assert desc.text == "Adversaries may spoof reporting messages."

    full = next(
        t for t in texts if t.node_id == "attack:attack-pattern--aaaa" and t.text_variant == "full_metadata"
    )
    assert full.text == "Adversaries may spoof reporting messages.. Monitor for unexpected reporting values."

    empty = next(
        t for t in texts if t.node_id == "attack:attack-pattern--cccc" and t.text_variant == "description"
    )
    assert empty.text == ""
