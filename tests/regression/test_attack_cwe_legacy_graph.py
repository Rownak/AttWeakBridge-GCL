"""Regression tests: parser edges vs. the legacy stage-1 graph JSONs.

Not part of ``tests/unit`` — these depend on ``legacy/`` (frozen pre-refactor
reference code/data) and the full raw source files, neither of which is part
of the fast unit-test contract. Run explicitly:

    uv run pytest tests/regression -q

``legacy/datasets/{cwe,ics_attack}/*_graph.json`` were produced by the legacy
scripts (``3_CWE_grpah_create.py``, ``4_Attack_graph_create.py``) against the
same source files committed under ``data/raw/attack_cwe/``. Comparing the new
parsers' edges against those ``links`` arrays (as unordered source/target
pairs — the legacy CWE graph contains some duplicate links) is a cheap guard
against silently changing edge-extraction semantics during the refactor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cga.adapters.attack_cwe.cwe_parser import iter_cwe_edges, parse_cwe_catalog
from cga.adapters.attack_cwe.stix_parser import iter_stix_edges, parse_stix_bundle

REPO_ROOT = Path(__file__).parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "attack_cwe"
LEGACY_CWE_GRAPH = REPO_ROOT / "legacy" / "datasets" / "cwe" / "cwec_v4.12_graph.json"
LEGACY_ATTACK_GRAPH = REPO_ROOT / "legacy" / "datasets" / "ics_attack" / "attack_graph.json"
STIX_BUNDLE = RAW_DIR / "ics-attack.json"
CWE_XML = RAW_DIR / "cwec_v4.12.xml"

_missing_fixtures = not (
    LEGACY_CWE_GRAPH.is_file() and LEGACY_ATTACK_GRAPH.is_file() and STIX_BUNDLE.is_file() and CWE_XML.is_file()
)


def _legacy_links(path: Path) -> set[tuple[str, str]]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    return {(link["source"], link["target"]) for link in graph["links"]}


@pytest.mark.skipif(_missing_fixtures, reason="legacy graph JSONs or raw source files not present")
def test_stix_parser_edges_match_legacy_attack_graph():
    legacy_links = _legacy_links(LEGACY_ATTACK_GRAPH)

    result = parse_stix_bundle(STIX_BUNDLE)
    edges = iter_stix_edges(result, graph_key="attack", graph_role="graph_1")
    parsed_links = {(e.src_id.split(":", 1)[1], e.dst_id.split(":", 1)[1]) for e in edges}

    assert parsed_links == legacy_links


@pytest.mark.skipif(_missing_fixtures, reason="legacy graph JSONs or raw source files not present")
def test_cwe_parser_edges_match_legacy_cwe_graph():
    legacy_links = _legacy_links(LEGACY_CWE_GRAPH)

    result = parse_cwe_catalog(CWE_XML)
    edges = iter_cwe_edges(result, graph_key="cwe", graph_role="graph_2", edge_relations=["ChildOf"])
    parsed_links = {(e.src_id.split(":", 1)[1], e.dst_id.split(":", 1)[1]) for e in edges}

    assert parsed_links == legacy_links
