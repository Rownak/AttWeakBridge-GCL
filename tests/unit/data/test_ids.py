"""Tests for node_id construction, parsing, and validation."""

from __future__ import annotations

import pytest

from cga.core.errors import ConfigError
from cga.data.ids import make_node_id, parse_node_id


def test_make_and_parse_roundtrip():
    node_id = make_node_id("cwe", "79")
    assert node_id == "cwe:79"
    assert parse_node_id(node_id) == ("cwe", "79")


def test_source_id_with_colons_roundtrips():
    node_id = make_node_id("attack", "attack-pattern--0a2a5b45-uuid:with:colons")
    graph_key, source_id = parse_node_id(node_id)
    assert graph_key == "attack"
    assert source_id == "attack-pattern--0a2a5b45-uuid:with:colons"


def test_make_node_id_rejects_colon_in_graph_key():
    with pytest.raises(ConfigError):
        make_node_id("bad:key", "123")


def test_make_node_id_rejects_empty_parts():
    with pytest.raises(ConfigError):
        make_node_id("", "123")
    with pytest.raises(ConfigError):
        make_node_id("cwe", "")


def test_parse_node_id_rejects_missing_separator():
    with pytest.raises(ConfigError):
        parse_node_id("no-colon-here")


def test_parse_node_id_rejects_empty_parts():
    with pytest.raises(ConfigError):
        parse_node_id(":source")
    with pytest.raises(ConfigError):
        parse_node_id("graph:")
