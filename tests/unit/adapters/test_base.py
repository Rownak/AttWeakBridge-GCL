"""Tests for the adapter registry decorator and DatasetAdapter defaults."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from cga.adapters.base import DatasetAdapter, GraphSpec, register_adapter
from cga.core.errors import ConfigError
from cga.core.registry import adapters


class _DummyConfig(BaseModel):
    pass


def test_register_adapter_adds_to_registry():
    @register_adapter("dummy_dataset_for_test")
    class DummyAdapter(DatasetAdapter):
        ConfigModel = _DummyConfig
        graphs = (
            GraphSpec(key="a", role="graph_1", display_name="A"),
            GraphSpec(key="b", role="graph_2", display_name="B"),
        )

        def iter_nodes(self):
            return iter(())

        def iter_edges(self):
            return iter(())

        def iter_texts(self):
            return iter(())

    assert adapters.get("dummy_dataset_for_test") is DummyAdapter
    assert DummyAdapter.name == "dummy_dataset_for_test"


def test_register_adapter_duplicate_name_raises():
    class _A(DatasetAdapter):
        ConfigModel = _DummyConfig
        graphs = (
            GraphSpec(key="a", role="graph_1", display_name="A"),
            GraphSpec(key="b", role="graph_2", display_name="B"),
        )

        def iter_nodes(self):
            return iter(())

        def iter_edges(self):
            return iter(())

        def iter_texts(self):
            return iter(())

    register_adapter("dummy_dataset_dup")(_A)
    with pytest.raises(ConfigError):
        register_adapter("dummy_dataset_dup")(_A)


def test_default_iter_alignments_and_fingerprint_are_empty():
    class _A(DatasetAdapter):
        ConfigModel = _DummyConfig
        graphs = (
            GraphSpec(key="a", role="graph_1", display_name="A"),
            GraphSpec(key="b", role="graph_2", display_name="B"),
        )

        def iter_nodes(self):
            return iter(())

        def iter_edges(self):
            return iter(())

        def iter_texts(self):
            return iter(())

    instance = _A()
    assert list(instance.iter_alignments()) == []
    assert instance.source_fingerprint() == {}
    assert instance.validate_sources() is None
