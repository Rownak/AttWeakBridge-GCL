"""Tests for the generic name->object Registry."""

from __future__ import annotations

import pytest

from cga.core.errors import ConfigError
from cga.core.registry import Registry


def test_register_and_get_roundtrip():
    reg: Registry[type] = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo


def test_duplicate_name_raises_config_error():
    reg: Registry[type] = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(ConfigError):

        @reg.register("foo")
        class Bar:
            pass


def test_get_unknown_name_raises_config_error_listing_known():
    reg: Registry[type] = Registry("widget")
    reg.register("foo")(object)

    with pytest.raises(ConfigError, match="foo"):
        reg.get("bar")


def test_names_returns_sorted_registered_keys():
    reg: Registry[type] = Registry("widget")
    reg.register("zeta")(object)
    reg.register("alpha")(object)

    assert reg.names() == ["alpha", "zeta"]
