"""Tests for the ``protocol`` guard helpers."""

from __future__ import annotations

import pytest

from cga.core.errors import ConfigError
from cga.core.protocol import get_protocol, is_unsupervised, refuse


def test_protocol_defaults_to_supervised():
    assert get_protocol({}) == "supervised"
    assert not is_unsupervised({})


def test_unsupervised_protocol_is_detected():
    assert is_unsupervised({"protocol": "unsupervised"})


def test_unknown_protocol_is_rejected():
    with pytest.raises(ConfigError, match="semi"):
        get_protocol({"protocol": "semi"})


def test_refuse_names_pipeline_and_option():
    with pytest.raises(ConfigError, match=r"gcl_train refuses loss.name=infonce_full .*mask"):
        refuse("gcl_train", "loss.name=infonce_full", "its softmax mask reads gold")
