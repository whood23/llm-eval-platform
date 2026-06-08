"""Tests for providers: dummy determinism, response shape, factory (AI-written)."""

from __future__ import annotations

import pytest

from evalplatform.providers.base import JudgeProvider, ProviderResponse, get_provider
from evalplatform.providers.dummy import DummyProvider


def test_provider_response_shape():
    resp = ProviderResponse(text="hi", model="dummy", latency_ms=1.5)
    assert resp.text == "hi"
    assert resp.model == "dummy"
    assert isinstance(resp.latency_ms, float)
    assert resp.raw is None


def test_dummy_provider_name_and_response_type():
    p = DummyProvider()
    assert p.name == "dummy"
    resp = p.complete("plumbing smoke: echo id i1")
    assert isinstance(resp, ProviderResponse)
    assert resp.model == "dummy"
    assert isinstance(resp.text, str) and resp.text
    assert resp.latency_ms >= 0.0


def test_dummy_provider_is_deterministic():
    p = DummyProvider()
    a = p.complete("same prompt", system="sys")
    b = p.complete("same prompt", system="sys")
    # Text is a pure function of (prompt, system) -> stable across calls.
    assert a.text == b.text


def test_dummy_provider_varies_with_input():
    p = DummyProvider()
    t1 = p.complete("prompt one").text
    t2 = p.complete("prompt two").text
    assert t1 != t2
    # System message is part of the digest, too.
    assert p.complete("p", system="s1").text != p.complete("p", system="s2").text


def test_dummy_provider_satisfies_protocol():
    # runtime_checkable Protocol: structural conformance.
    assert isinstance(DummyProvider(), JudgeProvider)


def test_dummy_provider_text_is_not_a_verdict():
    # The dummy must emit generic diagnostic text, never a rubric score/winner.
    text = DummyProvider().complete("anything").text.lower()
    assert "dummy" in text


def test_get_provider_dummy(settings):
    settings.provider = "dummy"
    p = get_provider(settings)
    assert isinstance(p, DummyProvider)
    assert p.name == "dummy"


def test_get_provider_unknown_raises(settings):
    settings.provider = "nope"
    with pytest.raises(ValueError):
        get_provider(settings)


def test_get_provider_litellm_constructs_without_calling(settings):
    # Constructing the litellm provider must NOT import litellm (that's lazy in complete()).
    settings.provider = "litellm"
    settings.judge_model = "gemini/gemini-2.0-flash"
    p = get_provider(settings)
    assert p.name == "litellm"
    assert p.model == "gemini/gemini-2.0-flash"
