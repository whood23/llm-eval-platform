"""Provider abstraction for the judge backend (scaffolding, AI-written).

A *provider* is the thin transport that turns a prompt into text. Statistics and the
judge rubric/parsing live elsewhere; a provider only knows how to call a model (or, for
the dummy, fabricate deterministic text) and report latency. This keeps the run loop
decoupled from any specific SDK and lets the platform run offline with ``DummyProvider``.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel


class ProviderResponse(BaseModel):
    """The normalized result of a single provider call."""

    text: str
    model: str
    latency_ms: float
    raw: Optional[dict] = None


@runtime_checkable
class JudgeProvider(Protocol):
    """Structural type every provider satisfies: a name plus a ``complete`` call."""

    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
    ) -> ProviderResponse:
        """Return model text for ``prompt`` (optionally steered by a ``system`` message)."""
        ...


def get_provider(settings) -> JudgeProvider:
    """Construct the provider named by ``settings.provider``.

    ``'dummy'`` -> offline deterministic provider; ``'litellm'`` -> real backend wired to
    ``settings.judge_model`` / ``settings.api_base`` / ``settings.temperature``.
    """
    # Local imports avoid a circular import (providers/__init__ pulls these modules in).
    from .dummy import DummyProvider
    from .litellm_provider import LiteLLMProvider

    provider = settings.provider
    if provider == "dummy":
        return DummyProvider()
    if provider == "litellm":
        return LiteLLMProvider(
            model=settings.judge_model,
            api_base=settings.api_base,
            temperature=settings.temperature,
        )
    raise ValueError(f"unknown provider: {provider!r} (expected 'dummy' or 'litellm')")
