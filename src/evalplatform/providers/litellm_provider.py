"""litellm-backed provider (scaffolding, AI-written).

Routes prompts to any model litellm supports (Gemini, DeepSeek, OpenAI-compatible,
Ollama, ...). ``litellm`` is an *optional* dependency: it is imported lazily inside
``complete`` so a bare checkout that only uses the dummy provider never needs it.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import ProviderResponse


class LiteLLMProvider:
    """Real judge transport built on litellm's unified ``completion`` API."""

    name = "litellm"

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.temperature = temperature

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
    ) -> ProviderResponse:
        # Lazy import: litellm is optional (see pyproject extra '[providers]').
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError("pip install .[providers] for litellm") from exc

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_base=self.api_base,
            temperature=temperature,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        # litellm mirrors the OpenAI response shape: choices[0].message.content.
        text = response.choices[0].message.content or ""

        # Best-effort serialization of the raw response for auditing.
        raw: Optional[dict]
        try:
            raw = response.model_dump()  # litellm responses expose pydantic .model_dump
        except Exception:  # pragma: no cover - defensive, shape varies by version
            try:
                raw = dict(response)
            except Exception:
                raw = None

        return ProviderResponse(
            text=text,
            model=self.model,
            latency_ms=latency_ms,
            raw=raw,
        )
