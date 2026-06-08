"""Offline deterministic provider (scaffolding, AI-written).

``DummyProvider`` lets the whole platform run with zero external deps or API keys. Its
output is a *generic diagnostic string* derived from a hash of the prompt+system, NOT a
rubric verdict — it exists only to exercise plumbing (store/cache/retry/ratelimit/runner).
The hash makes it stable across runs so caching and idempotency are testable.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from .base import ProviderResponse


class DummyProvider:
    """Deterministic, dependency-free provider for offline plumbing checks."""

    name = "dummy"

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
    ) -> ProviderResponse:
        start = time.perf_counter()

        # Stable digest of the full input; no randomness so identical inputs map to
        # identical output (required for cache-hit / idempotency testing).
        payload = f"{system or ''}\x00{prompt}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()

        # Generic diagnostic text — explicitly not a verdict the judge parser would trust.
        text = (
            f"[dummy provider] deterministic diagnostic response.\n"
            f"digest={digest[:16]} prompt_len={len(prompt)} temperature={temperature}"
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return ProviderResponse(
            text=text,
            model=self.name,
            latency_ms=latency_ms,
            raw={"digest": digest, "prompt_len": len(prompt)},
        )
