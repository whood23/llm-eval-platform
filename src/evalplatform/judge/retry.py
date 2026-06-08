"""Retry with exponential backoff + jitter (scaffolding, AI). Stdlib only.

Wraps a flaky provider call so transient errors (rate limits, timeouts, 5xx) are retried
with exponentially growing, jittered delays. No third-party deps: the contract forbids a
top-level ``tenacity`` import, so this is a small hand-rolled backoff using ``time.sleep``
and a *locally seeded* ``random.Random`` for the jitter (deterministic per attempt, never
the global RNG).
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """Call ``fn()`` with retries; reraise the last exception once retries are exhausted.

    The delay before retry ``attempt`` (1-based) is ``min(max_delay, base_delay * 2**(attempt-1))``
    multiplied by a jitter factor in ``[0.5, 1.0]`` drawn from a ``random.Random`` seeded from
    the attempt index, so backoff is reproducible and independent of global randomness.

    Parameters
    ----------
    fn:
        Zero-arg callable to (re)invoke. Wrap arguments with a lambda/partial at the call site.
    max_retries:
        Number of *retries* after the initial attempt (so ``fn`` runs up to ``max_retries + 1``
        times). ``0`` means try exactly once.
    exceptions:
        Only these exception types trigger a retry; anything else propagates immediately.
    on_retry:
        Optional hook called as ``on_retry(attempt, exc, sleep_seconds)`` right before sleeping,
        where ``attempt`` is 1-based for the upcoming retry.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except exceptions as exc:  # noqa: B902 - exceptions tuple is caller-supplied
            attempt += 1
            if attempt > max_retries:
                # Out of retries: reraise the most recent failure.
                raise
            # Exponential backoff, capped, with deterministic per-attempt jitter.
            backoff = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.Random(attempt).uniform(0.5, 1.0)
            sleep_s = backoff * jitter
            if on_retry is not None:
                on_retry(attempt, exc, sleep_s)
            time.sleep(sleep_s)
