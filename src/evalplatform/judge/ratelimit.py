"""Thread-safe requests-per-minute limiter (scaffolding, AI). Stdlib only.

A minimal token-spacing limiter: it enforces a *minimum interval* between successive
``acquire()`` calls so the aggregate request rate stays at or below ``rpm`` requests per
minute, even when many worker threads (see ``judge/runner.py``) call it concurrently. Pass
``rpm=None`` to disable limiting entirely (``acquire`` becomes a no-op).
"""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Optional, Type


class RateLimiter:
    """Spaces calls so the request rate does not exceed ``rpm`` per minute.

    Thread-safe via a ``threading.Lock`` guarding the timestamp of the last grant. Usable as a
    context manager (``with limiter: ...``) which simply calls :meth:`acquire` on entry.
    """

    def __init__(self, rpm: Optional[int]) -> None:
        if rpm is not None and rpm <= 0:
            raise ValueError("rpm must be a positive integer or None")
        self.rpm = rpm
        # Minimum seconds between consecutive grants (0 when unlimited).
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        # perf_counter timestamp of the last grant; None until the first acquire.
        self._last: Optional[float] = None

    def acquire(self) -> None:
        """Block until the next request is allowed under the rate limit.

        No-op when ``rpm`` is ``None``. Otherwise sleeps just enough so that successive
        grants are at least ``60 / rpm`` seconds apart. The sleep happens outside the lock so
        a waiting thread does not stall others.
        """
        if not self._min_interval:
            return
        while True:
            with self._lock:
                now = time.perf_counter()
                if self._last is None:
                    self._last = now
                    return
                earliest = self._last + self._min_interval
                if now >= earliest:
                    self._last = now
                    return
                wait = earliest - now
            # Sleep outside the lock, then re-check (another thread may have advanced ``_last``).
            time.sleep(wait)

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        return None
