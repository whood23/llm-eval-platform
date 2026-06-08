"""Small shared result containers for the stats layer (no logic here)."""

from __future__ import annotations

from typing import NamedTuple


class CI(NamedTuple):
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
