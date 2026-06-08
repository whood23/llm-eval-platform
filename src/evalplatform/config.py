"""Runtime configuration.

Settings load from environment variables (prefix ``EVAL_``) and an optional ``.env`` file,
so nothing is hard-coded. The defaults are chosen so the platform runs end-to-end on a bare
checkout with no API keys: ``provider='dummy'`` uses the offline deterministic judge.

Switch to a real judge by setting, e.g.::

    EVAL_PROVIDER=litellm
    EVAL_JUDGE_MODEL=gemini/gemini-2.0-flash      # free tier; the guide's default judge
    GEMINI_API_KEY=...                            # read by litellm itself

or point at a local / OpenAI-compatible endpoint::

    EVAL_PROVIDER=litellm
    EVAL_JUDGE_MODEL=ollama/qwen3:8b
    EVAL_API_BASE=http://localhost:11434
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = .../llm-eval-platform  (this file is src/evalplatform/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- provider / judge -------------------------------------------------------------
    provider: str = "dummy"  # "dummy" (offline) | "litellm" (Gemini/DeepSeek/Ollama/...)
    judge_model: str = "gemini/gemini-2.0-flash"  # litellm model id; ignored by dummy
    prompt_version: str = "v1"  # bump when the judge prompt/rubric changes
    api_base: Optional[str] = None  # base URL for local / OpenAI-compatible endpoints
    temperature: float = 0.0  # judges should be near-deterministic

    # --- concurrency / reliability ----------------------------------------------------
    max_concurrency: int = 4  # parallel in-flight judge calls (batching)
    max_retries: int = 5  # retry attempts on transient provider errors
    rpm_limit: Optional[int] = None  # requests-per-minute cap; None = unlimited
    request_timeout: float = 60.0  # per-call timeout, seconds

    # --- caching / storage ------------------------------------------------------------
    cache_enabled: bool = True  # skip re-judging unchanged items
    db_path: Path = REPO_ROOT / "data" / "eval.db"
    data_dir: Path = REPO_ROOT / "data"
    reports_dir: Path = REPO_ROOT / "reports"

    # --- stats defaults (consumed by the hand-coded stats layer) ----------------------
    bootstrap_resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
