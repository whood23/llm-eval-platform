"""llm-eval-platform v2 — an LLM evaluation platform built as a *measurement instrument*.

The thesis: an eval platform's job is to validate the instrument, not just emit scores.
Most eval repos call a judge and report a mean. This one proves the judge is reliable,
unbiased, calibrated, and predictive — and every number ships with error bars.

Layering (see ARCHITECTURE.md):
  - ``models``     : the data contract (eval items, candidates, judgments, run metadata).
  - ``config``     : runtime settings (provider, judge model, paths, concurrency).
  - ``providers``  : pluggable LLM backends via LiteLLM (Gemini/DeepSeek/Ollama) + an
                     offline ``dummy`` provider so the spine runs with no API keys.
  - ``store``      : SQLite persistence + judgment cache (raw judgment, model, prompt
                     version, candidate position, timestamp — everything Phase 2 needs).
  - ``judge``      : the run loop + reliability plumbing (retries, rate-limit, batching,
                     caching).  *The prompt/rubric, output parsing, and pairwise-swap
                     logic are hand-coded by the user (left as stubs).*
  - ``stats``      : the measurement-science layer (bootstrap CIs, position bias,
                     inter-rater agreement, calibration, predictive validity).
                     *Hand-coded by the user — these are stubs; verify with
                     ``handcode/verify_stats.py``.*
  - ``report``     : dashboards, plots, and the CI regression gate.

What the user hand-codes vs. what is scaffolded is documented per-module and in CLAUDE.md.
"""

__version__ = "2.0.0.dev0"

__all__ = ["__version__"]
