# llm-eval-platform — system guide & usage examples

A practical walkthrough of how the platform works and how to use each important part, with
copy-pasteable examples. For the positioning/pitch see `README.md`; for the module map see
`ARCHITECTURE.md`; for your hand-code reps see `handcode/README.md`.

- [1. Mental model](#1-mental-model)
- [2. The data flow](#2-the-data-flow)
- [3. What was used (stack)](#3-what-was-used-stack)
- [4. Install & 60-second smoke test](#4-install--60-second-smoke-test)
- [5. Configuration (env vars)](#5-configuration-env-vars)
- [6. How models are called (providers)](#6-how-models-are-called-providers)
- [7. The data contract (items, candidates, judgments)](#7-the-data-contract)
- [8. Running an eval — CLI](#8-running-an-eval--cli)
- [9. Running an eval — Python API](#9-running-an-eval--python-api)
- [10. Implementing the judge (your reps)](#10-implementing-the-judge-your-reps)
- [11. Implementing & verifying the statistics (your reps)](#11-implementing--verifying-the-statistics-your-reps)
- [12. Querying results & the cache](#12-querying-results--the-cache)
- [13. Human gold labels (Phase 2c)](#13-human-gold-labels-phase-2c)
- [14. Reports & the CI gate](#14-reports--the-ci-gate)
- [15. What runs today vs. what's yours](#15-what-runs-today-vs-whats-yours)

---

## 1. Mental model

An LLM-as-judge eval platform built as a **measurement instrument**. The job isn't just to
produce a score — it's to *prove the score means something*: error bars, position-bias
quantification, agreement with human labels, calibration. The platform splits cleanly into:

- **Plumbing (built, works today):** providers, store + cache, the run loop, datasets,
  reporting, CLI.
- **Measurement science + the judge rubric (your hand-coded reps):** everything in `stats/`,
  the judge prompt/parser, the pairwise swap logic, stratified sampling, trajectory scoring.
  These are stubs that raise `NotImplementedError` until you implement them.

## 2. The data flow

```
data/*.jsonl  ──loader──►  EvalItem + Candidate (Pydantic)
                                      │
                                      ▼
                       judge/runner.py  (the run loop)
   per task, in a bounded thread pool:
     1. cache lookup ............... store.get_cached_*   (hit → copy through, no model call)
     2. judge.build_prompt() ....... YOUR rubric stub
     3. provider.complete() ........ the model call (dummy | litellm)   ← rate-limit + retry
     4. judge.parse_response() ..... YOUR parser stub    (ValueError → parse_ok=False)
     5. store.insert_* ............. raw + parsed + latency + position
                                      │
                                      ▼
              store/ (SQLite)  ──►  stats/ (YOUR metrics)  ──►  report/ + gate/
```

## 3. What was used (stack)

| Concern | Choice | Notes |
|---|---|---|
| Data contract | Pydantic v2 (`models.py`) | typed objects, JSON round-trip into SQLite |
| Config | pydantic-settings + `.env` | env prefix `EVAL_`; nothing hard-coded |
| Model transport | **LiteLLM** (optional) + built-in **dummy** | one API for Gemini/DeepSeek/Ollama/OpenAI-compatible |
| Storage / cache | stdlib `sqlite3` | the judgment tables *are* the cache |
| Concurrency | `ThreadPoolExecutor` | judge calls are I/O-bound |
| Retry / rate-limit | stdlib (backoff + interval) | no extra deps for the spine |
| CLI | Typer | the `eval-platform` command |
| Stat oracles | scipy / scikit-learn | reference checks for your metrics |
| Reports | matplotlib (HTML) + optional Streamlit | static report works with no extra deps |

The **spine uses only packages already in your env**. `litellm`, `streamlit`, `plotly`,
`tenacity`, `diskcache` are optional extras, imported lazily — a bare checkout still runs.

## 4. Install & 60-second smoke test

```bash
pip install -e .            # spine only (dummy provider, store, run loop, stubs, reports)
# optional extras:
#   pip install -e .[providers]   # litellm (real judges)
#   pip install -e .[dashboard]   # streamlit + plotly
#   pip install -e .[dev]         # pytest
#   pip install -e .[all]

eval-platform init-db       # create data/eval.db + schema
eval-platform smoke --n 5   # exercise provider→store→cache→retry→ratelimit WITHOUT a judge
```

Expected `smoke` output (the dummy provider needs no keys/network):

```
smoke pass 1: provider=dummy n=5 cached=0 total_latency_ms=0.1 wall_ms=4.2
smoke pass 2: n=5 cached=5
smoke OK -> run_id=run_9f16e06d
```

`pass 2` showing `cached=5` proves the judgment cache works.

## 5. Configuration (env vars)

All settings live in `config.py` (`Settings`) and load from environment / `.env` with the
`EVAL_` prefix. Copy the template and edit:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_PROVIDER` | `dummy` | `dummy` (offline) or `litellm` |
| `EVAL_JUDGE_MODEL` | `gemini/gemini-2.0-flash` | litellm model id (ignored by dummy) |
| `EVAL_API_BASE` | — | base URL for local/OpenAI-compatible endpoints |
| `EVAL_PROMPT_VERSION` | `v1` | bump when you change the rubric (also a cache-key component) |
| `EVAL_MAX_CONCURRENCY` | `4` | parallel in-flight judge calls |
| `EVAL_MAX_RETRIES` | `5` | retry attempts on transient errors |
| `EVAL_RPM_LIMIT` | — | requests/minute cap (None = unlimited) |
| `EVAL_CACHE_ENABLED` | `true` | skip re-judging unchanged items |
| `EVAL_DB_PATH` | `data/eval.db` | SQLite file |

```python
from evalplatform.config import get_settings
s = get_settings()
print(s.provider, s.judge_model, s.max_concurrency)
```

## 6. How models are called (providers)

A **provider** is anything with a single method (`providers/base.py`):

```python
def complete(self, prompt, *, system=None, temperature=0.0, timeout=None) -> ProviderResponse
# ProviderResponse(text: str, model: str, latency_ms: float, raw: dict | None)
```

`get_provider(settings)` returns the configured backend.

**Dummy (offline, deterministic):**

```python
from evalplatform.providers.dummy import DummyProvider

resp = DummyProvider().complete("Score this answer 1-5.", system="You are a strict judge.")
print(resp.model)        # 'dummy'
print(resp.text)         # deterministic diagnostic text (sha256 of system+prompt)
print(resp.latency_ms)   # measured with perf_counter
```

It returns the **same text for the same input** (no randomness) — that's what makes caching
and idempotency testable. It is *not* a rubric verdict; it only exercises plumbing.

**LiteLLM (real judge):** lazily imports `litellm` and calls its unified API:

```python
# internally, for EVAL_PROVIDER=litellm:
litellm.completion(
    model=settings.judge_model,        # e.g. "gemini/gemini-2.0-flash"
    messages=[{"role": "system", ...}, {"role": "user", "content": prompt}],
    api_base=settings.api_base,
    temperature=settings.temperature,
    timeout=settings.request_timeout,
)
# -> response.choices[0].message.content
```

Pick a model with env vars (API keys are read by litellm itself, not by `EVAL_*`):

```bash
# Gemini (free tier — the guide's default judge)
EVAL_PROVIDER=litellm
EVAL_JUDGE_MODEL=gemini/gemini-2.0-flash
GEMINI_API_KEY=...

# Ollama (fully local, no key)
EVAL_PROVIDER=litellm
EVAL_JUDGE_MODEL=ollama/qwen3:8b
EVAL_API_BASE=http://localhost:11434

# DeepSeek (cheap, OpenAI-compatible)
EVAL_PROVIDER=litellm
EVAL_JUDGE_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=...
```

Every call the **runner** makes is wrapped in the reliability primitives:
`RateLimiter.acquire()` (if `EVAL_RPM_LIMIT` is set) → `retry_call(..., max_retries=...)`
(exponential backoff with jitter) → result.

## 7. The data contract

Defined in `models.py` (Pydantic). The store schema (`store/schema.sql`) mirrors it.

**`EvalItem`** — a task/question (independent of any answer):

```python
from evalplatform.models import EvalItem
EvalItem(id="q1", input="What is the capital of Australia?",
         reference="Canberra", stratum="factual", tags=["geo"])
```

**`Candidate`** — a system's answer to an item (the thing judged):

```python
from evalplatform.models import Candidate
Candidate(id="q1::alpha", item_id="q1", system="alpha-v1", output="Sydney")
```

**Judgments** (produced by the runner, persisted to SQLite):

- `PointwiseJudgment` — `score`, `label`, `rationale`, `raw_response`, `parse_ok`, `cached`,
  `latency_ms`, `judge_model`, `prompt_version`, `created_at`.
- `PairwiseJudgment` — adds `position` (`AB`/`BA`), `winner_slot` (`A`/`B`/`tie`), and
  `winner_candidate_id`. Two rows per logical comparison (both orders) — this is what makes
  position-bias measurable.

Dataset files are JSONL, one object per line (see `data/sample_eval.jsonl`):

```json
{"id": "q1", "input": "What is the capital of Australia?", "reference": "Canberra", "stratum": "factual", "tags": ["geo"], "metadata": {"difficulty": "easy"}}
```
```json
{"id": "q1::alpha", "item_id": "q1", "system": "alpha-v1", "output": "Sydney"}
{"id": "q1::beta",  "item_id": "q1", "system": "beta-v1",  "output": "Canberra"}
```

## 8. Running an eval — CLI

```bash
eval-platform run --mode pointwise \
  --items data/sample_eval.jsonl \
  --candidates data/sample_candidates.jsonl

eval-platform run --mode pairwise \
  --items data/sample_eval.jsonl \
  --candidates data/sample_candidates.jsonl

# override provider/model per invocation:
eval-platform run --mode pointwise --provider litellm --judge-model gemini/gemini-2.0-flash \
  --items data/sample_eval.jsonl --candidates data/sample_candidates.jsonl
```

Until you implement the judge rubric/parser, `run` exits cleanly (code 1) telling you exactly
what to write:

```
judge is not yet implemented -- nothing was judged.
  src/evalplatform/judge/pointwise.py:build_prompt is not implemented yet (Phase 1 hand-code).
  Implement it, or run 'eval-platform smoke' to test the plumbing without a judge.
```

Full command surface:

| Command | What it does |
|---|---|
| `eval-platform init-db` | create the SQLite store + schema |
| `eval-platform smoke --n 5` | wire-check provider+store+cache+retry+ratelimit (no judge) |
| `eval-platform run --mode pointwise\|pairwise --items P --candidates P` | run the judge over a dataset |
| `eval-platform verify-stats [--strict]` | check your hand-coded stats vs oracles |
| `eval-platform report --run-id RID [--out DIR]` | write a static HTML report |
| `eval-platform gate --current P --baseline P [--thresholds P]` | CI regression gate (exit ≠0 on regression) |
| `eval-platform dashboard` | launch the Streamlit dashboard (needs `.[dashboard]`) |

## 9. Running an eval — Python API

The same pipeline, wired by hand — useful for notebooks and tests:

```python
from evalplatform.config import get_settings
from evalplatform.store.db import Store
from evalplatform.providers.base import get_provider
from evalplatform.judge.cache import JudgmentCache
from evalplatform.judge.ratelimit import RateLimiter
from evalplatform.judge.runner import Runner, JudgeNotImplemented
from evalplatform.judge.pointwise import PointwiseJudge
from evalplatform.datasets.loader import load_items, load_candidates, candidates_by_item

settings  = get_settings()
store     = Store.open(settings)                         # connects + applies schema
provider  = get_provider(settings)                       # dummy by default
cache     = JudgmentCache(store, enabled=settings.cache_enabled)
limiter   = RateLimiter(settings.rpm_limit)              # no-op when rpm_limit is None
runner    = Runner(settings=settings, store=store, provider=provider,
                   cache=cache, rate_limiter=limiter)

items = load_items("data/sample_eval.jsonl")
cands = candidates_by_item(load_candidates("data/sample_candidates.jsonl"))
judge = PointwiseJudge(prompt_version=settings.prompt_version)

try:
    run = runner.run_pointwise(items, cands, judge)
    rows = store.pointwise_for_run(run.run_id)
    print(f"run {run.run_id}: {len(rows)} judgments, "
          f"{sum(r.parse_ok for r in rows)} parsed")
except JudgeNotImplemented as e:
    print("implement the judge first:", e)   # build_prompt is still a stub
```

## 10. Implementing the judge (your reps)

`judge/pointwise.py` and `judge/pairwise.py` are stubs. You write the rubric and the parser;
the runner calls them. Recommended: have the judge return JSON so the parser is robust.

```python
# src/evalplatform/judge/pointwise.py
class PointwiseJudge:
    def build_prompt(self, item, candidate) -> str:
        # YOUR rubric. Use item.input / item.reference / item.context.
        # Tip: ask for parseable output, e.g. {"score": 1-5, "rationale": "..."}.
        ...

    def parse_response(self, raw) -> ParsedPointwise:
        # Pull score/label/rationale out of `raw`. Raise ValueError if unparseable
        # (the runner records parse_ok=False and keeps the raw response).
        ...
```

For `pairwise.py` you also implement `iter_presentations(item, candidates)` — **the swap
logic**: yield each pair *twice*, once `Position.AB` and once `Position.BA`. Refer to the two
answers only as "A" and "B" in the prompt so the only thing that differs between the two
presentations is the order — that's what lets Phase-2b measure position bias.

Once implemented, `eval-platform run ...` produces real judgments end-to-end.

## 11. Implementing & verifying the statistics (your reps)

Each function in `stats/` is a stub whose docstring states the contract, the reference oracle,
and a hint. Workflow:

```bash
# 1. open the stub, read the docstring contract
#    src/evalplatform/stats/bootstrap.py  ->  def bootstrap_ci(data, statistic=np.mean, ...)-> CI
# 2. implement the body (the recipe is in the docstring; don't peek at a solution first)
# 3. verify against the oracle:
eval-platform verify-stats        # or: python handcode/verify_stats.py
```

`verify-stats` prints one line per metric, checked against scipy/sklearn/hand-built fixtures:

```
[TODO]  2a  bootstrap CI                 bootstrap_ci      -> stub not implemented yet
[PASS]  2a  bootstrap CI                 bootstrap_ci      -> point=4.998 CI=[4.81,5.18]
[FAIL]  2c  Cohen's kappa                cohen_kappa       -> kappa=0.41 != sklearn 0.39
...
0 fail -> exit 0   (use --strict to also fail while any [TODO] remains)
```

Once a metric is green, *use* it (this is plumbing, not a rep):

```python
from evalplatform.stats import bootstrap_ci
ci = bootstrap_ci([0.82, 0.79, 0.91, 0.88, 0.74])
print(f"mean={ci.point:.3f}  95% CI=[{ci.low:.3f}, {ci.high:.3f}]")
```

The full set you'll implement: `bootstrap_ci`, `position_flip_rate`, `consistent_winrate`,
`cohen_kappa`, `krippendorff_alpha`, `expected_calibration_error`, `reliability_curve`,
`embedding_diversity`, `predictive_validity` (+ `datasets/sampling.stratified_sample`,
`trajectory/score.score_trajectory`).

## 12. Querying results & the cache

```python
from evalplatform.config import get_settings
from evalplatform.store.db import Store

store = Store.open(get_settings())
run = store.latest_run()                       # most recent run (or pass mode="pointwise")
rows = store.pointwise_for_run(run.run_id)
for r in rows[:3]:
    print(r.candidate_id, "score=", r.score, "parsed=", r.parse_ok,
          "cached=", r.cached, "raw=", (r.raw_response or "")[:50])
```

**Caching:** before any model call the runner checks the store. The cache key is
`(candidate_id, judge_model, prompt_version)` for pointwise and
`(candidate_a, candidate_b, position, judge_model, prompt_version)` for pairwise. A hit is
copied into the new run with `cached=True` and **no provider call**. Consequences:

- Re-running the same eval is nearly free (only new/changed items hit the model).
- Bumping `EVAL_PROMPT_VERSION` (i.e. changing your rubric) invalidates the cache — correct,
  because a different prompt is a different measurement.

## 13. Human gold labels (Phase 2c)

To measure judge↔human agreement you label a goldset yourself. The store has a table + API
for it (`data/goldset_template.jsonl` shows the format):

```python
store.insert_gold_label(item_id="q1", labeler="me", label="correct", score=1.0)
labels = store.gold_labels(item_id="q1")
```

Your hand-coded `stats.cohen_kappa` / `stats.krippendorff_alpha` then compare these human
labels against the judge's labels.

## 14. Reports & the CI gate

```bash
# static HTML report for a run (renders counts, parse-ok rate, sample judgments, and —
# once your stats are implemented — CIs / calibration / bias; otherwise placeholders)
eval-platform report --run-id <run_id>
#  -> reports/<run_id>/report.html

# regression gate for CI: fail the build if a metric drops beyond threshold
echo '{"accuracy": 0.81, "kappa": 0.62}' > current.json
echo '{"accuracy": 0.80, "kappa": 0.60}' > baseline.json
eval-platform gate --current current.json --baseline baseline.json
#  GATE PASS - 2 metric(s) compared, no regressions.   (exit 0)
#  GATE FAIL - 1 regression(s): accuracy dropped ...    (exit 1)
```

The report tolerates unimplemented stats (it wraps each in `try/except NotImplementedError`
and shows a "hand-code in stats/..." placeholder), so it works at every stage.

## 15. What runs today vs. what's yours

| Runs today (scaffolded) | Your reps (stubs) |
|---|---|
| providers (dummy + LiteLLM), `get_provider` | judge rubric prompts + parsers (`judge/pointwise.py`, `judge/pairwise.py`) |
| SQLite store + schema + cache | pairwise **swap logic** (`iter_presentations`) |
| run loop: concurrency, retry, rate-limit, persistence | all of `stats/` (bootstrap, bias, kappa/alpha, ECE, diversity, predictive) |
| dataset loader, versioning, dataset cards | stratified sampling (`datasets/sampling.py`) |
| trajectory capture | trajectory scoring (`trajectory/score.py`) |
| reports, plots, dashboard, CI gate, CLI | — |
| `handcode/verify_stats.py` (the oracle harness) | (you make it go green) |

**Suggested first move** (per the build guide): implement `stats/bootstrap.py`, run
`eval-platform verify-stats`, and watch `2a bootstrap CI` flip to `[PASS]`.
