# llm-eval-platform

**An LLM evaluation platform built as a *measurement instrument*.**

Most eval repos call a judge model, average the scores, and print a number. That number is
unfalsifiable: you can't tell whether it moved because your system improved or because the
judge is biased, miscalibrated, noisy, or measuring the wrong thing on the wrong data.

This project treats the eval the way an experimentalist treats a thermometer: **before you
trust the reading, you validate the instrument.** Every metric ships with error bars, the
judge is checked for position bias, agreement with humans, and calibration, and the evalset
itself is sampled and measured for coverage. The scores are downstream of proving the
scores mean something.

> **How to talk about it:** "It's an LLM eval platform built as a measurement instrument —
> the judge is a validated, bias-quantified, calibrated measuring device, and every number
> reported comes with a confidence interval."

---

## The thesis: validate the instrument, not just the system

| Most eval repos report... | This platform additionally proves... |
|---|---|
| a mean judge score | a **bootstrap confidence interval** on every metric |
| a pairwise win rate | the judge's **position-bias / flip rate**, and a bias-corrected win rate |
| "the LLM judge agrees with me" | **Cohen's κ / Krippendorff's α** vs. a human goldset |
| a calibrated-sounding score | an **ECE + reliability diagram** showing whether score ⇒ correctness |
| results on whatever data was handy | a **stratified, coverage-measured** evalset |
| an offline metric | its **predictive validity** (correlation, with CI) against a real-world proxy |

---

## Quickstart (offline, no API keys)

The platform runs end-to-end on a bare checkout with the deterministic **`dummy`** provider —
no keys, no network. Use it to exercise all the plumbing (provider → store → cache → retry →
rate-limit → run loop) before wiring up a real judge.

```bash
# 1. install (editable; pulls only the always-available deps)
pip install -e .

# 2. create the SQLite store + schema
eval-platform init-db

# 3. smoke-test the plumbing WITHOUT any judge rubric
#    sends a fixed diagnostic prompt through the provider/retry/ratelimit/cache stack
eval-platform smoke --n 5
#    -> prints how many judgments were written + total latency
#    -> a second run reports cache hits (cached=True)
```

`smoke` never touches the judge rubric or parser, so it works even though those are
hand-coded stubs (see below). It is the fastest way to confirm the spine is wired correctly.

### Running a real eval

Sample data ships in `data/`:

```bash
eval-platform run --mode pointwise \
  --items data/sample_eval.jsonl \
  --candidates data/sample_candidates.jsonl

eval-platform run --mode pairwise \
  --items data/sample_eval.jsonl \
  --candidates data/sample_candidates.jsonl
```

Until you hand-code the judge prompt/parser (and, for pairwise, the swap logic), `run` exits
cleanly with a message naming the exact file/method to implement and pointing you back at
`smoke` to test plumbing. That is by design — see the split below.

---

## Switching to a real judge (Gemini / Ollama / DeepSeek)

The real provider is [LiteLLM](https://github.com/BerriAI/litellm), kept **optional** so the
spine installs and runs on a bare environment. Install it and select a model via env vars
(prefix `EVAL_`). Copy `.env.example` to `.env` and edit, or export the variables directly.

```bash
pip install -e .[providers]      # installs litellm
```

**Gemini (the default judge — free tier):**
```bash
EVAL_PROVIDER=litellm
EVAL_JUDGE_MODEL=gemini/gemini-2.0-flash
GEMINI_API_KEY=...                # read by litellm directly, not by EVAL_*
```

**Ollama (fully local, no key):**
```bash
EVAL_PROVIDER=litellm
EVAL_JUDGE_MODEL=ollama/qwen3:8b
EVAL_API_BASE=http://localhost:11434
```

**DeepSeek:**
```bash
EVAL_PROVIDER=litellm
EVAL_JUDGE_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=...
```

Then re-run any `eval-platform run ...` command, or override per-invocation with
`--provider` / `--judge-model`.

---

## The phase map

The build follows a phased plan; each phase adds a layer of instrument validation. The
**plumbing for every phase is scaffolded**; the **measurement core of each is hand-coded**
(stubs in `stats/`, `judge/`, `datasets/sampling.py`, `trajectory/`).

| Phase | Question it answers | Scaffolded plumbing | You hand-code |
|---|---|---|---|
| **1** | Can I run a judge at all? | run loop, batching, retries, rate-limit, cache, persistence, providers | pointwise & pairwise **rubric prompts + parsers**, the pairwise **swap logic** |
| **2a** | Does my metric have error bars? | metric storage, reporting hooks | **bootstrap CI** (`stats/bootstrap.py`) |
| **2b** | Is the judge order-biased? | both-orders persistence (`position`) | **position flip rate + bias-corrected win rate** (`stats/position_bias.py`) |
| **2c** | Does the judge agree with humans? | `gold_labels` table, goldset template | **Cohen's κ + Krippendorff's α** (`stats/agreement.py`) |
| **2d** | Does a higher score mean more-correct? | — | **ECE + reliability curve** (`stats/calibration.py`) |
| **3** | Does my evalset cover the distribution? | dataset loader, versioning, dataset card | **stratified sampling** (`datasets/sampling.py`) + **diversity/coverage** (`stats/diversity.py`) |
| **4** | Does the offline metric predict reality? | — | **predictive validity** (`stats/predictive.py`) |
| **5** | Why did the agent fail, and where? | trace capture (`trajectory/capture.py`) | **trajectory scoring + failure localization** (`trajectory/score.py`) |
| **6** | How do I ship & gate on this? | static HTML report, plots, Streamlit dashboard, CI regression gate | (the metrics the gate compares come from your stats) |

Verify your hand-coded statistics against independent oracles at any time:

```bash
eval-platform verify-stats          # or: python handcode/verify_stats.py / make verify-stats
```

It prints `[PASS] / [TODO] / [FAIL] / [SKIP]` per metric, checked against `scipy` / `sklearn`
/ hand-built fixtures. See `handcode/README.md`.

---

## Scaffolded vs. you-hand-code

The dividing line is deliberate: **the plumbing that's the same in every eval repo is
scaffolded; the measurement science that makes this an *instrument* is hand-coded** (and is
the part worth being able to whiteboard).

| Scaffolded (built, works today) | You hand-code (stubs that raise `NotImplementedError`) |
|---|---|
| `providers/` — dummy + LiteLLM backends, `get_provider` | — |
| `store/` — SQLite store, schema, judgment cache | — |
| `judge/runner.py`, `cache.py`, `retry.py`, `ratelimit.py` — run loop, caching, backoff, RPM limiting | `judge/pointwise.py`, `judge/pairwise.py` — **rubric prompts, parsers, pairwise swap logic** |
| `datasets/loader.py`, `versioning.py` — JSONL I/O, dataset hash + card | `datasets/sampling.py` — **stratified sampling** |
| `trajectory/capture.py` — trace recorder + JSONL I/O | `trajectory/score.py` — **trajectory scoring / failure localization** |
| `report/` — static HTML report, matplotlib plots, Streamlit dashboard | — |
| `gate/regression_gate.py` — threshold/compare logic | (compares metrics your stats produce) |
| `cli.py` — the `eval-platform` command surface | — |
| `stats/_types.py` — shared `CI` container | **all of `stats/`** — `bootstrap_ci`, `position_flip_rate`, `consistent_winrate`, `cohen_kappa`, `krippendorff_alpha`, `expected_calibration_error`, `reliability_curve`, `embedding_diversity`, `predictive_validity` |

The scaffolding **calls** the stubs and degrades gracefully: the runner raises an actionable
`JudgeNotImplemented` naming the method to write, and the report/dashboard render a "not yet
implemented — hand-code in `stats/...`" placeholder instead of crashing. Nothing in the
scaffolding implements a statistic, a rubric, a parser, or the swap logic.

---

## CLI

The `eval-platform` command (entry point → `evalplatform.cli:app`, built with Typer):

| Command | What it does |
|---|---|
| `eval-platform init-db` | create the SQLite store and apply the schema |
| `eval-platform version` | print the platform version |
| `eval-platform smoke --n 5` | wire-check provider + store + cache + retry + rate-limit **without** the judge; second run reports cache hits |
| `eval-platform run --mode pointwise\|pairwise --items PATH --candidates PATH [--provider P] [--judge-model M]` | load data, run the judge through the full pipeline, persist judgments; prints what to hand-code if a judge stub is unimplemented |
| `eval-platform verify-stats [--strict]` | run `handcode/verify_stats.py`; `--strict` also fails while any stat is still a `[TODO]` |
| `eval-platform report --run-id RID [--out DIR]` | write a self-contained static HTML report for a run |
| `eval-platform gate --current PATH --baseline PATH [--thresholds PATH]` | CI regression gate; exits non-zero on a regression |
| `eval-platform dashboard` | launch the Streamlit dashboard (requires `.[dashboard]`) |

---

## Install profiles

```bash
pip install -e .                 # spine: dummy provider, store, run loop, stats stubs, static reports
pip install -e .[providers]      # + litellm (Gemini / DeepSeek / Ollama / OpenAI-compatible judges)
pip install -e .[reliability]    # + tenacity / diskcache (optional retry/cache backends)
pip install -e .[dashboard]      # + streamlit / plotly (interactive dashboard)
pip install -e .[dev]            # + pytest / pytest-cov
pip install -e .[all]            # everything above
```

Lazy-imported, optional deps (`litellm`, `tenacity`, `diskcache`, `streamlit`, `plotly`,
`pytest`) are never imported at module top level; if one is missing, the relevant command
prints a clear install hint and the rest of the platform keeps working.

---

## Layout

```
src/evalplatform/      the package (importable as `evalplatform`)
  models.py            the data contract (Pydantic)
  config.py            EVAL_* settings
  providers/           dummy + LiteLLM judge backends
  store/               SQLite store, schema.sql, judgment cache
  judge/               run loop, retry, rate-limit, cache  +  rubric/parser/swap STUBS
  stats/               measurement-science STUBS (you hand-code) + shared CI type
  datasets/            JSONL loader, versioning  +  stratified-sampling STUB
  trajectory/          trace capture  +  scoring STUB
  report/              static HTML report, plots, Streamlit dashboard
  gate/                CI regression gate
  cli.py               `eval-platform` entry point
handcode/              verify_stats.py (the oracle harness) + your-reps README
data/                  sample_eval.jsonl, sample_candidates.jsonl, goldset_template.jsonl
```

See **ARCHITECTURE.md** for the module map, the data contract, and the run conventions.
