# Architecture

How the platform is put together: the module map, the data contract every layer shares, the
scaffold / hand-code split, and the import & run conventions.

The guiding idea (see `README.md`): the eval is a **measurement instrument**. The plumbing
that runs a judge and stores its output is *scaffolding*; the statistics, rubrics, parsers,
sampler, and trajectory scorer that make the measurement trustworthy are *hand-coded* and
left as `NotImplementedError` stubs. The scaffolding is written so it composes with those
stubs and degrades gracefully when they are not yet implemented.

---

## Module map

```
src/evalplatform/
├── models.py            Data contract: Pydantic models shared by every layer.
├── config.py            Settings (pydantic-settings, env prefix EVAL_) + get_settings().
│
├── providers/           Pluggable LLM judge backends.
│   ├── base.py          ProviderResponse, JudgeProvider Protocol, get_provider(settings).
│   ├── dummy.py         DummyProvider — deterministic, offline, hash-derived text (no deps).
│   └── litellm_provider.py  LiteLLMProvider — Gemini/DeepSeek/Ollama/... (lazy `import litellm`).
│
├── store/               Persistence + the judgment cache.
│   ├── schema.sql       SQLite schema (runs, items, candidates, *_judgments, gold_labels).
│   └── db.py            Store: connect/init, upserts, judgment writes, cache lookups.
│
├── judge/               The run loop and its reliability plumbing  +  the judge STUBS.
│   ├── runner.py        Runner: batching/concurrency/cache/retry/ratelimit/persistence.
│   ├── cache.py         JudgmentCache: delegates to Store cache lookups (no-op when disabled).
│   ├── retry.py         retry_call: exponential backoff + deterministic jitter (stdlib only).
│   ├── ratelimit.py     RateLimiter: thread-safe RPM cap (min-interval).
│   ├── pointwise.py     [STUB] PointwiseJudge.build_prompt / parse_response.
│   └── pairwise.py      [STUB] PairwiseJudge.iter_presentations (swap), build_prompt, parse_response.
│
├── stats/               The measurement-science layer — ALL hand-coded.
│   ├── _types.py        CI(point, low, high) — shared result container (no logic).
│   ├── bootstrap.py     [STUB] bootstrap_ci                      (Phase 2a)
│   ├── position_bias.py [STUB] position_flip_rate, consistent_winrate (Phase 2b)
│   ├── agreement.py     [STUB] cohen_kappa, krippendorff_alpha   (Phase 2c)
│   ├── calibration.py   [STUB] expected_calibration_error, reliability_curve, ReliabilityCurve (Phase 2d)
│   ├── diversity.py     [STUB] embedding_diversity               (Phase 3)
│   └── predictive.py    [STUB] predictive_validity               (Phase 4)
│
├── datasets/            Dataset I/O & provenance  +  the sampling STUB.
│   ├── loader.py        load/write JSONL, items, candidates; candidates_by_item.
│   ├── versioning.py    dataset_version (content hash), write_dataset_card.
│   └── sampling.py      [STUB] stratified_sample                 (Phase 3)
│
├── trajectory/          Agent-trace capture  +  the scoring STUB.
│   ├── capture.py       TraceRecorder, trace_to_jsonl, load_traces.
│   └── score.py         [STUB] score_trajectory                  (Phase 5)
│
├── report/              Output surfaces (Phase 6).
│   ├── plots.py         matplotlib renderers (reliability_diagram, ci_bar, bias_bar, savefig).
│   ├── report.py        build_report — self-contained static HTML for a run.
│   └── dashboard.py     Streamlit app (lazy import; install hint if missing).
│
├── gate/                CI regression gate (Phase 6).
│   └── regression_gate.py  GateResult, gate(), load/save_baseline, run_gate_cli.
│
└── cli.py               Typer app; `eval-platform` entry point.

handcode/
├── verify_stats.py      The oracle harness: checks each stat vs scipy/sklearn/fixtures.
└── README.md            "your reps" — the working rule and the per-stat verify table.

data/                    Sample inputs: sample_eval.jsonl, sample_candidates.jsonl,
                         goldset_template.jsonl. Runtime DB is written to data/eval.db.
```

`[STUB]` = raises `NotImplementedError`; the user implements it. Everything else is built.

---

## The data contract (`models.py`)

A single set of Pydantic models is the source of truth shared by the store, run loop, judge,
stats, and report layers. The SQLite schema in `store/schema.sql` mirrors them exactly; the
`Store` converts rows ↔ models (JSON-encoding `list`/`dict` columns).

| Model | Role |
|---|---|
| `EvalItem` | one task/question: `id`, `input`, optional `reference` / `context` / `stratum`, `tags`, `metadata`. |
| `Candidate` | a system's response to an item: `id`, `item_id`, `system`, `output`, `metadata`. |
| `PointwiseJudgment` | a judge's rubric score for one candidate: `score`/`label`/`rationale`, plus the audit fields below. |
| `PairwiseJudgment` | a judge's A-vs-B verdict for **one presentation**: `candidate_a_id`/`candidate_b_id` (slots as presented), `position`, `winner_slot`, resolved `winner_candidate_id`. |
| `ParsedPointwise` / `ParsedPairwise` | what the hand-coded `parse_response` returns; the runner copies these onto the judgment. |
| `RunMeta` | provenance for a run: `run_id`, `mode`, `judge_model`, `prompt_version`, dataset id/version, `n_items`, config snapshot, timestamps. |
| `TraceStep` / `Trace` | a captured agent run (ordered steps + final output), for Phase 5. |
| `JudgeMode` (`pointwise`/`pairwise`), `Position` (`AB`/`BA`/`NA`) | enums. |
| `new_id(prefix='')` | short, optionally namespaced unique id. |

**Why the audit fields exist.** Phase 2 validates the judge *as an instrument*, and that is
only possible if, on every judgment, you persist *now*: the **raw judge response**, the
**judge model**, the **prompt version**, the **candidate position/order**, the **parse_ok**
flag, the **cached** flag, latency, and a timestamp. These are first-class columns even
though the statistics that consume them are hand-coded later.

**The pairwise two-row invariant.** A single logical comparison of candidates X and Y
produces **two** `PairwiseJudgment` rows — one presented `AB`, one `BA`. The hand-coded
`PairwiseJudge.iter_presentations` (the *swap logic*) emits both; the runner persists each
with its `position`; the Phase-2b stats group the two rows by
`(item_id, frozenset{candidate_a_id, candidate_b_id})` and compare `winner_candidate_id` to
measure flips. Position is therefore designed *into* the data model, not bolted on.

### The cache key

The judgment tables double as the cache. Two unique indexes define a cache hit:

- `idx_pw_cache` on `(candidate_id, judge_model, prompt_version)`.
- `idx_pr_cache` on `(candidate_a_id, candidate_b_id, position, judge_model, prompt_version)`.

A cache hit is simply "a row already exists for this key". Writes use `INSERT OR REPLACE`, so
re-runs are idempotent on the cache key. Bumping `EVAL_PROMPT_VERSION` invalidates the cache
by changing the key — which is exactly why the judge prompt carries a version.

---

## The scaffold / hand-code split

The contract the scaffolding holds to (an adversarial reviewer checks it):

- **Scaffolding never implements a core.** No statistics bodies, no rubric prompt text, no
  output parser, no pairwise swap logic, no stratified sampling, no trajectory scoring —
  anywhere, including tests. When a scaffolded module needs one of these, it **calls the
  stub's public function** and lets `NotImplementedError` propagate, handled gracefully.
- **Graceful degradation, two patterns:**
  - The **runner** does a probing call to the judge stub; on `NotImplementedError` from
    `build_prompt` / `iter_presentations` / `parse_response` it raises `JudgeNotImplemented`
    with a message naming the exact file/method to implement (the CLI prints this and exits 1,
    suggesting `eval-platform smoke` to test plumbing without a rubric).
  - The **report** and **dashboard** wrap every stats call in `try/except NotImplementedError`
    and render a "not yet implemented — hand-code in `stats/...`" placeholder. Plots only
    *render* values handed to them; they never compute a statistic.
- **`smoke` proves the spine independent of the cores.** It pushes a fixed diagnostic prompt
  (not a rubric) through provider → retry → rate-limit → store → cache, writing
  `PointwiseJudgment` rows with `judge_model='__smoke__'`, `prompt_version='__smoke__'`,
  `parse_ok=False`. A second run hits the cache (`cached=True`).

What the user owns: everything tagged `[STUB]` above — all of `stats/`, the two judge classes
(prompts + parsers + swap), `datasets/sampling.py`, and `trajectory/score.py`. Verify the
statistics with `handcode/verify_stats.py`.

---

## Control flow of a run

```
CLI run  ──>  load_items / load_candidates (datasets/loader)
         ──>  get_provider(settings)            (dummy | litellm)
         ──>  Store.open(settings)              (mkdir, connect, init schema)
         ──>  PointwiseJudge|PairwiseJudge(prompt_version=settings.prompt_version)   [STUB]
         ──>  Runner(settings, store, provider, cache, rate_limiter)
                 • insert RunMeta; upsert items + candidates
                 • probe the judge stub  ──> JudgeNotImplemented if unimplemented
                 • ThreadPoolExecutor(max_workers=settings.max_concurrency):
                     for each task:
                       cache hit? ── yes ─> persist cached copy (cached=True)
                                  └─ no  ─> rate_limiter.acquire()
                                            retry_call(provider.complete, max_retries=...)
                                            judge.parse_response(raw)   [STUB]
                                              • ValueError -> parse_ok=False, keep raw
                                            (pairwise) resolve winner_candidate_id from
                                              parsed winner_slot + presentation slot assignment
                                            store the judgment (raw_response always kept)
                 • finish_run(...)  ──> return RunMeta
         ──>  report.build_report(store, run_id)   (CALLS stats, tolerates NotImplementedError)
```

---

## Import & run conventions

- **Package & layout.** src-layout: the package lives at `src/evalplatform/`, importable as
  `evalplatform` after `pip install -e .`, or by putting `src/` on `sys.path`
  (`handcode/verify_stats.py` does the latter so it runs without an install).
- **Entry point.** `pyproject.toml` maps `eval-platform → evalplatform.cli:app`. The CLI is
  the supported interface; see the command table in `README.md`.
- **Configuration.** All runtime knobs are `EVAL_*` env vars / `.env`, read by
  `config.Settings` (singleton via `get_settings()`; defaults run offline with no keys).
  `.env.example` documents every variable. Provider API keys (`GEMINI_API_KEY`,
  `DEEPSEEK_API_KEY`, ...) are read by **litellm itself**, not by the `EVAL_` settings.
- **Dependencies.** Always-available: `pydantic`, `pydantic-settings`, `numpy`, `scipy`,
  `scikit-learn`, `pandas`, `matplotlib`, `typer`, `pyyaml`, `python-dotenv`, stdlib.
  **Optional / lazy-imported only** (never at module top level; raise a clear install hint if
  missing): `litellm`, `tenacity`, `diskcache`, `streamlit`, `plotly`, `pytest`. Install via
  extras: `.[providers]`, `.[reliability]`, `.[dashboard]`, `.[dev]`, `.[all]`.
- **Storage.** SQLite at `EVAL_DB_PATH` (default `data/eval.db`, WAL mode). `eval-platform
  init-db` creates it; the `Store` is a context manager. Reports are written under
  `EVAL_REPORTS_DIR` (default `reports/`). Neither directory is committed.
- **Determinism.** The `dummy` provider derives its text from a hash (stable across runs);
  `retry.py` jitter uses a locally-seeded `random.Random` — no global/nondeterministic
  randomness in the plumbing, so smoke runs and tests are reproducible.
- **Module docstrings.** Every scaffolded module opens with a one-line docstring noting it is
  scaffolding (AI); every stub module states its contract, the oracle to verify against, and
  a hint — but never the implementation.
