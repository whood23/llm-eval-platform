-- SQLite schema for the eval platform. Mirrors src/evalplatform/models.py.
--
-- Everything Phase-2 instrument-validation needs is persisted here: the raw judge
-- response, judge model, prompt version, candidate position/order, and timestamps.
-- The `judgment` tables double as the judgment cache (see store/db.py / judge/cache.py):
-- a cache hit is "a row already exists for this (judge_model, prompt_version, item,
-- candidate(s), position) key".

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One row per eval run (provenance / reproducibility).
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    mode            TEXT NOT NULL,            -- 'pointwise' | 'pairwise'
    judge_model     TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    dataset_id      TEXT,
    dataset_version TEXT,
    n_items         INTEGER NOT NULL DEFAULT 0,
    git_sha         TEXT,
    config_snapshot TEXT,                     -- JSON
    started_at      REAL NOT NULL,
    finished_at     REAL,
    notes           TEXT
);

-- Eval items (the dataset). Stable across runs; keyed by their own id.
CREATE TABLE IF NOT EXISTS items (
    id        TEXT PRIMARY KEY,
    input     TEXT NOT NULL,
    reference TEXT,
    context   TEXT,
    stratum   TEXT,
    tags      TEXT,                           -- JSON array
    metadata  TEXT                            -- JSON object
);

-- Candidate responses under evaluation.
CREATE TABLE IF NOT EXISTS candidates (
    id       TEXT PRIMARY KEY,
    item_id  TEXT NOT NULL REFERENCES items(id),
    system   TEXT NOT NULL,
    output   TEXT NOT NULL,
    metadata TEXT                             -- JSON object
);
CREATE INDEX IF NOT EXISTS idx_candidates_item ON candidates(item_id);

-- Pointwise rubric judgments.
CREATE TABLE IF NOT EXISTS pointwise_judgments (
    id             TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    item_id        TEXT NOT NULL REFERENCES items(id),
    candidate_id   TEXT NOT NULL REFERENCES candidates(id),
    judge_model    TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    score          REAL,
    label          TEXT,
    rationale      TEXT,
    raw_response   TEXT,
    parse_ok       INTEGER NOT NULL DEFAULT 0,
    cached         INTEGER NOT NULL DEFAULT 0,
    latency_ms     REAL,
    created_at     REAL NOT NULL,
    metadata       TEXT
);
CREATE INDEX IF NOT EXISTS idx_pw_run ON pointwise_judgments(run_id);
-- Cache key: a unique (candidate, judge, prompt-version) judgment.
CREATE UNIQUE INDEX IF NOT EXISTS idx_pw_cache
    ON pointwise_judgments(candidate_id, judge_model, prompt_version);

-- Pairwise A-vs-B judgments. Two rows per logical comparison (AB and BA presentations);
-- `position` records the physical order so position bias can be measured later.
CREATE TABLE IF NOT EXISTS pairwise_judgments (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    item_id             TEXT NOT NULL REFERENCES items(id),
    candidate_a_id      TEXT NOT NULL REFERENCES candidates(id),
    candidate_b_id      TEXT NOT NULL REFERENCES candidates(id),
    position            TEXT NOT NULL,        -- 'AB' | 'BA'
    winner_slot         TEXT,                 -- 'A' | 'B' | 'tie'
    winner_candidate_id TEXT,
    judge_model         TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    rationale           TEXT,
    raw_response        TEXT,
    parse_ok            INTEGER NOT NULL DEFAULT 0,
    cached              INTEGER NOT NULL DEFAULT 0,
    latency_ms          REAL,
    created_at          REAL NOT NULL,
    metadata            TEXT
);
CREATE INDEX IF NOT EXISTS idx_pr_run ON pairwise_judgments(run_id);
-- Cache key: a unique (pair-in-this-slot-order, position, judge, prompt-version) judgment.
CREATE UNIQUE INDEX IF NOT EXISTS idx_pr_cache
    ON pairwise_judgments(candidate_a_id, candidate_b_id, position, judge_model, prompt_version);

-- Optional: human gold labels for Phase-2c judge-vs-human agreement (kappa / alpha).
-- One row per (item or candidate) per labeler. The agreement *statistics* are hand-coded
-- in stats/agreement.py; this table just stores the labels they consume.
CREATE TABLE IF NOT EXISTS gold_labels (
    id           TEXT PRIMARY KEY,
    item_id      TEXT NOT NULL REFERENCES items(id),
    candidate_id TEXT REFERENCES candidates(id),
    labeler      TEXT NOT NULL,               -- which human (or judge) produced this label
    label        TEXT,                        -- categorical label
    score        REAL,                        -- or numeric score
    created_at   REAL NOT NULL,
    metadata     TEXT
);
CREATE INDEX IF NOT EXISTS idx_gold_item ON gold_labels(item_id);
