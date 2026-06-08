"""SQLite persistence layer (scaffolding, AI).

Stdlib ``sqlite3`` only. JSON-encodes list/dict columns and converts rows <-> the
Pydantic models in :mod:`evalplatform.models`. The judgment tables double as the
judge cache: the unique indexes ``idx_pw_cache`` / ``idx_pr_cache`` make a re-run
idempotent, and ``INSERT OR REPLACE`` upserts keep the cache key authoritative.

Thread-safety: the connection is opened with ``check_same_thread=False`` so the Runner's
``ThreadPoolExecutor`` can share one :class:`Store`, and every method that touches the
connection holds a re-entrant lock for the full execute(+fetch)+commit span. A single
``sqlite3.Connection`` is NOT safe for concurrent use, so this lock — not the GIL — is what
serializes access. Each write is also idempotent on its cache key (``INSERT OR REPLACE``),
so a concurrent get-then-insert at most re-writes the same row.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from ..models import (
    Candidate,
    EvalItem,
    JudgeMode,
    PairwiseJudgment,
    PointwiseJudgment,
    Position,
    RunMeta,
)


def _dumps(value: Any) -> Optional[str]:
    """JSON-encode a list/dict column; ``None`` stays ``None``."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Optional[str], default: Any) -> Any:
    """Decode a JSON column, falling back to ``default`` when empty/null."""
    if value is None or value == "":
        return default
    return json.loads(value)


def _to_bool_int(value: bool) -> int:
    """SQLite stores booleans as 0/1."""
    return 1 if value else 0


def _position_value(position: Position | str) -> str:
    """Accept either a :class:`Position` enum or its ``.value`` string."""
    return position.value if isinstance(position, Position) else str(position)


class Store:
    """Thin SQLite wrapper that speaks Pydantic models. Safe to share across threads."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        # check_same_thread=False so the Runner's ThreadPoolExecutor can share one Store;
        # `_lock` (re-entrant) serializes every execute(+fetch)+commit so the shared
        # connection is never touched by two threads at once.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._lock = threading.RLock()

    # --- lifecycle -------------------------------------------------------------------

    def init_db(self) -> None:
        """Execute the bundled schema (idempotent: schema uses IF NOT EXISTS)."""
        schema_path = Path(__file__).with_name("schema.sql")
        sql = schema_path.read_text(encoding="utf-8")
        with self._lock:
            self.conn.executescript(sql)
            self.conn.commit()

    @classmethod
    def open(cls, settings: Any) -> "Store":
        """Open a store from settings: mkdir parent, connect, init schema."""
        db_path = Path(settings.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(db_path)
        store.init_db()
        return store

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- runs ------------------------------------------------------------------------

    def insert_run(self, run: RunMeta) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, mode, judge_model, prompt_version, dataset_id, dataset_version,
                    n_items, git_sha, config_snapshot, started_at, finished_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.mode.value if isinstance(run.mode, JudgeMode) else str(run.mode),
                    run.judge_model,
                    run.prompt_version,
                    run.dataset_id,
                    run.dataset_version,
                    run.n_items,
                    run.git_sha,
                    _dumps(run.config_snapshot),
                    run.started_at,
                    run.finished_at,
                    run.notes,
                ),
            )
            self.conn.commit()

    def finish_run(self, run_id: str, *, finished_at: float, n_items: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE runs SET finished_at = ?, n_items = ? WHERE run_id = ?",
                (finished_at, n_items, run_id),
            )
            self.conn.commit()

    def get_run(self, run_id: str) -> Optional[RunMeta]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_run(row) if row else None

    def latest_run(self, mode: str | None = None) -> Optional[RunMeta]:
        with self._lock:
            if mode is not None:
                row = self.conn.execute(
                    "SELECT * FROM runs WHERE mode = ? ORDER BY started_at DESC LIMIT 1",
                    (mode,),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
        return self._row_to_run(row) if row else None

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> RunMeta:
        return RunMeta(
            run_id=row["run_id"],
            mode=JudgeMode(row["mode"]),
            judge_model=row["judge_model"],
            prompt_version=row["prompt_version"],
            dataset_id=row["dataset_id"],
            dataset_version=row["dataset_version"],
            n_items=row["n_items"],
            git_sha=row["git_sha"],
            config_snapshot=_loads(row["config_snapshot"], {}),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            notes=row["notes"],
        )

    # --- items -----------------------------------------------------------------------

    def upsert_item(self, item: EvalItem) -> None:
        self.upsert_items([item])

    def upsert_items(self, items: Sequence[EvalItem]) -> None:
        with self._lock:
            for item in items:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO items
                        (id, input, reference, context, stratum, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.input,
                        item.reference,
                        item.context,
                        item.stratum,
                        _dumps(item.tags),
                        _dumps(item.metadata),
                    ),
                )
            self.conn.commit()

    def get_item(self, item_id: str) -> Optional[EvalItem]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM items WHERE id = ?", (item_id,)
            ).fetchone()
        return self._row_to_item(row) if row else None

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> EvalItem:
        return EvalItem(
            id=row["id"],
            input=row["input"],
            reference=row["reference"],
            context=row["context"],
            stratum=row["stratum"],
            tags=_loads(row["tags"], []),
            metadata=_loads(row["metadata"], {}),
        )

    # --- candidates ------------------------------------------------------------------

    def upsert_candidate(self, c: Candidate) -> None:
        self.upsert_candidates([c])

    def upsert_candidates(self, cs: Sequence[Candidate]) -> None:
        with self._lock:
            for c in cs:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO candidates (id, item_id, system, output, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (c.id, c.item_id, c.system, c.output, _dumps(c.metadata)),
                )
            self.conn.commit()

    def get_candidates(self, item_id: str) -> list[Candidate]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM candidates WHERE item_id = ? ORDER BY id", (item_id,)
            ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> Candidate:
        return Candidate(
            id=row["id"],
            item_id=row["item_id"],
            system=row["system"],
            output=row["output"],
            metadata=_loads(row["metadata"], {}),
        )

    # --- pointwise judgments ---------------------------------------------------------

    def insert_pointwise(self, j: PointwiseJudgment) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO pointwise_judgments (
                    id, run_id, item_id, candidate_id, judge_model, prompt_version,
                    score, label, rationale, raw_response, parse_ok, cached, latency_ms,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    j.id,
                    j.run_id,
                    j.item_id,
                    j.candidate_id,
                    j.judge_model,
                    j.prompt_version,
                    j.score,
                    j.label,
                    j.rationale,
                    j.raw_response,
                    _to_bool_int(j.parse_ok),
                    _to_bool_int(j.cached),
                    j.latency_ms,
                    j.created_at,
                    _dumps(j.metadata),
                ),
            )
            self.conn.commit()

    def pointwise_for_run(self, run_id: str) -> list[PointwiseJudgment]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM pointwise_judgments WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [self._row_to_pointwise(r) for r in rows]

    @staticmethod
    def _row_to_pointwise(row: sqlite3.Row) -> PointwiseJudgment:
        return PointwiseJudgment(
            id=row["id"],
            run_id=row["run_id"],
            item_id=row["item_id"],
            candidate_id=row["candidate_id"],
            judge_model=row["judge_model"],
            prompt_version=row["prompt_version"],
            score=row["score"],
            label=row["label"],
            rationale=row["rationale"],
            raw_response=row["raw_response"],
            parse_ok=bool(row["parse_ok"]),
            cached=bool(row["cached"]),
            latency_ms=row["latency_ms"],
            created_at=row["created_at"],
            metadata=_loads(row["metadata"], {}),
        )

    def get_cached_pointwise(
        self, candidate_id: str, judge_model: str, prompt_version: str
    ) -> Optional[PointwiseJudgment]:
        """Cache lookup via the unique key (candidate_id, judge_model, prompt_version)."""
        with self._lock:
            row = self.conn.execute(
                """
                SELECT * FROM pointwise_judgments
                WHERE candidate_id = ? AND judge_model = ? AND prompt_version = ?
                """,
                (candidate_id, judge_model, prompt_version),
            ).fetchone()
        return self._row_to_pointwise(row) if row else None

    # --- pairwise judgments ----------------------------------------------------------

    def insert_pairwise(self, j: PairwiseJudgment) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO pairwise_judgments (
                    id, run_id, item_id, candidate_a_id, candidate_b_id, position,
                    winner_slot, winner_candidate_id, judge_model, prompt_version,
                    rationale, raw_response, parse_ok, cached, latency_ms, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    j.id,
                    j.run_id,
                    j.item_id,
                    j.candidate_a_id,
                    j.candidate_b_id,
                    _position_value(j.position),
                    j.winner_slot,
                    j.winner_candidate_id,
                    j.judge_model,
                    j.prompt_version,
                    j.rationale,
                    j.raw_response,
                    _to_bool_int(j.parse_ok),
                    _to_bool_int(j.cached),
                    j.latency_ms,
                    j.created_at,
                    _dumps(j.metadata),
                ),
            )
            self.conn.commit()

    def pairwise_for_run(self, run_id: str) -> list[PairwiseJudgment]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM pairwise_judgments WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [self._row_to_pairwise(r) for r in rows]

    @staticmethod
    def _row_to_pairwise(row: sqlite3.Row) -> PairwiseJudgment:
        return PairwiseJudgment(
            id=row["id"],
            run_id=row["run_id"],
            item_id=row["item_id"],
            candidate_a_id=row["candidate_a_id"],
            candidate_b_id=row["candidate_b_id"],
            position=Position(row["position"]),
            winner_slot=row["winner_slot"],
            winner_candidate_id=row["winner_candidate_id"],
            judge_model=row["judge_model"],
            prompt_version=row["prompt_version"],
            rationale=row["rationale"],
            raw_response=row["raw_response"],
            parse_ok=bool(row["parse_ok"]),
            cached=bool(row["cached"]),
            latency_ms=row["latency_ms"],
            created_at=row["created_at"],
            metadata=_loads(row["metadata"], {}),
        )

    def get_cached_pairwise(
        self,
        candidate_a_id: str,
        candidate_b_id: str,
        position: Position | str,
        judge_model: str,
        prompt_version: str,
    ) -> Optional[PairwiseJudgment]:
        """Cache lookup via the unique key (a_id, b_id, position, judge_model, prompt_version)."""
        with self._lock:
            row = self.conn.execute(
                """
                SELECT * FROM pairwise_judgments
                WHERE candidate_a_id = ? AND candidate_b_id = ? AND position = ?
                  AND judge_model = ? AND prompt_version = ?
                """,
                (
                    candidate_a_id,
                    candidate_b_id,
                    _position_value(position),
                    judge_model,
                    prompt_version,
                ),
            ).fetchone()
        return self._row_to_pairwise(row) if row else None

    # --- gold labels -----------------------------------------------------------------

    def insert_gold_label(
        self,
        *,
        item_id: str,
        labeler: str,
        label: Optional[str] = None,
        score: Optional[float] = None,
        candidate_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        from ..models import new_id

        gid = new_id("gold")
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO gold_labels (
                    id, item_id, candidate_id, labeler, label, score, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gid,
                    item_id,
                    candidate_id,
                    labeler,
                    label,
                    score,
                    _now_seconds(),
                    _dumps(metadata or {}),
                ),
            )
            self.conn.commit()
        return gid

    def gold_labels(self, item_id: str | None = None) -> list[dict]:
        with self._lock:
            if item_id is not None:
                rows = self.conn.execute(
                    "SELECT * FROM gold_labels WHERE item_id = ? ORDER BY created_at",
                    (item_id,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM gold_labels ORDER BY created_at"
                ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            d["metadata"] = _loads(r["metadata"], {})
            out.append(d)
        return out


def _now_seconds() -> float:
    """Wall-clock seconds; local helper so gold-label timestamps need no model import dance."""
    import time

    return time.time()
