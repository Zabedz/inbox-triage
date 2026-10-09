"""SQLite storage for the decision log and pending approvals."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# An email that fails this many runs is left for manual triage.
MAX_ATTEMPTS = 3

DECISION_COLUMNS = (
    "email_id",
    "category",
    "category_probs",
    "matched_id",
    "match_prob",
    "action",
    "action_prob",
    "has_draft",
    "guard_decision",
    "reason",
    "reviewed_action",
    "verdict",
    "category_threshold",
    "match_threshold",
    "decision_backend",
    "llm_model",
)

DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id TEXT NOT NULL,
    category TEXT,
    category_probs TEXT,
    matched_id INTEGER,
    match_prob REAL,
    action TEXT,
    action_prob REAL,
    has_draft INTEGER,
    guard_decision TEXT,
    reason TEXT,
    reviewed_action TEXT,
    verdict TEXT,
    category_threshold REAL,
    match_threshold REAL,
    decision_backend TEXT,
    llm_model TEXT,
    timestamp TEXT NOT NULL
)
"""

PENDING_DDL = """
CREATE TABLE IF NOT EXISTS pending_approvals (
    thread_id TEXT PRIMARY KEY,
    email_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(path: str | Path) -> sqlite3.Connection:
    # Graph nodes may run off the calling thread.
    return sqlite3.connect(str(path), check_same_thread=False)


class DecisionLog:
    def __init__(self, path: str | Path, context: dict[str, Any] | None = None) -> None:
        """``context`` holds fields fixed for a run (backend, model) merged into every row."""
        self._conn = _connect(path)
        self._conn.execute(DECISIONS_DDL)
        self._add_missing_columns()
        self._context = context or {}

    def _add_missing_columns(self) -> None:
        """Bring a table created by an older schema up to ``DECISION_COLUMNS``."""
        present = {row[1] for row in self._conn.execute("PRAGMA table_info(decisions)")}
        for column in DECISION_COLUMNS:
            if column not in present:
                self._conn.execute(f"ALTER TABLE decisions ADD COLUMN {column}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def log(self, entry: dict[str, Any]) -> None:
        row = {**self._context, **entry}
        row["category_probs"] = json.dumps(row.get("category_probs") or {})
        values = [row.get(col) for col in DECISION_COLUMNS]
        placeholders = ",".join("?" * (len(DECISION_COLUMNS) + 1))
        self._conn.execute(
            f"INSERT INTO decisions ({','.join(DECISION_COLUMNS)}, timestamp) "  # noqa: S608
            f"VALUES ({placeholders})",
            (*values, _now()),
        )
        self._conn.commit()

    def log_error(self, email_id: str, error: str) -> None:
        self.log({"email_id": email_id, "guard_decision": "error", "reason": error})

    def is_done(self, email_id: str) -> bool:
        """True once the email has a decision or has failed ``MAX_ATTEMPTS`` runs."""
        cur = self._conn.execute(
            "SELECT SUM(guard_decision IS NOT 'error'), SUM(guard_decision IS 'error') "
            "FROM decisions WHERE email_id = ?",
            (email_id,),
        )
        decided, errors = cur.fetchone()
        return bool(decided) or (errors or 0) >= MAX_ATTEMPTS


class PendingApprovals:
    def __init__(self, path: str | Path) -> None:
        self._conn = _connect(path)
        self._conn.execute(PENDING_DDL)

    def close(self) -> None:
        self._conn.close()

    def add(self, thread_id: str, email_id: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pending_approvals (thread_id, email_id, payload, timestamp) "
            "VALUES (?,?,?,?)",
            (thread_id, email_id, json.dumps(payload), _now()),
        )
        self._conn.commit()

    def list(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT thread_id, email_id, payload FROM pending_approvals ORDER BY timestamp"
        )
        return [
            {"thread_id": r[0], "email_id": r[1], "payload": json.loads(r[2])}
            for r in cur.fetchall()
        ]

    def has(self, email_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM pending_approvals WHERE email_id = ? LIMIT 1", (email_id,)
        )
        return cur.fetchone() is not None

    def remove(self, thread_id: str) -> None:
        self._conn.execute("DELETE FROM pending_approvals WHERE thread_id = ?", (thread_id,))
        self._conn.commit()
