"""SQLite storage for the decision log and pending approvals."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id TEXT NOT NULL,
    category TEXT,
    category_probs TEXT,
    matched_id INTEGER,
    match_prob REAL,
    action TEXT,
    guard_decision TEXT,
    reason TEXT,
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


class DecisionLog:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(DECISIONS_DDL)

    def close(self) -> None:
        self._conn.close()

    def log(self, entry: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO decisions (email_id, category, category_probs, matched_id, "
            "match_prob, action, guard_decision, reason, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                entry.get("email_id"),
                entry.get("category"),
                json.dumps(entry.get("category_probs", {})),
                entry.get("matched_id"),
                entry.get("match_prob"),
                entry.get("action"),
                entry.get("guard_decision"),
                entry.get("reason"),
                _now(),
            ),
        )
        self._conn.commit()

    def has(self, email_id: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM decisions WHERE email_id = ? LIMIT 1", (email_id,))
        return cur.fetchone() is not None


class PendingApprovals:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
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
