"""Read-only access to the local tracker SQLite snapshot."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from triage.tracker.schema import Application


def _row_to_application(row: sqlite3.Row) -> Application:
    return Application(
        id=int(row["id"]),
        date_applied=date.fromisoformat(row["date_applied"]),
        company=row["company"],
        via=row["via"],
        role=row["role"],
        location=row["location"],
        score=row["score"],
        status=row["status"],
        pdf=row["pdf"],
        report=row["report"],
        notes=row["notes"],
        url=row["url"],
        requisition_id=row["requisition_id"],
    )


class TrackerDB:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def search(self, query: str, limit: int = 10) -> list[Application]:
        like = f"%{query.strip()}%"
        cur = self._conn.execute(
            "SELECT * FROM applications WHERE company LIKE ? OR role LIKE ? LIMIT ?",
            (like, like, limit),
        )
        return [_row_to_application(r) for r in cur.fetchall()]

    def search_tracker(
        self, company: str = "", role: str = "", limit: int = 10
    ) -> list[Application]:
        company = company.strip().lower()
        role = role.strip().lower()
        if company and role:
            cur = self._conn.execute(
                "SELECT * FROM applications "
                "WHERE lower(company) LIKE ? AND lower(role) LIKE ? LIMIT ?",
                (f"%{company}%", f"%{role}%", limit),
            )
        elif company:
            cur = self._conn.execute(
                "SELECT * FROM applications WHERE lower(company) LIKE ? LIMIT ?",
                (f"%{company}%", limit),
            )
        elif role:
            cur = self._conn.execute(
                "SELECT * FROM applications WHERE lower(role) LIKE ? LIMIT ?",
                (f"%{role}%", limit),
            )
        else:
            cur = self._conn.execute("SELECT * FROM applications LIMIT ?", (limit,))
        return [_row_to_application(r) for r in cur.fetchall()]

    def get(self, tracker_id: int) -> Application | None:
        cur = self._conn.execute("SELECT * FROM applications WHERE id = ?", (tracker_id,))
        row = cur.fetchone()
        return _row_to_application(row) if row is not None else None

    def ids(self) -> set[int]:
        cur = self._conn.execute("SELECT id FROM applications")
        return {r["id"] for r in cur.fetchall()}
