"""Copy a tracker snapshot into this project's data dir and load it into SQLite."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from triage.config import Config
from triage.tracker.parser import (
    parse_active_interviews,
    parse_applications,
    parse_status_log,
)
from triage.tracker.schema import ActiveInterview, Application, StatusLogRow

APPLICATIONS_DDL = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY,
    date_applied TEXT NOT NULL,
    company TEXT NOT NULL,
    via TEXT NOT NULL,
    role TEXT NOT NULL,
    location TEXT NOT NULL,
    score TEXT NOT NULL,
    status TEXT NOT NULL,
    pdf TEXT NOT NULL,
    report TEXT NOT NULL,
    notes TEXT NOT NULL,
    url TEXT NOT NULL,
    requisition_id TEXT NOT NULL DEFAULT ''
)
"""

STATUS_LOG_DDL = """
CREATE TABLE IF NOT EXISTS status_log (
    tracker_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT NOT NULL
)
"""

ACTIVE_INTERVIEWS_DDL = """
CREATE TABLE IF NOT EXISTS active_interviews (
    company TEXT NOT NULL,
    role TEXT NOT NULL,
    round TEXT NOT NULL,
    date_time TEXT,
    interviewer TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT NOT NULL
)
"""


def copy_snapshot(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def _write_applications(conn: sqlite3.Connection, apps: list[Application]) -> None:
    conn.execute("DELETE FROM applications")
    conn.executemany(
        "INSERT INTO applications VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                a.id,
                a.date_applied.isoformat(),
                a.company,
                a.via,
                a.role,
                a.location,
                a.score,
                a.status,
                a.pdf,
                a.report,
                a.notes,
                a.url,
                a.requisition_id,
            )
            for a in apps
        ],
    )


def _write_status_log(conn: sqlite3.Connection, rows: list[StatusLogRow]) -> None:
    conn.execute("DELETE FROM status_log")
    conn.executemany(
        "INSERT INTO status_log VALUES (?,?,?,?,?,?)",
        [
            (r.tracker_id, r.date.isoformat(), r.from_, r.to, r.source, r.note)
            for r in rows
        ],
    )


def _write_active_interviews(conn: sqlite3.Connection, rows: list[ActiveInterview]) -> None:
    conn.execute("DELETE FROM active_interviews")
    conn.executemany(
        "INSERT INTO active_interviews VALUES (?,?,?,?,?,?,?)",
        [
            (
                r.company,
                r.role,
                r.round,
                r.date_time.isoformat() if r.date_time else None,
                r.interviewer,
                r.status,
                r.notes,
            )
            for r in rows
        ],
    )


def load_into_sqlite(
    db_path: Path,
    apps: list[Application],
    status_log: list[StatusLogRow],
    interviews: list[ActiveInterview],
) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS applications")
        conn.execute(APPLICATIONS_DDL)
        conn.execute(STATUS_LOG_DDL)
        conn.execute(ACTIVE_INTERVIEWS_DDL)
        _write_applications(conn, apps)
        _write_status_log(conn, status_log)
        _write_active_interviews(conn, interviews)
        conn.commit()
    finally:
        conn.close()
    return len(apps)


def snapshot(cfg: Config) -> int:
    copy_snapshot(cfg.tracker.source_applications, cfg.tracker.snapshot_md)
    apps = parse_applications(cfg.tracker.snapshot_md.read_text())
    status_log = parse_status_log(cfg.tracker.source_status_log.read_text())
    interviews = parse_active_interviews(cfg.tracker.source_active_interviews.read_text())
    return load_into_sqlite(cfg.tracker.db, apps, status_log, interviews)
