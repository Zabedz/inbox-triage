"""Update a tracker row's status in the local SQLite snapshot."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def set_status(db_path: str | Path, tracker_id: int, status: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE applications SET status = ? WHERE id = ?", (status, tracker_id))
        conn.commit()
    finally:
        conn.close()
