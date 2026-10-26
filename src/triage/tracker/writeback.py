"""Write status changes to the source tracker through career-ops' `set-status.mjs`."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

# set-status.mjs exits 4 when the tracker lock is held; the write can be retried.
LOCK_TIMEOUT = 4
TIMEOUT_S = 60.0

Runner = Callable[..., Any]


class TrackerWriteError(Exception):
    """set-status.mjs failed; ``retryable`` when the tracker was busy."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def write_status(
    node: str,
    script: Path,
    row: int,
    status: str,
    on: date | None = None,
    runner: Runner = subprocess.run,
) -> bool:
    """Set ``row``'s status in the source tracker; return whether it changed."""
    args = [node, str(script), "--row", str(row), status, "--json"]
    if on is not None:
        args += ["--on", on.isoformat()]
    try:
        done = runner(
            args, cwd=script.parent, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TrackerWriteError(f"set-status timed out for row {row}", retryable=True) from exc
    except OSError as exc:
        raise TrackerWriteError(f"cannot run {node} {script}: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stdout or done.stderr or "").strip()[:200]
        raise TrackerWriteError(
            f"set-status exited {done.returncode} for row {row}: {detail}",
            retryable=done.returncode == LOCK_TIMEOUT,
        )
    try:
        return bool(json.loads(done.stdout)["changed"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        message = f"set-status gave no result for row {row}: {done.stdout!r:.200}"
        raise TrackerWriteError(message) from exc
