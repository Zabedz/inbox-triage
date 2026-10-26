from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from triage.tracker.writeback import LOCK_TIMEOUT, TrackerWriteError, write_status

SCRIPT = Path("/career-ops/set-status.mjs")


def runner(returncode: int = 0, stdout: str = '{"changed": true}', calls: list | None = None):
    def run(args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

    return run


def test_writes_row_status_and_event_date() -> None:
    calls: list = []

    changed = write_status("node", SCRIPT, 407, "Rejected", date(2026, 10, 12), runner(calls=calls))

    assert changed is True
    args, kwargs = calls[0]
    assert args == ["node", str(SCRIPT), "--row", "407", "Rejected", "--json", "--on", "2026-10-12"]
    assert kwargs["cwd"] == SCRIPT.parent


def test_no_op_reports_unchanged() -> None:
    assert write_status("node", SCRIPT, 363, "Rejected", runner=runner(stdout='{"changed": false}')) is False


@pytest.mark.parametrize(("code", "retryable"), [(LOCK_TIMEOUT, True), (2, False), (3, False)])
def test_failure_raises_with_retry_hint(code: int, retryable: bool) -> None:
    with pytest.raises(TrackerWriteError) as err:
        write_status("node", SCRIPT, 9, "Offer", runner=runner(code, '{"error": "x"}'))
    assert err.value.retryable is retryable
    assert "row 9" in str(err.value)


def test_missing_node_is_not_retryable() -> None:
    def missing(args, **kwargs):
        raise FileNotFoundError("node")

    with pytest.raises(TrackerWriteError) as err:
        write_status("node", SCRIPT, 1, "Rejected", runner=missing)
    assert err.value.retryable is False


def test_unparseable_output_raises() -> None:
    with pytest.raises(TrackerWriteError):
        write_status("node", SCRIPT, 1, "Rejected", runner=runner(stdout="ok"))
