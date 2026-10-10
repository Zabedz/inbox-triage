"""`triage stats`: what the agent did in real use, from the decision log."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from triage.storage import MAX_ATTEMPTS, count_pending, read_decisions

# Verdicts where the reviewer answered; agreement is measured over these.
ANSWERED = {"approved", "edited", "rejected"}


def stats(
    db_path: str | Path, days: int | None = None, now: datetime | None = None
) -> dict[str, Any]:
    """Counts over decision rows logged in the last ``days`` (all rows when None).

    Reviewed emails are logged when the review completes, so the window is by log time.
    """
    since = None
    if days is not None:
        since = ((now or datetime.now(UTC)) - timedelta(days=days)).isoformat()
    rows = read_decisions(db_path, since)

    decided = [r for r in rows if r["guard_decision"] != "error"]
    reviewed = [
        r for r in decided if r["guard_decision"] == "approval" and r["verdict"] in ANSWERED
    ]
    agreed = [
        r for r in reviewed if r["verdict"] == "approved" and r["reviewed_action"] == r["action"]
    ]
    decided_ids = {r["email_id"] for r in decided}
    failures = Counter(r["email_id"] for r in rows if r["guard_decision"] == "error")
    by_category: dict[str, Counter[str]] = {}
    for r in decided:
        by_category.setdefault(r["category"] or "unknown", Counter())[r["guard_decision"]] += 1
    return {
        "emails": len(decided_ids),
        "routes": dict(Counter(r["guard_decision"] for r in decided)),
        "queued": count_pending(db_path),
        "auto_actions": dict(
            Counter(r["action"] for r in decided if r["guard_decision"] == "auto")
        ),
        "verdicts": dict(
            Counter(r["verdict"] for r in decided if r["guard_decision"] == "approval")
        ),
        "reviewed": len(reviewed),
        "approved_as_proposed": len(agreed),
        "error_rows": sum(failures.values()),
        "abandoned": sorted(
            e for e, n in failures.items() if n >= MAX_ATTEMPTS and e not in decided_ids
        ),
        "backends": dict(Counter(r["decision_backend"] or "unknown" for r in decided)),
        "by_category": {c: dict(counts) for c, counts in sorted(by_category.items())},
    }


def render(summary: dict[str, Any]) -> str:
    reviewed = summary["reviewed"]
    agreement = f"{summary['approved_as_proposed'] / reviewed:.0%}" if reviewed else "n/a"
    lines = [
        f"emails decided: {summary['emails']}  backends: {summary['backends']}",
        f"routes: {summary['routes']}  still queued for review: {summary['queued']}",
        f"auto-applied actions: {summary['auto_actions']}",
        f"review verdicts: {summary['verdicts']}",
        f"approved as proposed: {summary['approved_as_proposed']}/{reviewed} ({agreement})",
        f"error rows: {summary['error_rows']}  abandoned emails: {len(summary['abandoned'])}",
        "by category:",
        *(f"  {c}: {counts}" for c, counts in summary["by_category"].items()),
    ]
    return "\n".join(lines)
