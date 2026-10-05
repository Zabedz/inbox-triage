"""Custom promptfoo assertions for field accuracy."""

from __future__ import annotations

import json
import re
from typing import Any

_TZ = {
    "gmt": "etc/utc",
    "utc": "etc/utc",
    "pst": "america/los_angeles",
    "est": "america/new_york",
    "bst": "europe/london",
}

_NORMALIZED_FIELDS = {"company", "role", "requisition_id", "agency", "contact"}


def _norm(value: str) -> str:
    return "".join(c for c in value.lower() if c.isalnum())


def _norm_tz(value: str) -> str:
    return _TZ.get(value.strip().lower(), value.strip().lower())


def _date_part(value: str | None) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", value or "")
    return match.group(1) if match else None


def _time_part(value: str | None) -> str | None:
    match = re.search(r"(\d{1,2}:\d{2})", value or "")
    return match.group(1) if match else None


def _compare(field: str, predicted: str | None, expected: str | None) -> bool:
    if not expected:
        return not predicted
    if not predicted:
        return False
    if field in _NORMALIZED_FIELDS:
        return _norm(predicted) == _norm(expected)
    if field == "timezone":
        return _norm_tz(predicted) == _norm_tz(expected)
    if field in ("date_time", "deadline"):
        if _date_part(predicted) != _date_part(expected):
            return False
        if _time_part(expected) and _time_part(predicted) != _time_part(expected):
            return False
        return True
    return predicted.strip() == expected.strip()


def get_assert(output: str, context: dict[str, Any]) -> dict[str, Any]:
    expected = context.get("vars", {}).get("extracted", {})
    if isinstance(expected, str):
        expected = json.loads(expected)
    try:
        predicted = json.loads(output) if isinstance(output, str) else output
    except (json.JSONDecodeError, TypeError):
        return {"pass": False, "score": 0, "reason": "output is not valid JSON"}

    mismatches = []
    for field, expected_value in expected.items():
        if not _compare(field, predicted.get(field), expected_value):
            mismatches.append(f"{field}: got {predicted.get(field)!r}, want {expected_value!r}")
    if mismatches:
        return {"pass": False, "score": 0, "reason": "; ".join(mismatches)}
    return {"pass": True, "score": 1, "reason": "all fields match"}


def match_assert(output: str, context: dict[str, Any]) -> dict[str, Any]:
    expected = context.get("vars", {}).get("matched_tracker_id")
    if expected == "none":
        expected = None
    elif isinstance(expected, str):
        expected = int(expected)
    try:
        predicted = json.loads(output) if isinstance(output, str) else output
    except (json.JSONDecodeError, TypeError):
        return {"pass": False, "score": 0, "reason": "invalid output"}
    if predicted == expected:
        return {"pass": True, "score": 1, "reason": "correct match"}
    return {"pass": False, "score": 0, "reason": f"got {predicted!r}, want {expected!r}"}
