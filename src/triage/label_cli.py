"""Keyboard labelling CLI: label emails against a tracker snapshot."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from triage.gmail.client import Email
from triage.guard import Action
from triage.labels import CATEGORIES, DEFAULT_ACTION, Category, ExtractedFields, LabeledEmail
from triage.tracker.db import TrackerDB

FIELDS = list(ExtractedFields.model_fields)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def load_emails(path: str | Path) -> list[Email]:
    emails: list[Email] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            emails.append(Email.model_validate(json.loads(line)))
    return emails


def _read_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    ids: set[str] = set()
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line).get("email_id"))
        except json.JSONDecodeError:
            continue
    return ids


def _prompt_category(input_fn: InputFn, print_fn: OutputFn) -> Category | None:
    while True:
        raw = input_fn("Category: ").strip().lower()
        if raw in {"q", "quit"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(CATEGORIES):
            return CATEGORIES[int(raw) - 1]
        if raw in CATEGORIES:
            return raw
        if raw in {"", "o", "other"}:
            return "other"
        print_fn(f"unrecognised category {raw!r}")


def _label_one(
    email: Email, tracker: TrackerDB, input_fn: InputFn, print_fn: OutputFn
) -> LabeledEmail | None:
    print_fn(f"\n=== {email.id} ===")
    print_fn(f"From: {email.from_}")
    print_fn(f"Subject: {email.subject}")
    print_fn(f"Date: {email.date}")
    print_fn(email.body_plain[:2000])
    print_fn("")
    for idx, cat in enumerate(CATEGORIES, 1):
        print_fn(f"  {idx}) {cat}")

    category = _prompt_category(input_fn, print_fn)
    if category is None:
        return None

    extracted: dict[str, str | None] = {}
    for field in FIELDS:
        value = input_fn(f"  {field} (enter to skip): ").strip()
        extracted[field] = value or None

    matched_id: int | None = None
    while True:
        matched_id = None
        query = input_fn("  tracker search (enter for none): ").strip()
        if not query:
            break
        results = tracker.search(query)
        if not results:
            print_fn("  no matches")
            continue
        for row in results:
            print_fn(f"    {row.id}) {row.company} - {row.role} [{row.status}]")
        sel = input_fn("  id or n: ").strip()
        if sel == "n":
            break
        try:
            matched_id = int(sel)
        except ValueError:
            print_fn(f"  {sel!r} is not a tracker id")
            continue
        if matched_id in {r.id for r in results}:
            break
        print_fn(f"  {matched_id} not in results")
        matched_id = None

    default = DEFAULT_ACTION[category]
    raw_action = input_fn(f"  action [{default.value if default else 'none'}]: ").strip()
    action: Action | None = default
    if raw_action:
        try:
            action = Action(raw_action)
        except ValueError:
            print_fn(f"  unknown action {raw_action!r}; using default")

    return LabeledEmail(
        email_id=email.id,
        from_=email.from_,
        subject=email.subject,
        body=email.body_plain,
        category=category,
        extracted=ExtractedFields(**extracted),
        matched_tracker_id=matched_id,
        action=action,
        date=email.date.date().isoformat() if email.date else None,
    )


def label(
    emails: list[Email],
    tracker: TrackerDB,
    out: str | Path,
    input_fn: InputFn = input,
    print_fn: OutputFn = print,
) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done(out)
    count = 0
    for email in emails:
        if email.id in done:
            continue
        record = _label_one(email, tracker, input_fn, print_fn)
        if record is None:
            break
        with out.open("a") as fh:
            fh.write(record.model_dump_json() + "\n")
        done.add(email.id)
        count += 1
    os.chmod(out, 0o600)
    return count
