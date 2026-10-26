"""Terminal rendering for `triage review`.

Email-derived text only ever enters a renderable as ``Text``, so it is never parsed as
rich markup.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triage.agent import has_effect
from triage.guard import Action

CATEGORY_STYLES: dict[str, str] = {
    "rejection": "red",
    "application_ack": "blue",
    "interview_invite": "green",
    "interview_reschedule": "green",
    "assessment": "magenta",
    "recruiter_outreach": "cyan",
    "offer": "bold green",
    "other": "bright_black",
}

ACTION_LABELS: dict[Action, str] = {
    Action.LABEL: "Label in Gmail",
    Action.ACK: "Note the acknowledgement",
    Action.TRACKER_REJECTED: "Mark as Rejected",
    Action.CALENDAR_EVENT: "Add a calendar event",
    Action.CALENDAR_CHANGE: "Move the calendar event",
    Action.TRACKER_INTERVIEW: "Mark as Interview",
    Action.TRACKER_OFFER: "Mark as Offer",
    Action.REPLY_DRAFT: "Draft a reply",
}

DETAIL_LABELS: dict[str, str] = {
    "role": "Role",
    "date_time": "When",
    "timezone": "Timezone",
    "deadline": "Deadline",
    "contact": "Contact",
    "agency": "Agency",
    "requisition_id": "Req ID",
}

CONFIDENT = 0.95
UNSURE = 0.75
BAR_WIDTH = 10
RUNNER_UP_MIN = 0.05
EXCERPT_LINES = 12
SUBJECT_WIDTH = 60

CHOICES = Text.assemble(
    ("a", "bold green"),
    " approve   ",
    ("e", "bold cyan"),
    " edit   ",
    ("r", "bold red"),
    " reject   ",
    ("s", "bold"),
    " skip   ",
    ("q", "bold"),
    " quit  ",
    ("> ", "bold"),
)


def printable(value: object) -> str:
    """Escape control and bidi characters so email-derived text cannot drive the terminal."""
    return "".join(c if c in "\n\t" or c.isprintable() else f"\\x{ord(c):02x}" for c in str(value))


def _plain(value: object) -> str:
    """Normalise line endings and non-breaking spaces, then escape the rest."""
    text = "" if value is None else str(value)
    return printable(text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " "))


def safe(value: object, style: str = "") -> Text:
    return Text(_plain(value), style=style)


def confidence(prob: float | None) -> Text:
    p = prob or 0.0
    style = "green" if p >= CONFIDENT else "yellow" if p >= UNSURE else "red"
    filled = round(p * BAR_WIDTH)
    return Text.assemble(
        ("#" * filled, style), ("." * (BAR_WIDTH - filled), "bright_black"), (f" {p:4.0%}", style)
    )


def category_text(category: str | None) -> Text:
    name = category or "unknown"
    return Text(name.replace("_", " "), style=CATEGORY_STYLES.get(name, "bold"))


def action_label(action: str | None) -> str:
    if not action:
        return "Nothing proposed"
    try:
        return ACTION_LABELS[Action(action)]
    except ValueError:
        return action


def _effect_note(action: str | None, matched_id: int | None, category: str | None) -> Text | None:
    if not action:
        return None
    try:
        parsed = Action(action)
    except ValueError:
        return None
    if has_effect(parsed, matched_id, category):
        return None
    if parsed in (Action.TRACKER_REJECTED, Action.TRACKER_INTERVIEW, Action.TRACKER_OFFER):
        return Text("No tracker row matched, so approving changes nothing.", style="yellow")
    return Text(
        "Approving only logs this; calendar and drafts are not connected yet.", style="yellow"
    )


def queue_summary(items: Sequence[Mapping[str, Any]]) -> Table:
    counts = Counter(
        (
            i["payload"].get("action"),
            i["payload"].get("category"),
            _effect_note(
                i["payload"].get("action"), i["payload"].get("matched_id"), i["payload"].get("category")
            )
            is None,
        )
        for i in items
    )
    table = Table(
        title=f"{len(items)} emails waiting for review",
        title_style="bold",
        caption="Tracker changes are written to data/tracker.db.",
        caption_style="bright_black",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Proposed action")
    table.add_column("Kind of email")
    table.add_column("Emails", justify="right")
    table.add_column("On approval")
    for (action, category, applies), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if not action:
            effect = Text("nothing to apply", style="bright_black")
        elif applies:
            effect = Text("applies", style="green")
        else:
            effect = Text("logged only", style="yellow")
        table.add_row(action_label(action), category_text(category), str(n), effect)
    return table


def no_action_list(items: Sequence[Mapping[str, Any]]) -> Table:
    table = Table(
        title=f"{len(items)} emails with nothing proposed",
        title_style="bold",
        box=box.SIMPLE,
        show_edge=False,
    )
    table.add_column("From", style="bright_black", no_wrap=True)
    table.add_column("Subject", max_width=SUBJECT_WIDTH, no_wrap=True, overflow="ellipsis")
    table.add_column("Kind")
    for item in items:
        payload = item["payload"]
        table.add_row(
            safe(payload.get("sender_domain")),
            safe(payload.get("subject")),
            category_text(payload.get("category")),
        )
    return table


def bulk_prompt(n: int) -> Text:
    return Text.assemble(f"Reject all {n} of these in one go? ", ("[y/N] ", "bold"), "> ")


def _runner_ups(probs: Mapping[str, float], chosen: str | None) -> Text | None:
    others = sorted(
        ((c, p) for c, p in probs.items() if c != chosen and p >= RUNNER_UP_MIN),
        key=lambda cp: -cp[1],
    )
    if not others:
        return None
    parts: list[Text | str] = []
    for c, p in others:
        if parts:
            parts.append("   ")
        parts.extend([category_text(c), Text(f" {p:.0%}", style="bright_black")])
    return Text.assemble(*parts)


def _excerpt(body: str | None) -> Text | None:
    if not body:
        return None
    lines: list[str] = []
    for line in _plain(body).splitlines():
        stripped = line.rstrip()
        if not stripped and (not lines or not lines[-1]):
            continue
        lines.append(stripped)
    while lines and not lines[-1]:
        lines.pop()
    hidden = len(lines) - EXCERPT_LINES
    text = Text("\n".join(lines[:EXCERPT_LINES]))
    if hidden > 0:
        text.append(f"\n... {hidden} more lines", style="bright_black italic")
    return text


def card(index: int, total: int, payload: Mapping[str, Any], values: Mapping[str, Any]) -> Panel:
    category = payload.get("category")
    action = payload.get("action")
    matched_id = payload.get("matched_id")

    # Label, confidence, value: confidences line up in one column whatever the value width.
    fields = Table.grid(padding=(0, 2))
    fields.add_column(style="bold bright_black", no_wrap=True)
    fields.add_column(no_wrap=True)
    fields.add_column()

    fields.add_row(
        "Kind", confidence(payload.get("category_probs", {}).get(category)), category_text(category)
    )
    also = _runner_ups(payload.get("category_probs") or {}, category)
    if also is not None:
        fields.add_row("Also", "", also)
    if matched_id is None:
        fields.add_row("Tracker", "", Text("no matching row", style="bright_black"))
    else:
        row = Text.assemble((f"#{matched_id} ", "bold"), safe(payload.get("matched_row")))
        fields.add_row("Tracker", confidence(payload.get("match_prob")), row)
    fields.add_row(
        "Proposed",
        confidence(payload.get("action_prob")) if action else "",
        Text(action_label(action), style="bold" if action else "bright_black"),
    )
    fields.add_row("Held", "", safe(payload.get("reason"), style="yellow"))
    note = _effect_note(action, matched_id, category)
    if note is not None:
        fields.add_row("", "", note)

    reply = values.get("needs_reply")
    if reply is not None or values.get("urgency"):
        fields.add_row(
            "Reply",
            "",
            Text.assemble(
                ("needed" if reply else "not needed", "bold" if reply else ""),
                ("   urgency ", "bright_black"),
                safe(values.get("urgency") or "unknown"),
            ),
        )
    extracted = values.get("extracted") or {}
    for key, label in DETAIL_LABELS.items():
        if extracted.get(key):
            fields.add_row(label, "", safe(extracted[key]))

    parts: list[RenderableType] = [
        safe(payload.get("subject") or "(no subject)", style="bold"),
        Text.assemble(("from ", "bright_black"), safe(payload.get("sender_domain"))),
        Text(""),
        fields,
    ]
    excerpt = _excerpt(values.get("body_text"))
    if excerpt is None:
        parts.append(Text(""))
        parts.append(
            Text("No plain-text body; the models saw only the subject.", style="bright_black")
        )
    else:
        parts.append(Text(""))
        parts.append(
            Panel(
                excerpt, title="Email (redacted)", title_align="left", border_style="bright_black"
            )
        )
    if payload.get("draft"):
        parts.append(Text(""))
        parts.append(
            Panel(
                safe(payload["draft"]), title="Draft reply", title_align="left", border_style="cyan"
            )
        )

    return Panel(
        Group(*parts),
        title=Text.assemble((f" {index} of {total} ", "bold"), " ", category_text(category), " "),
        title_align="left",
        border_style=CATEGORY_STYLES.get(category or "", "white"),
        padding=(1, 2),
    )


def edit_options(allowed: Sequence[str]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_column(style="bright_black")
    for n, name in enumerate(allowed, 1):
        table.add_row(str(n), action_label(name), name)
    return table


def edit_prompt() -> Text:
    return Text.assemble(
        "Number or name of the new action ", ("(blank to reject) ", "bright_black"), "> "
    )


def outcome(verdict: str, action: str | None, matched_id: int | None) -> Text:
    if verdict == "rejected" or not action:
        return Text("Rejected. Nothing applied.", style="yellow")
    target = f" on #{matched_id}" if matched_id is not None else ""
    verb = "Edited" if verdict == "edited" else "Approved"
    return Text(f"{verb}: {action_label(action)}{target}.", style="bold green")


def session_summary(tally: Counter[str], left: int) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY, title="Review finished", title_style="bold", show_header=False
    )
    table.add_column(style="bold")
    table.add_column(justify="right")
    for key, style in (
        ("approved", "green"),
        ("edited", "cyan"),
        ("rejected", "yellow"),
        ("skipped", ""),
        ("failed", "red"),
    ):
        if tally.get(key):
            table.add_row(Text(key.capitalize(), style=style), str(tally[key]))
    table.add_row(Text("Still waiting", style="bright_black"), str(left))
    return table
