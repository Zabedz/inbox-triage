"""`triage review`: approve, edit or reject pending actions, then resume the graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.types import Command

from triage.agent import (
    EMAIL_ERRORS,
    has_effect,
    open_checkpointer,
    open_decision_log,
    private_dir,
    production_graph,
)
from triage.config import Config
from triage.gmail.ingest import build_service
from triage.guard import ALLOWED, Action
from triage.storage import PendingApprovals

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def printable(value: object) -> str:
    """Escape control and bidi characters so email-derived text cannot drive the terminal."""
    return "".join(c if c in "\n\t" or c.isprintable() else f"\\x{ord(c):02x}" for c in str(value))


def _show(payload: dict[str, Any], email_id: str, print_fn: OutputFn) -> None:
    lines = [
        f"\n=== {email_id} ===",
        f"from: {payload.get('sender_domain')}  subject: {payload.get('subject')}",
        f"category: {payload.get('category')}  probs: {payload.get('category_probs')}",
        f"tracker: {payload.get('matched_id')} {payload.get('matched_row') or ''}  "
        f"match_prob: {payload.get('match_prob')}",
        f"action: {payload.get('action')}  action_prob: {payload.get('action_prob')}  "
        f"reason: {payload.get('reason')}",
    ]
    if payload.get("draft"):
        lines.append(f"draft:\n{payload['draft']}")
    for line in lines:
        print_fn(printable(line))


def _resume_value(
    choice: str, payload: dict[str, Any], input_fn: InputFn, print_fn: OutputFn
) -> dict[str, Any] | None:
    """Map a reviewer choice to a resume value, or None to leave the item pending."""
    if choice == "a":
        return {"action": payload.get("action"), "reason": "approved"}
    if choice == "r":
        return {"action": None, "reason": "rejected"}
    if choice == "e":
        allowed = sorted(a.value for a in ALLOWED.get(payload.get("category") or "", set()))
        new_action = input_fn(f"new action {allowed} (blank to reject): ").strip() or None
        if new_action is not None and new_action not in allowed:
            print_fn(f"action {new_action!r} not allowed for {payload.get('category')!r}; skipped")
            return None
        return {"action": new_action, "reason": "edited"}
    if choice != "s":
        print_fn(f"unrecognised choice {choice!r}; skipped")
    return None


def review(
    cfg: Config,
    graph: Any = None,
    pending: PendingApprovals | None = None,
    input_fn: InputFn = input,
    print_fn: OutputFn = print,
) -> int:
    """Prompt for each pending approval; return how many were resolved."""
    private_dir(cfg.tracker.data_dir)
    pending_store = pending or PendingApprovals(cfg.agent_db)
    items = pending_store.list()
    if not items:
        return 0
    if graph is None:
        graph = production_graph(
            cfg,
            build_service(cfg, interactive=False),
            open_decision_log(cfg),
            open_checkpointer(cfg.checkpoints_db),
        )

    handled = 0
    for item in items:
        thread_id, email_id = item["thread_id"], item["email_id"]
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        if not snapshot.values:
            print_fn(f"{email_id}: no checkpoint for {thread_id}; skipped")
            continue
        if not snapshot.next:
            pending_store.remove(thread_id)
            print_fn(f"{email_id}: already finished; removed")
            continue

        if snapshot.interrupts:
            payload = item["payload"]
            _show(payload, email_id, print_fn)
            choice = input_fn("approve/edit/reject/skip [a/e/r/s]: ").strip().lower()
            resume = _resume_value(choice, payload, input_fn, print_fn)
            if resume is None:
                continue
            command: Any = Command(resume=resume)
        else:
            # An earlier approval failed mid-apply; only finishing it is safe.
            action = snapshot.values.get("reviewed_action")
            print_fn(f"{email_id}: approved {action!r} did not finish applying")
            if input_fn("retry apply? [y/N]: ").strip().lower() != "y":
                continue
            command = None

        try:
            final = graph.invoke(command, config=config)
        except EMAIL_ERRORS as exc:
            print_fn(f"{email_id}: not applied ({type(exc).__name__}: {printable(exc)})")
            continue
        applied = final.get("reviewed_action")
        if applied and not has_effect(Action(applied), final.get("matched_id")):
            print_fn(f"{email_id}: {applied} has no apply path yet; logged only")
        pending_store.remove(thread_id)
        if graph.checkpointer is not None:
            graph.checkpointer.delete_thread(thread_id)
        handled += 1

    return handled
