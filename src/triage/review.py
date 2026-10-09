"""Review pending approvals: approve / edit / reject, then resume the graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.types import Command

from triage.config import Config
from triage.gmail.ingest import build_service
from triage.guard import ALLOWED
from triage.run import build_agent_graph, open_checkpointer
from triage.storage import DecisionLog, PendingApprovals

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


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
    return None


def review(
    cfg: Config,
    graph: Any = None,
    pending: PendingApprovals | None = None,
    input_fn: InputFn = input,
    print_fn: OutputFn = print,
) -> int:
    """Prompt for each pending approval; return how many were resolved."""
    pending_store = pending or PendingApprovals(cfg.agent_db)
    items = pending_store.list()
    if not items:
        return 0
    if graph is None:
        graph = build_agent_graph(
            cfg, open_checkpointer(cfg), DecisionLog(cfg.agent_db), build_service(cfg)
        )

    handled = 0
    for item in items:
        payload = item["payload"]
        print_fn(f"\n=== {item['email_id']} ===")
        print_fn(f"category: {payload.get('category')}  probs: {payload.get('category_probs')}")
        print_fn(f"tracker: {payload.get('matched_id')}  match_prob: {payload.get('match_prob')}")
        print_fn(f"action: {payload.get('action')}  reason: {payload.get('reason')}")
        if payload.get("draft"):
            print_fn(f"draft:\n{payload['draft']}")

        choice = input_fn("approve/edit/reject/skip [a/e/r/s]: ").strip().lower()
        resume = _resume_value(choice, payload, input_fn, print_fn)
        if resume is None:
            continue
        config = {"configurable": {"thread_id": item["thread_id"]}}
        graph.invoke(Command(resume=resume), config=config)
        pending_store.remove(item["thread_id"])
        handled += 1

    return handled
