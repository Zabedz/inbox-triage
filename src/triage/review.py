"""`triage review`: approve, edit or reject pending actions, then resume the graph."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from langgraph.types import Command
from rich.console import Console
from rich.text import Text

from triage import review_view as view
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
from triage.review_view import printable
from triage.storage import PendingApprovals

__all__ = ["printable", "review"]

InputFn = Callable[[Text], str]

REJECT = {"action": None, "reason": "rejected"}


def _resume_value(
    choice: str, payload: dict[str, Any], ask: InputFn, console: Console
) -> dict[str, Any] | None:
    """Map a reviewer choice to a resume value, or None to leave the item pending."""
    if choice == "a":
        return {"action": payload.get("action"), "reason": "approved"}
    if choice == "r":
        return dict(REJECT)
    if choice == "e":
        allowed = sorted(a.value for a in ALLOWED.get(payload.get("category") or "", set()))
        console.print(view.edit_options(allowed))
        answer = ask(view.edit_prompt()).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(allowed):
            answer = allowed[int(answer) - 1]
        if not answer:
            return dict(REJECT)
        if answer not in allowed:
            console.print(
                view.safe(
                    f"action {answer!r} not allowed for {payload.get('category')!r}; skipped"
                ),
                style="red",
            )
            return None
        return {"action": answer, "reason": "edited"}
    if choice != "s":
        console.print(view.safe(f"unrecognised choice {choice!r}; skipped"), style="red")
    return None


def _open(graph: Any, item: dict[str, Any], pending: PendingApprovals, console: Console) -> Any:
    """The item's checkpoint snapshot, or None when there is nothing left to resume."""
    snapshot = graph.get_state({"configurable": {"thread_id": item["thread_id"]}})
    if not snapshot.values:
        console.print(
            f"{item['email_id']}: no checkpoint for {item['thread_id']}; skipped", style="red"
        )
        return None
    if not snapshot.next:
        pending.remove(item["thread_id"])
        console.print(f"{item['email_id']}: already finished; removed", style="bright_black")
        return None
    return snapshot


def _finish(
    graph: Any, item: dict[str, Any], command: Any, pending: PendingApprovals, console: Console
) -> dict[str, Any] | None:
    """Resume the graph; return its final state, or None when applying failed."""
    thread_id = item["thread_id"]
    try:
        final: dict[str, Any] = graph.invoke(
            command, config={"configurable": {"thread_id": thread_id}}
        )
    except EMAIL_ERRORS as exc:
        console.print(
            view.safe(f"{item['email_id']}: not applied ({type(exc).__name__}: {exc})"), style="red"
        )
        return None
    applied = final.get("reviewed_action")
    if applied and not has_effect(Action(applied), final.get("matched_id"), final.get("category")):
        console.print(
            f"{item['email_id']}: {applied} has no apply path yet; logged only", style="yellow"
        )
    pending.remove(thread_id)
    if graph.checkpointer is not None:
        graph.checkpointer.delete_thread(thread_id)
    return final


def _bulk_reject(
    graph: Any,
    items: list[dict[str, Any]],
    pending: PendingApprovals,
    ask: InputFn,
    console: Console,
    tally: Counter[str],
) -> set[str]:
    """Offer to reject every item with no proposed action; return the threads resolved."""
    idle = [i for i in items if not i["payload"].get("action")]
    if not idle:
        return set()
    console.print(view.no_action_list(idle))
    if ask(view.bulk_prompt(len(idle))).strip().lower() != "y":
        return set()
    done: set[str] = set()
    for item in idle:
        snapshot = _open(graph, item, pending, console)
        if snapshot is None or not snapshot.interrupts:
            continue
        if _finish(graph, item, Command(resume=dict(REJECT)), pending, console) is None:
            tally["failed"] += 1
            continue
        done.add(item["thread_id"])
        tally["rejected"] += 1
    console.print(Text(f"Rejected {len(done)} emails with nothing proposed.", style="yellow"))
    return done


def review(
    cfg: Config,
    graph: Any = None,
    pending: PendingApprovals | None = None,
    input_fn: InputFn | None = None,
    console: Console | None = None,
) -> int:
    """Prompt for each pending approval; return how many were resolved."""
    console = console or Console()
    ask = input_fn or console.input
    private_dir(cfg.tracker.data_dir)
    pending_store = pending or PendingApprovals(cfg.agent_db)
    items = pending_store.list()
    if not items:
        console.print("Nothing is waiting for review.")
        return 0
    if graph is None:
        graph = production_graph(
            cfg,
            build_service(cfg, interactive=False),
            open_decision_log(cfg),
            open_checkpointer(cfg.checkpoints_db),
        )

    tally: Counter[str] = Counter()
    console.print(view.queue_summary(items))
    done = _bulk_reject(graph, items, pending_store, ask, console, tally)
    queue = [i for i in items if i["thread_id"] not in done]

    for index, item in enumerate(queue, 1):
        snapshot = _open(graph, item, pending_store, console)
        if snapshot is None:
            continue
        payload = item["payload"]
        if snapshot.interrupts:
            console.print(view.card(index, len(queue), payload, snapshot.values))
            choice = ask(view.CHOICES).strip().lower()
            if choice == "q":
                break
            resume = _resume_value(choice, payload, ask, console)
            if resume is None:
                tally["skipped"] += 1
                continue
            command: Any = Command(resume=resume)
            verdict = resume["reason"] if resume["action"] else "rejected"
        else:
            # An earlier approval failed mid-apply; only finishing it is safe.
            action = snapshot.values.get("reviewed_action")
            console.print(
                f"{item['email_id']}: approved {action!r} did not finish applying", style="yellow"
            )
            if ask(Text("Retry apply? [y/N] > ")).strip().lower() != "y":
                tally["skipped"] += 1
                continue
            command, verdict = None, "approved"

        final = _finish(graph, item, command, pending_store, console)
        if final is None:
            tally["failed"] += 1
            continue
        tally[verdict] += 1
        console.print(view.outcome(verdict, final.get("reviewed_action"), final.get("matched_id")))

    console.print(view.session_summary(tally, len(pending_store.list())))
    return sum(tally[k] for k in ("approved", "edited", "rejected"))
