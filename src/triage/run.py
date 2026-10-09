"""Composition root: build the agent graph and run it over new email."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from triage.config import Config
from triage.gmail.client import GmailClient
from triage.gmail.ingest import build_service
from triage.gmail.labels import apply_label
from triage.graph.graph import build_graph
from triage.graph.nodes.match import build_match
from triage.graph.nodes.propose_action import build_propose_action
from triage.graph.state import State
from triage.guard import Action, GuardThresholds
from triage.models import build_decision_model
from triage.models.llm import build_generative_model
from triage.storage import DecisionLog, PendingApprovals
from triage.tracker.db import TrackerDB
from triage.tracker.update import set_status

logger = logging.getLogger("triage.run")

ACTION_STATUS = {
    Action.TRACKER_REJECTED: "Rejected",
    Action.TRACKER_INTERVIEW: "Interview",
    Action.TRACKER_OFFER: "Offer",
}

LABEL_ACTIONS = {Action.LABEL, Action.ACK}


def make_apply_sink(cfg: Config, gmail_service: object) -> Callable[[State], None]:
    def apply_sink(state: State) -> None:
        action = state.get("action")
        if action is None:
            return
        if action in LABEL_ACTIONS:
            apply_label(gmail_service, state.get("email_id", ""))
        matched_id = state.get("matched_id")
        if action in ACTION_STATUS and matched_id is not None:
            set_status(cfg.tracker.db, matched_id, ACTION_STATUS[Action(action)])

    return apply_sink


def open_checkpointer(cfg: Config) -> SqliteSaver:
    conn = sqlite3.connect(cfg.checkpoints_db, check_same_thread=False)
    return SqliteSaver(conn)


def build_agent_graph(
    cfg: Config, checkpointer: SqliteSaver, decision_log: DecisionLog, gmail_service: object
) -> Any:
    decision = build_decision_model(cfg)
    generative = build_generative_model(cfg) if cfg.llm_provider else None
    match_node = None
    if generative is not None:
        match_node = build_match(TrackerDB(cfg.tracker.db), generative, decision)

    return build_graph(
        thresholds=GuardThresholds(
            category=cfg.guard_category_threshold, match=cfg.guard_match_threshold
        ),
        redact_names=cfg.candidate_names,
        decision_model=decision,
        extract_model=generative,
        match_node=match_node,
        propose_action_node=build_propose_action(decision, generative),
        apply_sink=make_apply_sink(cfg, gmail_service),
        log_sink=decision_log.log,
        checkpointer=checkpointer,
    )


def run(
    cfg: Config,
    graph: Any = None,
    client: GmailClient | None = None,
    pending: PendingApprovals | None = None,
    decision_log: DecisionLog | None = None,
) -> list[str]:
    """Triage each new email; return the thread ids left awaiting approval.

    An email already in the decision log or pending approvals is skipped, so the
    schedule can rerun the same Gmail query safely.
    """
    pending_store = pending or PendingApprovals(cfg.agent_db)
    log_store = decision_log or DecisionLog(cfg.agent_db)
    if graph is None or client is None:
        service = build_service(cfg)
        client = client or GmailClient(service)
        graph = graph or build_agent_graph(cfg, open_checkpointer(cfg), log_store, service)

    pending_threads: list[str] = []
    for message_id in client.list_job_messages(cfg.gmail.query):
        if log_store.has(message_id) or pending_store.has(message_id):
            continue
        email = client.get_email(message_id)
        thread_id = f"email-{message_id}"
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke({"email_id": message_id, "email": email}, config=config)
        for interrupt in result.get("__interrupt__") or []:
            pending_store.add(thread_id, message_id, interrupt.value)
            pending_threads.append(thread_id)

    logger.info("run complete: %d awaiting approval", len(pending_threads))
    return pending_threads
