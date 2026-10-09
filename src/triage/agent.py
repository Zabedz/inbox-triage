"""Agent wiring shared by `triage run`, `triage review` and the end-to-end eval."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
from googleapiclient.errors import HttpError
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from triage.config import Config
from triage.gmail.labels import apply_label
from triage.graph.graph import build_graph
from triage.graph.nodes.match import build_match
from triage.graph.nodes.propose_action import build_propose_action
from triage.graph.state import State
from triage.guard import Action, GuardError, GuardThresholds
from triage.models import build_decision_model
from triage.models.base import DecisionModel, GenerativeModel
from triage.models.jev import JevAPIError, JevError
from triage.models.llm import GenerativeAPIError, GenerativeError, build_generative_model
from triage.storage import DecisionLog
from triage.tracker.db import TrackerDB
from triage.tracker.update import set_status

ApplySink = Callable[[Action, State], None]

ACTION_STATUS = {
    Action.TRACKER_REJECTED: "Rejected",
    Action.TRACKER_INTERVIEW: "Interview",
    Action.TRACKER_OFFER: "Offer",
}

LABEL_ACTIONS = {Action.LABEL, Action.ACK}

# Errors a run or review survives; is_transient splits outages from bad emails.
EMAIL_ERRORS: tuple[type[Exception], ...] = (
    JevError,
    GenerativeError,
    GuardError,
    HttpError,
    httpx.HTTPError,
    sqlite3.OperationalError,
)

# Auth and rate-limit rejections hit every email alike.
SYSTEMIC_STATUSES = {401, 403, 429}


def is_transient(exc: Exception) -> bool:
    """True for outages and auth or rate limits, which say nothing about the email."""
    if isinstance(exc, HttpError):
        status = int(exc.resp.status)
        return status >= 500 or status in SYSTEMIC_STATUSES
    if isinstance(exc, (JevAPIError, GenerativeAPIError)):
        return exc.status in SYSTEMIC_STATUSES
    return not isinstance(exc, GuardError)


def has_effect(action: Action, matched_id: int | None) -> bool:
    """Whether applying ``action`` changes anything yet (calendar and drafts are log-only)."""
    return action in LABEL_ACTIONS or (action in ACTION_STATUS and matched_id is not None)


def make_apply_sink(label: Callable[[str], None], tracker_db: Path) -> ApplySink:
    def apply_sink(action: Action, state: State) -> None:
        if action in LABEL_ACTIONS:
            label(state.get("email_id", ""))
        matched_id = state.get("matched_id")
        if action in ACTION_STATUS and matched_id is not None:
            set_status(tracker_db, matched_id, ACTION_STATUS[action])

    return apply_sink


def assemble(
    *,
    decision: DecisionModel,
    generative: GenerativeModel | None,
    tracker: TrackerDB,
    thresholds: GuardThresholds,
    apply_sink: ApplySink | None,
    log_sink: Callable[[dict[str, Any]], None] | None,
    checkpointer: Any,
    redact_names: Sequence[str] = (),
) -> Any:
    """Compile the agent graph from models and sinks; without an LLM, extract and match stub."""
    match_node = build_match(tracker, generative, decision) if generative is not None else None
    return build_graph(
        thresholds=thresholds,
        redact_names=redact_names,
        decision_model=decision,
        extract_model=generative,
        match_node=match_node,
        propose_action_node=build_propose_action(decision, generative),
        apply_sink=apply_sink,
        log_sink=log_sink,
        checkpointer=checkpointer,
    )


def production_graph(
    cfg: Config, gmail_service: object, decision_log: DecisionLog, checkpointer: SqliteSaver
) -> Any:
    generative = build_generative_model(cfg) if cfg.llm_provider else None
    return assemble(
        decision=build_decision_model(cfg),
        generative=generative,
        tracker=TrackerDB(cfg.tracker.db),
        thresholds=GuardThresholds(
            category=cfg.guard_category_threshold, match=cfg.guard_match_threshold
        ),
        apply_sink=make_apply_sink(lambda mid: apply_label(gmail_service, mid), cfg.tracker.db),
        log_sink=decision_log.log,
        checkpointer=checkpointer,
        redact_names=cfg.candidate_names,
    )


def open_decision_log(cfg: Config) -> DecisionLog:
    model = cfg.llm_model if cfg.llm_provider else None
    return DecisionLog(
        cfg.agent_db, context={"decision_backend": cfg.decision_backend, "llm_model": model}
    )


def open_checkpointer(path: Path) -> SqliteSaver:
    conn = sqlite3.connect(path, check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("triage.gmail.client", "Email")])
    return SqliteSaver(conn, serde=serde)


def private_dir(path: Path) -> None:
    """Create ``path`` readable by the owner only; it holds email content."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
