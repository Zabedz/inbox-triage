"""LangGraph workflow: stub nodes and structure, real guard and routing."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from triage.graph.nodes.classify import build_classify
from triage.graph.nodes.extract import build_extract
from triage.graph.state import State
from triage.guard import Action, GuardError, GuardThresholds, decide
from triage.models.base import DecisionModel, GenerativeModel
from triage.redact import redact as redact_text

logger = logging.getLogger("triage.graph")

RETRYABLE = {"classify", "extract", "match", "propose_action"}


def sender_domain(from_: str) -> str:
    if "@" in from_:
        return from_.rsplit("@", 1)[1].strip("> ")
    return from_


def fetch_context(state: State) -> dict[str, Any]:
    email = state["email"]
    return {
        "thread_history": [],
        "sender_domain": sender_domain(email.from_),
    }


def redact(state: State, names: Sequence[str] = ()) -> dict[str, Any]:
    email = state["email"]
    return {
        "body_text": redact_text(email.body_plain, names=names).text,
        "subject_text": redact_text(email.subject, names=names).text,
    }


def classify(state: State) -> dict[str, Any]:
    return {
        "category": "other",
        "category_probs": {"other": 1.0},
        "needs_reply": False,
        "urgency": "low",
    }


def extract(state: State) -> dict[str, Any]:
    return {"extracted": {}}


def match(state: State) -> dict[str, Any]:
    return {"candidates": [], "matched_id": None, "match_prob": 0.0}


def propose_action(state: State) -> dict[str, Any]:
    return {"action": None, "action_prob": 0.0, "draft": None}


def guard(state: State, thresholds: GuardThresholds) -> dict[str, Any]:
    action = state.get("action")
    if action is None:
        return {"guard_decision": "approval", "reason": "no action proposed"}
    try:
        resolved = Action(action)
    except ValueError as exc:
        raise GuardError(f"unknown action {action!r}") from exc
    decision = decide(
        category=state["category"],
        category_prob=state.get("category_probs", {}).get(state["category"], 0.0),
        matched_id=state.get("matched_id"),
        match_prob=state.get("match_prob", 0.0),
        action=resolved,
        thresholds=thresholds,
    )
    return {"guard_decision": decision.route, "reason": decision.reason}


def approval(state: State) -> dict[str, Any]:
    payload = {
        "email_id": state.get("email_id"),
        "category": state.get("category"),
        "category_probs": state.get("category_probs"),
        "matched_id": state.get("matched_id"),
        "match_prob": state.get("match_prob"),
        "action": state.get("action"),
        "draft": state.get("draft"),
        "reason": state.get("reason"),
    }
    decision = interrupt(payload)
    if decision is None:
        return {}
    return {
        "action": decision.get("action", state.get("action")),
        "reason": decision.get("reason"),
    }


def _route_after_guard(state: State) -> str:
    return state.get("guard_decision", "approval")


def build_graph(
    nodes: dict[str, Callable[[State], dict[str, Any]]] | None = None,
    thresholds: GuardThresholds | None = None,
    checkpointer: SqliteSaver | None = None,
    redact_names: Sequence[str] = (),
    decision_model: DecisionModel | None = None,
    extract_model: GenerativeModel | None = None,
    match_node: Callable[[State], dict[str, Any]] | None = None,
    propose_action_node: Callable[[State], dict[str, Any]] | None = None,
    apply_sink: Callable[[State], None] | None = None,
    log_sink: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    thresholds = thresholds or GuardThresholds()
    overrides = nodes or {}

    def _redact(state: State) -> dict[str, Any]:
        return redact(state, names=redact_names)

    classify_node: Callable[[State], dict[str, Any]] = classify
    if decision_model is not None:
        classify_node = build_classify(decision_model)

    extract_node: Callable[[State], dict[str, Any]] = extract
    if extract_model is not None:
        extract_node = build_extract(extract_model)

    def _apply(state: State) -> dict[str, Any]:
        logger.info(
            "applying action=%s for email=%s (%s)",
            state.get("action"),
            state.get("email_id"),
            state.get("reason"),
        )
        if apply_sink is not None:
            apply_sink(state)
        return {}

    def _log(state: State) -> dict[str, Any]:
        entry = {
            "email_id": state.get("email_id"),
            "category": state.get("category"),
            "category_probs": state.get("category_probs"),
            "matched_id": state.get("matched_id"),
            "match_prob": state.get("match_prob"),
            "action": state.get("action"),
            "guard_decision": state.get("guard_decision"),
            "reason": state.get("reason"),
        }
        trail = list(state.get("trail", []))
        trail.append(entry)
        if log_sink is not None:
            log_sink(entry)
        return {"trail": trail}

    graph = StateGraph(State)

    def add(name: str, action: Callable[[State], dict[str, Any]]) -> None:
        node_action = cast(Any, action)
        if name in RETRYABLE:
            graph.add_node(name, node_action, retry_policy=RetryPolicy(max_attempts=3))
        else:
            graph.add_node(name, node_action)

    add("fetch_context", overrides.get("fetch_context", fetch_context))
    add("redact", overrides.get("redact", _redact))
    add("classify", overrides.get("classify", classify_node))
    add("extract", overrides.get("extract", extract_node))
    add("match", overrides.get("match", match_node or match))
    add("propose_action", overrides.get("propose_action", propose_action_node or propose_action))
    add("guard", overrides.get("guard", lambda s: guard(s, thresholds)))
    add("apply", overrides.get("apply", _apply))
    add("approval", overrides.get("approval", approval))
    add("log", overrides.get("log", _log))

    graph.add_edge(START, "fetch_context")
    graph.add_edge("fetch_context", "redact")
    graph.add_edge("redact", "classify")
    graph.add_edge("classify", "extract")
    graph.add_edge("extract", "match")
    graph.add_edge("match", "propose_action")
    graph.add_edge("propose_action", "guard")
    graph.add_conditional_edges(
        "guard", _route_after_guard, {"auto": "apply", "approval": "approval"}
    )
    graph.add_edge("apply", "log")
    graph.add_edge("approval", "apply")
    graph.add_edge("log", END)

    return graph.compile(checkpointer=checkpointer)
