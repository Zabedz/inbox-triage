from __future__ import annotations

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from triage.gmail.client import Email
from triage.graph.graph import build_graph
from triage.guard import GuardThresholds


def _email() -> Email:
    return Email(
        id="e1",
        thread_id="t1",
        from_="no-reply@acme.com",
        to="me@example.com",
        subject="Application update",
        date=None,
        body_plain="Thank you for applying. Unfortunately we will not be moving forward.",
        body_html="",
        snippet="",
    )


def test_graph_redacts_body() -> None:
    email = Email(
        id="e1",
        thread_id="t1",
        from_="hr@acme.com",
        to="me@example.com",
        subject="x",
        date=None,
        body_plain="Email me at john@acme.com - Zabed",
        body_html="",
        snippet="",
    )
    graph = build_graph(redact_names=["Zabed"])
    result = graph.invoke({"email_id": "e1", "email": email})
    assert "john@acme.com" not in result["body_text"]
    assert "[EMAIL_1]" in result["body_text"]
    assert "[PERSON_1]" in result["body_text"]


def test_auto_path_runs_to_log() -> None:
    def classify(state):
        return {"category": "rejection", "category_probs": {"rejection": 0.99}, "needs_reply": False, "urgency": "low"}

    def match(state):
        return {"candidates": [{"id": 12, "company": "Acme"}], "matched_id": 12, "match_prob": 0.99}

    def propose_action(state):
        return {"action": "tracker_rejected", "action_prob": 0.99, "draft": None}

    graph = build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(category=0.9, match=0.9),
    )
    result = graph.invoke({"email_id": "e1", "email": _email()})

    assert result["guard_decision"] == "auto"
    assert result["trail"][-1]["action"] == "tracker_rejected"


def test_approval_path_interrupts_and_resumes() -> None:
    def classify(state):
        return {"category": "interview_invite", "category_probs": {"interview_invite": 0.98}, "needs_reply": True, "urgency": "high"}

    def match(state):
        return {"candidates": [], "matched_id": 7, "match_prob": 0.98}

    def propose_action(state):
        return {"action": "calendar_event", "action_prob": 0.98, "draft": "Sure, Thursday works."}

    graph = build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(),
        checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)),
    )
    config = {"configurable": {"thread_id": "t-approval"}}
    result = graph.invoke({"email_id": "e1", "email": _email()}, config=config)
    assert result.get("__interrupt__")

    resumed = graph.invoke(
        Command(resume={"action": "calendar_event", "reason": "approved"}),
        config=config,
    )
    assert resumed["trail"][-1]["guard_decision"] == "approval"


def test_checkpointer_persists_across_restart(tmp_path: pytest.TempPathFactory) -> None:
    db = tmp_path / "checkpoints.db"

    def classify(state):
        return {"category": "interview_invite", "category_probs": {"interview_invite": 0.98}, "needs_reply": True, "urgency": "high"}

    def match(state):
        return {"candidates": [], "matched_id": 7, "match_prob": 0.98}

    def propose_action(state):
        return {"action": "calendar_event", "action_prob": 0.98, "draft": None}

    config = {"configurable": {"thread_id": "t-restart"}}

    conn = sqlite3.connect(db, check_same_thread=False)
    graph = build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(),
        checkpointer=SqliteSaver(conn),
    )
    result = graph.invoke({"email_id": "e1", "email": _email()}, config=config)
    assert result.get("__interrupt__")
    conn.close()

    # New connection: a "restart". Resume from the durable checkpoint.
    conn2 = sqlite3.connect(db, check_same_thread=False)
    graph2 = build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(),
        checkpointer=SqliteSaver(conn2),
    )
    resumed = graph2.invoke(
        Command(resume={"action": "calendar_event", "reason": "approved"}),
        config=config,
    )
    conn2.close()
    assert resumed["trail"][-1]["guard_decision"] == "approval"
