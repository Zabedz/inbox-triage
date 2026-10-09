from __future__ import annotations

import sqlite3

import pytest
from conftest import FakeClient, make_cfg
from langgraph.types import Command
from test_run import recording_sink

from triage.agent import open_checkpointer
from triage.graph.graph import build_graph
from triage.guard import GuardThresholds
from triage.review import printable, review
from triage.run import run
from triage.storage import DecisionLog, PendingApprovals


def interview_graph(db, sink, log: DecisionLog):
    def classify(state):
        return {"category": "interview_invite", "category_probs": {"interview_invite": 0.98}}

    def match(state):
        candidates = [{"id": 7, "company": "Initech", "role": "Data Scientist"}]
        return {"candidates": candidates, "matched_id": 7, "match_prob": 0.98}

    def propose_action(state):
        return {"action": "calendar_event", "action_prob": 0.97, "draft": "Thursday works."}

    return build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(),
        checkpointer=open_checkpointer(db),
        apply_sink=sink,
        log_sink=log.log,
    )


def setup_pending(tmp_path, ids: list[str]):
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []
    graph = interview_graph(cfg.checkpoints_db, recording_sink(applied), log)
    run(cfg, graph=graph, client=FakeClient(ids), pending=pending, decision_log=log)
    assert len(pending.list()) == len(ids)
    # A fresh graph over the same checkpoint file stands in for a new process.
    return cfg, log, pending, applied, interview_graph(cfg.checkpoints_db, recording_sink(applied), log)


def scripted(*answers: str):
    queue = list(answers)
    return lambda prompt: queue.pop(0)


def quiet(line: str) -> None:
    return None


def decision_row(cfg) -> tuple:
    conn = sqlite3.connect(cfg.agent_db)
    row = conn.execute("SELECT action, reason, reviewed_action, verdict FROM decisions").fetchone()
    conn.close()
    return row


def test_approve_resumes_after_restart(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("a"), print_fn=quiet)

    assert handled == 1
    assert pending.list() == []
    assert applied == [("m1", "calendar_event")]
    assert decision_row(cfg) == (
        "calendar_event", "action 'calendar_event' requires approval", "calendar_event", "approved"
    )
    assert not graph.get_state({"configurable": {"thread_id": "email-m1"}}).values


def test_review_shows_context_and_probabilities(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])
    printed: list[str] = []

    review(cfg, graph=graph, pending=pending, input_fn=scripted("s"), print_fn=printed.append)

    text = "\n".join(printed)
    assert "acme.com" in text
    assert "Initech - Data Scientist" in text
    assert "0.98" in text
    assert "action_prob: 0.97" in text


def test_reject_applies_nothing_and_keeps_proposal(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    review(cfg, graph=graph, pending=pending, input_fn=scripted("r"), print_fn=quiet)

    assert pending.list() == []
    assert applied == []
    assert decision_row(cfg)[0] == "calendar_event"
    assert decision_row(cfg)[2:] == (None, "rejected")


def test_edit_to_allowed_action(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    review(cfg, graph=graph, pending=pending, input_fn=scripted("e", "tracker_interview"), print_fn=quiet)

    assert pending.list() == []
    assert applied == [("m1", "tracker_interview")]


def test_edit_to_disallowed_action_stays_pending(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])
    printed: list[str] = []

    handled = review(
        cfg, graph=graph, pending=pending, input_fn=scripted("e", "tracker_offer"), print_fn=printed.append
    )

    assert handled == 0
    assert len(pending.list()) == 1
    assert applied == []
    assert any("not allowed" in line for line in printed)


def test_direct_resume_with_disallowed_action_is_refused(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])
    config = {"configurable": {"thread_id": "email-m1"}}

    graph.invoke(Command(resume={"action": "tracker_offer", "reason": "x"}), config=config)

    assert applied == []
    assert decision_row(cfg)[2:] == (None, "refused 'tracker_offer'")


@pytest.mark.parametrize("resume", ["yes", {}, {"reason": "approved"}])
def test_resume_without_action_applies_nothing(tmp_path, resume) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    graph.invoke(Command(resume=resume), config={"configurable": {"thread_id": "email-m1"}})

    assert applied == []


def test_reject_after_failed_apply_does_not_apply_earlier_approval(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []
    graph = interview_graph(cfg.checkpoints_db, recording_sink(applied), log)
    run(cfg, graph=graph, client=FakeClient(["m1"]), pending=pending, decision_log=log)

    def failing(action, state):
        raise sqlite3.OperationalError("database is locked")

    broken = interview_graph(cfg.checkpoints_db, failing, log)
    printed: list[str] = []
    review(cfg, graph=broken, pending=pending, input_fn=scripted("e", "tracker_interview"), print_fn=printed.append)
    assert any("not applied" in line for line in printed)
    assert len(pending.list()) == 1

    healthy = interview_graph(cfg.checkpoints_db, recording_sink(applied), log)
    handled = review(cfg, graph=healthy, pending=pending, input_fn=scripted("n"), print_fn=printed.append)
    assert handled == 0
    assert applied == []
    assert any("did not finish applying" in line for line in printed)

    handled = review(cfg, graph=healthy, pending=pending, input_fn=scripted("y"), print_fn=quiet)
    assert handled == 1
    assert applied == [("m1", "tracker_interview")]


def test_skip_and_unrecognised_leave_pending(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1", "m2", "m3"])
    printed: list[str] = []

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("s", "approve", "a"), print_fn=printed.append)

    assert handled == 1
    assert [item["email_id"] for item in pending.list()] == ["m1", "m2"]
    assert applied == [("m3", "calendar_event")]
    assert any("unrecognised choice 'approve'" in line for line in printed)


def test_missing_checkpoint_keeps_pending(tmp_path) -> None:
    cfg, log, pending, applied, _ = setup_pending(tmp_path, ["m1"])
    empty = interview_graph(str(tmp_path / "other.db"), recording_sink(applied), log)
    printed: list[str] = []

    assert review(cfg, graph=empty, pending=pending, input_fn=scripted(), print_fn=printed.append) == 0
    assert len(pending.list()) == 1
    assert any("no checkpoint" in line for line in printed)


def test_no_pending_builds_nothing(tmp_path) -> None:
    cfg = make_cfg(tmp_path)

    assert review(cfg, pending=PendingApprovals(cfg.agent_db), input_fn=scripted()) == 0


def test_printable_escapes_terminal_controls() -> None:
    assert printable("ok\x1b[2J\u202eevil\nline") == "ok\\x1b[2J\\x202eevil\nline"
