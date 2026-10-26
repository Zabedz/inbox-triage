from __future__ import annotations

import io
import sqlite3

import pytest
from conftest import FakeClient, make_cfg
from langgraph.types import Command
from rich.console import Console
from test_run import recording_sink

from triage.agent import open_checkpointer
from triage.graph.graph import build_graph
from triage.guard import GuardThresholds
from triage.review import printable, review
from triage.review_view import card
from triage.run import run
from triage.storage import DecisionLog, PendingApprovals


def interview_graph(db, sink, log: DecisionLog, action: str | None = "calendar_event"):
    def classify(state):
        return {"category": "interview_invite", "category_probs": {"interview_invite": 0.98}}

    def match(state):
        candidates = [{"id": 7, "company": "Initech", "role": "Data Scientist"}]
        return {"candidates": candidates, "matched_id": 7, "match_prob": 0.98}

    def propose_action(state):
        return {"action": action, "action_prob": 0.97, "draft": "Thursday works."}

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


def capture() -> Console:
    return Console(file=io.StringIO(), width=120, color_system=None)


def output(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[attr-defined]


def decision_row(cfg) -> tuple:
    conn = sqlite3.connect(cfg.agent_db)
    row = conn.execute("SELECT action, reason, reviewed_action, verdict FROM decisions").fetchone()
    conn.close()
    return row


def test_approve_resumes_after_restart(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("a"), console=capture())

    assert handled == 1
    assert pending.list() == []
    assert applied == [("m1", "calendar_event")]
    assert decision_row(cfg) == (
        "calendar_event", "action 'calendar_event' requires approval", "calendar_event", "approved"
    )
    assert not graph.get_state({"configurable": {"thread_id": "email-m1"}}).values


def test_review_shows_context_and_probabilities(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])
    out = capture()

    review(cfg, graph=graph, pending=pending, input_fn=scripted("s"), console=out)

    text = output(out)
    assert "acme.com" in text
    assert "#7 Initech - Data Scientist" in text
    assert "Add a calendar event" in text
    assert "98%" in text
    assert "97%" in text
    assert "Thursday works." in text


def test_reject_applies_nothing_and_keeps_proposal(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    review(cfg, graph=graph, pending=pending, input_fn=scripted("r"), console=capture())

    assert pending.list() == []
    assert applied == []
    assert decision_row(cfg)[0] == "calendar_event"
    assert decision_row(cfg)[2:] == (None, "rejected")


def test_edit_to_allowed_action(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    review(cfg, graph=graph, pending=pending, input_fn=scripted("e", "tracker_interview"), console=capture())

    assert pending.list() == []
    assert applied == [("m1", "tracker_interview")]


def test_edit_to_disallowed_action_stays_pending(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])
    out = capture()

    handled = review(
        cfg, graph=graph, pending=pending, input_fn=scripted("e", "tracker_offer"), console=out
    )

    assert handled == 0
    assert len(pending.list()) == 1
    assert applied == []
    assert "not allowed" in output(out)


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
    out = capture()
    review(cfg, graph=broken, pending=pending, input_fn=scripted("e", "tracker_interview"), console=out)
    assert "not applied" in output(out)
    assert len(pending.list()) == 1

    healthy = interview_graph(cfg.checkpoints_db, recording_sink(applied), log)
    handled = review(cfg, graph=healthy, pending=pending, input_fn=scripted("n"), console=out)
    assert handled == 0
    assert applied == []
    assert "did not finish applying" in output(out)

    handled = review(cfg, graph=healthy, pending=pending, input_fn=scripted("y"), console=capture())
    assert handled == 1
    assert applied == [("m1", "tracker_interview")]


def test_skip_and_unrecognised_leave_pending(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1", "m2", "m3"])
    out = capture()

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("s", "approve", "a"), console=out)

    assert handled == 1
    assert [item["email_id"] for item in pending.list()] == ["m1", "m2"]
    assert applied == [("m3", "calendar_event")]
    assert "unrecognised choice 'approve'" in output(out)


def test_missing_checkpoint_keeps_pending(tmp_path) -> None:
    cfg, log, pending, applied, _ = setup_pending(tmp_path, ["m1"])
    empty = interview_graph(str(tmp_path / "other.db"), recording_sink(applied), log)
    out = capture()

    assert review(cfg, graph=empty, pending=pending, input_fn=scripted(), console=out) == 0
    assert len(pending.list()) == 1
    assert "no checkpoint" in output(out)


def test_no_pending_builds_nothing(tmp_path) -> None:
    cfg = make_cfg(tmp_path)

    assert review(cfg, pending=PendingApprovals(cfg.agent_db), input_fn=scripted()) == 0


def test_printable_escapes_terminal_controls() -> None:
    assert printable("ok\x1b[2J\u202eevil\nline") == "ok\\x1b[2J\\x202eevil\nline"


def test_bulk_reject_clears_items_with_nothing_proposed(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []
    idle = interview_graph(cfg.checkpoints_db, recording_sink(applied), log, action=None)
    run(cfg, graph=idle, client=FakeClient(["m1", "m2"]), pending=pending, decision_log=log)
    busy = interview_graph(cfg.checkpoints_db, recording_sink(applied), log)
    run(cfg, graph=busy, client=FakeClient(["m3"]), pending=pending, decision_log=log)
    out = capture()

    handled = review(cfg, graph=busy, pending=pending, input_fn=scripted("y", "s"), console=out)

    assert handled == 2
    assert [item["email_id"] for item in pending.list()] == ["m3"]
    assert applied == []
    assert "2 emails with nothing proposed" in output(out)


def test_declining_bulk_reject_reviews_each_item(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []
    graph = interview_graph(cfg.checkpoints_db, recording_sink(applied), log, action=None)
    run(cfg, graph=graph, client=FakeClient(["m1"]), pending=pending, decision_log=log)

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("n", "s"), console=capture())

    assert handled == 0
    assert len(pending.list()) == 1


def test_edit_by_number(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1"])

    review(cfg, graph=graph, pending=pending, input_fn=scripted("e", "3"), console=capture())

    assert applied == [("m1", "tracker_interview")]


def test_quit_stops_and_keeps_the_rest_pending(tmp_path) -> None:
    cfg, log, pending, applied, graph = setup_pending(tmp_path, ["m1", "m2"])

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("q"), console=capture())

    assert handled == 0
    assert len(pending.list()) == 2


def test_card_never_parses_email_text_as_markup() -> None:
    out = capture()
    payload = {
        "category": "rejection",
        "category_probs": {"rejection": 1.0},
        "subject": "[link=https://evil.example]click[/link] [bold]x",
        "sender_domain": "acme.com",
        "reason": "[red]held",
    }
    out.print(card(1, 1, payload, {"body_text": "[blink]hi\x1b[2J"}))

    text = output(out)
    assert "[link=https://evil.example]click[/link] [bold]x" in text
    assert "[red]held" in text
    assert "[blink]hi\\x1b[2J" in text
