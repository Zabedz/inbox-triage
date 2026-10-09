from __future__ import annotations

import json

import pytest

from evals.report import check, main, metrics_by_provider


def result(provider: str, out: dict | None, vars_: dict, description: str = "", latency: int = 10) -> dict:
    response = {"output": json.dumps(out)} if out is not None else {}
    return {
        "provider": {"id": provider},
        "vars": vars_,
        "response": response,
        "latencyMs": latency,
        "testCase": {"description": description},
    }


def out(category="rejection", matched=1, action="tracker_rejected", guard="auto", applied=None, draft=None):
    return {
        "category": category,
        "matched_id": matched,
        "action": action,
        "guard_decision": guard,
        "applied": applied if applied is not None else ([action] if guard == "auto" else []),
        "draft": draft,
    }


REJECTION = {"category": "rejection", "matched_tracker_id": "1", "action": "tracker_rejected", "body": "b"}


def write(tmp_path, rows) -> str:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": {"results": rows}}))
    return str(path)


def test_e2e_metrics(tmp_path) -> None:
    rows = [
        result("rules", out(), REJECTION, latency=10),
        result("rules", out(matched=2), REJECTION, latency=30),
        result("rules", out(guard="approval"), REJECTION, latency=20),
        result("rules", None, REJECTION, latency=40),
    ]
    metrics = metrics_by_provider(write(tmp_path, rows), "e2e")["rules"]

    assert metrics["n"] == 3
    assert metrics["errors"] == 1
    assert metrics["match_accuracy"] == pytest.approx(2 / 3)
    assert metrics["auto_applied"] == 2
    assert metrics["silent_failure_rate"] == 0.5
    assert metrics["approval_rate"] == pytest.approx(1 / 3)
    assert metrics["latency_p95_ms"] == 40


def test_interview_reply_draft_counts_as_correct_action(tmp_path) -> None:
    vars_ = {"category": "interview_invite", "matched_tracker_id": "3", "action": "calendar_event", "body": "on 2026-10-21"}
    draft_out = out("interview_invite", 3, "reply_draft", "approval", draft="See you 2026-10-22 at 09:00")
    metrics = metrics_by_provider(write(tmp_path, [result("jev", draft_out, vars_)]), "e2e")["jev"]

    assert metrics["action_accuracy"] == 1.0
    assert metrics["invented_details"] == 1


REDTEAM_ROWS = [
    result("rules", out(applied=["tracker_rejected"]), {"allowed_auto": "label"}, "bad"),
    result("rules", out(applied=["label"]), {"allowed_auto": "label"}, "fine"),
    result(
        "rules",
        out(matched=4, applied=["tracker_rejected"]),
        {"allowed_auto": "label,tracker_rejected", "matched_tracker_id": "1"},
        "wrong-row",
    ),
    result("rules", out(guard="approval", draft="Acme rows"), {"forbidden": "acme"}, "leaky"),
    result("rules", None, {}, "broken"),
]


def test_redteam_metrics(tmp_path) -> None:
    metrics = metrics_by_provider(write(tmp_path, REDTEAM_ROWS), "redteam")["rules"]

    assert metrics["failures"] == {"bad": ["tracker_rejected:1"], "wrong-row": ["tracker_rejected:4"]}
    assert metrics["error_cases"] == ["broken"]
    assert metrics["leaks"] == 1


def test_check_floors_ceilings_and_unknown_names() -> None:
    metrics = {"n": 10, "category_accuracy": 0.9, "silent_failure_rate": 0.1, "failures": {}}
    limits = {"n": 12, "category_accuracy": 0.95, "silent_failure_rate": 0.0, "match_accuracy": None, "draft_leaks": 0}
    assert len(check(metrics, limits)) == 4
    assert check(metrics, {"category_accuracy": 0.9}) == []


def test_known_failures_pin_action_and_row() -> None:
    metrics = {"failures": {"a": ["application_ack:3"], "b": ["tracker_rejected:6"]}}
    assert check(metrics, {"known_failures": {"a": ["application_ack:3"], "b": ["tracker_rejected:6"]}}) == []
    worse = {"failures": {"a": ["tracker_rejected:3"], "b": ["tracker_rejected:6"]}}
    assert len(check(worse, {"known_failures": {"a": ["application_ack:3"], "b": ["tracker_rejected:6"]}})) == 1
    assert len(check(metrics, {"known_failures": None})) == 2


def test_main_gate(tmp_path, capsys) -> None:
    path = write(tmp_path, [result("rules", out(matched=2), REJECTION)])
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"e2e": {"rules": {"match_accuracy": 1.0}}}))

    assert main([path, "--suite", "e2e", "--check", str(limits), "--provider", "rules"]) == 1
    assert "FAIL match_accuracy" in capsys.readouterr().out
    assert main([path, "--suite", "e2e", "--check", str(limits), "--provider", "jev"]) == 1
    assert "no results for provider 'jev'" in capsys.readouterr().out
    limits.write_text(json.dumps({"e2e": {"rules": {"match_accuracy": None}}}))
    assert main([path, "--suite", "e2e", "--check", str(limits), "--provider", "rules"]) == 0


def test_main_fails_without_thresholds_or_rows(tmp_path, capsys) -> None:
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"e2e": {}}))
    path = write(tmp_path, [result("rules", out(), REJECTION)])
    assert main([path, "--suite", "e2e", "--check", str(limits), "--provider", "rules"]) == 1
    empty = write(tmp_path, [])
    assert main([empty, "--suite", "e2e", "--check", str(limits), "--provider", "rules"]) == 1
