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


def out(category="rejection", matched=1, action="tracker_rejected", guard="auto", applied=None, draft=None, **extra):
    return {**extra,
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

    assert metrics["n"] == 4
    assert metrics["errors"] == 1
    assert metrics["match_accuracy"] == pytest.approx(2 / 4)
    assert metrics["category_accuracy"] == pytest.approx(3 / 4)
    assert metrics["auto_applied"] == 2
    assert metrics["silent_failure_rate"] == 0.5
    assert metrics["approval_rate"] == pytest.approx(1 / 4)
    assert metrics["class_f1"] == {"rejection": pytest.approx(0.857, abs=0.001)}


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


def test_graph_latency_and_node_profile(tmp_path) -> None:
    timed = {"timings_ms": {"a": 10.0, "b": 30.0}, "node_tokens": {"a": 0, "b": 120}}
    metrics = metrics_by_provider(write(tmp_path, [result("jev", out(**timed), REJECTION)]), "e2e")["jev"]

    assert metrics["graph_latency_mean_ms"] == 40.0
    assert metrics["nodes"]["b"] == {"mean_ms": 30.0, "p95_ms": 30.0, "mean_tokens": 120.0}


def extracted(fields: dict, ran: bool = True) -> dict:
    return out(extracted=fields, timings_ms={"extract": 5.0} if ran else {"classify": 5.0})


def labelled(**fields) -> dict:
    return {**REJECTION, "extracted": fields}


def scores_for(tmp_path, rows) -> dict:
    return metrics_by_provider(write(tmp_path, rows), "e2e")["jev"]["fields"]


def test_recall_counts_only_labelled_emails(tmp_path) -> None:
    rows = [
        result("jev", extracted({"agency": "Hays"}), labelled(agency="Hays")),
        result("jev", extracted({}), labelled(agency="Hays")),
        result("jev", extracted({}), labelled(agency=None)),
        result("jev", extracted({}), labelled(agency=None)),
    ]
    agency = scores_for(tmp_path, rows)["agency"]

    assert (agency["found"], agency["labelled"], agency["recall"]) == (1, 2, 0.5)


def test_false_positives_count_only_null_labels(tmp_path) -> None:
    rows = [
        result("jev", extracted({"agency": "Hays"}), labelled(agency=None)),
        result("jev", extracted({"agency": ""}), labelled(agency=None)),
        result("jev", extracted({}), labelled(agency=None)),
        result("jev", extracted({"agency": "Wrong"}), labelled(agency="Hays")),
    ]
    agency = scores_for(tmp_path, rows)["agency"]

    assert (agency["false_positives"], agency["unlabelled"]) == (1, 3)
    assert agency["fp_rate"] == pytest.approx(1 / 3, abs=0.001)
    assert (agency["found"], agency["labelled"]) == (0, 1)


def test_emails_skipped_before_extract_are_left_out(tmp_path) -> None:
    rows = [
        result("jev", extracted({"company": "Acme Corp"}), labelled(company="Acme Corp", role=None)),
        result("jev", extracted({}, ran=False), labelled(company="Acme Corp", role=None)),
        result("jev", out(extracted={}), labelled(company="Acme Corp", role=None)),
    ]
    scores = scores_for(tmp_path, rows)

    assert (scores["company"]["found"], scores["company"]["labelled"]) == (1, 1)
    assert scores["role"]["unlabelled"] == 1


def test_errored_email_counts_in_recall_only(tmp_path) -> None:
    vars_ = labelled(company="Acme Corp", agency=None)
    rows = [result("jev", extracted({"company": "Acme Corp"}), vars_), result("jev", None, vars_)]
    scores = scores_for(tmp_path, rows)

    assert (scores["company"]["found"], scores["company"]["labelled"]) == (1, 2)
    assert (scores["agency"]["false_positives"], scores["agency"]["unlabelled"]) == (0, 1)


def test_labels_given_as_json_string(tmp_path) -> None:
    vars_ = {**REJECTION, "extracted": json.dumps({"company": "Acme Corp"})}
    company = scores_for(tmp_path, [result("jev", extracted({"company": "acme corp"}), vars_)])["company"]

    assert company["recall"] == 1.0
