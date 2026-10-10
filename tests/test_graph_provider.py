from __future__ import annotations

import json

import pytest

from evals.datasets.generate import TRACKER
from evals.providers import graph_provider
from evals.providers.graph_provider import _build_state, call_api


def test_build_state_redacts_and_derives_domain() -> None:
    state = _build_state(
        {
            "subject": "Interview with Zabed",
            "from_": "Recruiting <hr@acme.com>",
            "body": "Email me at john@acme.com",
        },
        names=("Zabed",),
    )
    assert state["sender_domain"] == "acme.com"
    assert "john@acme.com" not in state["body_text"]
    assert "[EMAIL_1]" in state["body_text"]
    assert "Zabed" not in state["subject_text"]
    assert "[PERSON_1]" in state["subject_text"]


def test_call_api_fake_provider() -> None:
    result = call_api(
        "body text",
        {"config": {"model": "fake"}},
        {"vars": {"body": "We will not be moving forward.", "subject": "s", "from_": "hr@acme.com"}},
    )
    assert result["output"] == "rejection"
    assert result["namedScores"]["p_rejection"] == 1.0


@pytest.fixture
def synthetic_tracker(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_tracker.json"
    path.write_text(json.dumps(TRACKER))
    monkeypatch.setattr(graph_provider, "_TRACKER_PATH", path)
    monkeypatch.setattr(graph_provider, "_synthetic_db", None)


def e2e(vars_: dict) -> dict:
    result = call_api("", {"config": {"task": "e2e", "model": "rules"}}, {"vars": vars_})
    return json.loads(result["output"])


def test_e2e_confident_rejection_auto_applies(synthetic_tracker) -> None:
    out = e2e(
        {
            "email_id": "e1",
            "from_": "no-reply@globex.com",
            "subject": "Your application to Globex",
            "body": "Thank you for applying to the Software Engineer role at Globex. "
            "We have decided not to move forward.",
        }
    )
    assert (out["category"], out["matched_id"], out["guard_decision"]) == ("rejection", 2, "auto")
    assert out["applied"] == ["tracker_rejected"]


def test_e2e_interview_invite_waits_for_approval(synthetic_tracker) -> None:
    out = e2e(
        {
            "email_id": "e2",
            "from_": "recruiting@initech.com",
            "subject": "Interview invitation - Initech",
            "body": "We would like to invite you to interview for the Data Scientist position "
            "at Initech (req IN-3001) on 2026-10-23 at 14:00 GMT.",
        }
    )
    assert (out["category"], out["matched_id"], out["action"]) == ("interview_invite", 3, "calendar_event")
    assert out["guard_decision"] == "approval"
    assert out["applied"] == []


def test_e2e_real_model_without_llm_provider_refuses(synthetic_tracker, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "")
    with pytest.raises(ValueError, match="needs LLM_PROVIDER"):
        call_api("", {"config": {"task": "e2e", "model": "jev"}}, {"vars": {"body": "x"}})


def test_e2e_reports_timings_and_redaction_mode(synthetic_tracker) -> None:
    vars_ = {"email_id": "e3", "from_": "hr@globex.com", "subject": "s", "body": "Hi Jordan Lee, unfortunately no."}
    redacted = json.loads(
        call_api("", {"config": {"task": "e2e", "model": "rules", "candidate_names": "Jordan Lee"}}, {"vars": vars_})["output"]
    )
    raw = json.loads(
        call_api("", {"config": {"task": "e2e", "model": "rules", "redact": False}}, {"vars": vars_})["output"]
    )

    assert redacted["redacted"] is True and raw["redacted"] is False
    assert {"redact", "classify", "guard"} <= set(redacted["timings_ms"])
    assert redacted["tokens"]["decision"]["calls"] >= 2


def test_skip_redaction_keeps_raw_text() -> None:
    from triage.gmail.client import Email

    email = Email(id="e", thread_id="t", from_="a@b.com", to="", subject="Hi Jordan", date=None,
                  body_plain="Call 020 7946 0123", body_html="", snippet="")
    assert graph_provider._skip_redaction({"email": email}) == {"body_text": "Call 020 7946 0123", "subject_text": "Hi Jordan"}
