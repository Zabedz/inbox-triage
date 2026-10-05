from __future__ import annotations

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
