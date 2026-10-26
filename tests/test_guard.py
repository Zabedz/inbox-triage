from __future__ import annotations

import pytest

from triage.guard import Action, GuardError, GuardThresholds, decide, sender_matches, status_action

T = GuardThresholds(category=0.9, match=0.9)


def test_label_auto() -> None:
    d = decide("rejection", 0.5, 1, 0.5, Action.LABEL, T)
    assert d.route == "auto"


def test_ack_auto() -> None:
    d = decide("application_ack", 0.5, None, 0.0, Action.ACK, T)
    assert d.route == "auto"


def test_rejection_auto_when_thresholds_met() -> None:
    d = decide("rejection", 0.95, 12, 0.95, Action.TRACKER_REJECTED, T, "acme.com", ["Acme Corp"])
    assert d.route == "auto"


def test_rejection_approval_below_category() -> None:
    d = decide("rejection", 0.8, 12, 0.95, Action.TRACKER_REJECTED, T)
    assert d.route == "approval"


def test_rejection_approval_no_match() -> None:
    d = decide("rejection", 0.99, None, 0.0, Action.TRACKER_REJECTED, T)
    assert d.route == "approval"


def test_interview_always_approval() -> None:
    for action in (Action.CALENDAR_EVENT, Action.TRACKER_INTERVIEW, Action.REPLY_DRAFT):
        d = decide("interview_invite", 0.99, 12, 0.99, action, T)
        assert d.route == "approval"


def test_offer_always_approval() -> None:
    d = decide("offer", 0.99, 12, 0.99, Action.TRACKER_OFFER, T)
    assert d.route == "approval"


def test_disallowed_action_raises() -> None:
    with pytest.raises(GuardError):
        decide("rejection", 0.99, 12, 0.99, Action.CALENDAR_EVENT, T)


def test_calendar_event_resolves_to_interview_for_interview_invite() -> None:
    assert status_action(Action.CALENDAR_EVENT, "interview_invite") == Action.TRACKER_INTERVIEW


def test_calendar_event_stays_itself_for_assessment() -> None:
    assert status_action(Action.CALENDAR_EVENT, "assessment") == Action.CALENDAR_EVENT


def test_unknown_category_raises() -> None:
    with pytest.raises(GuardError):
        decide("bogus", 0.99, 12, 0.99, Action.LABEL, T)


@pytest.mark.parametrize(
    ("category_prob", "match_prob", "route"),
    [
        (0.99, 0.99, "auto"),
        (0.989, 0.99, "approval"),
        (0.99, 0.989, "approval"),
        (0.999, 0.5, "approval"),
        (0.5, 0.999, "approval"),
    ],
)
def test_rejection_needs_both_thresholds(category_prob, match_prob, route) -> None:
    thresholds = GuardThresholds(category=0.99, match=0.99)
    d = decide(
        "rejection", category_prob, 12, match_prob, Action.TRACKER_REJECTED, thresholds,
        "acme.com", ["Acme Corp"],
    )
    assert d.route == route


@pytest.mark.parametrize(
    ("domain", "names"),
    [
        ("acme.com", ["Acme Corp"]),
        ("acmecorp.com", ["Acme Corp"]),
        ("careers.globex.com", ["Globex"]),
        ("mail.acme.co.uk", ["Acme Ltd"]),
        ("talentbridge.com", ["Hooli", "TalentBridge"]),
        ("stark-industries.com", ["Stark Industries"]),
    ],
)
def test_sender_matches_company_or_agency(domain, names) -> None:
    assert sender_matches(domain, names)


@pytest.mark.parametrize(
    ("domain", "names"),
    [
        ("hooli-careers.example", ["Hooli"]),
        ("h\u043e\u043eli.com", ["Hooli"]),
        ("hooli.com.evil.example", ["Hooli"]),
        ("ats-mail.example", ["Hooli"]),
        ("acme.com", []),
        ("", ["Acme"]),
    ],
)
def test_sender_mismatch(domain, names) -> None:
    assert not sender_matches(domain, names)


def test_rejection_from_other_sender_needs_approval() -> None:
    d = decide("rejection", 0.999, 6, 0.999, Action.TRACKER_REJECTED, T, "hooli-careers.example", ["Hooli"])
    assert d.route == "approval"
    assert "hooli-careers.example" in d.reason


def test_label_ignores_sender() -> None:
    assert decide("rejection", 0.5, None, 0.0, Action.LABEL, T, "anything.example").route == "auto"


def test_status_already_set_is_skipped() -> None:
    d = decide("rejection", 0.5, 4, 0.5, Action.TRACKER_REJECTED, T, "ats.example", ["Acme"], "Rejected")
    assert (d.route, d.reason) == ("skip", "tracker already says Rejected")


def test_different_status_is_not_skipped() -> None:
    d = decide("rejection", 0.5, 4, 0.5, Action.TRACKER_REJECTED, T, "ats.example", ["Acme"], "Interview")
    assert d.route == "approval"


def test_unmatched_status_action_is_never_skipped() -> None:
    d = decide("offer", 0.99, None, 0.0, Action.TRACKER_OFFER, T, row_status="Offer")
    assert d.route == "approval"
