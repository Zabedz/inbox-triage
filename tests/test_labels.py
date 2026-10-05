from __future__ import annotations

from triage.labels import CATEGORIES, DEFAULT_ACTION, ExtractedFields, LabeledEmail


def test_categories_have_eight_values() -> None:
    assert len(CATEGORIES) == 8
    assert CATEGORIES[0] == "rejection"
    assert CATEGORIES[-1] == "other"


def test_label_round_trips_json() -> None:
    record = LabeledEmail(
        email_id="e1",
        from_="a@b.com",
        subject="s",
        body="b",
        category="rejection",
        extracted=ExtractedFields(company="Acme"),
        matched_tracker_id=3,
        action="tracker_rejected",
    )
    restored = LabeledEmail.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.extracted.company == "Acme"


def test_default_actions() -> None:
    assert DEFAULT_ACTION["rejection"] == "tracker_rejected"
    assert DEFAULT_ACTION["other"] is None
