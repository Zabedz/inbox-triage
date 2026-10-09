from __future__ import annotations

import json

import pytest

from evals.datasets.generate import generate
from triage.guard import ALLOWED, Action
from triage.labels import CATEGORIES, LabeledEmail


@pytest.fixture
def synthetic(tmp_path: pytest.TempPathFactory) -> list[LabeledEmail]:
    out = tmp_path / "synthetic.jsonl"
    generate(out, seed=42, n_per_category=5)
    return [LabeledEmail.model_validate(json.loads(line)) for line in out.read_text().splitlines()]


def test_covers_all_categories(synthetic: list[LabeledEmail]) -> None:
    seen = {e.category for e in synthetic}
    assert seen == set(CATEGORIES)


def test_deterministic(tmp_path: pytest.TempPathFactory) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    generate(a, seed=7)
    generate(b, seed=7)
    assert a.read_text() == b.read_text()


def test_seed_changes_output(tmp_path: pytest.TempPathFactory) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    generate(a, seed=1)
    generate(b, seed=2)
    assert a.read_text() != b.read_text()


def test_special_cases_present(synthetic: list[LabeledEmail]) -> None:
    ids = {e.email_id for e in synthetic}
    assert "syn-special-agency-unnamed" in ids
    assert "syn-special-near-duplicate" in ids
    assert "syn-special-timezone" in ids
    assert "syn-special-ambiguous" in ids


def test_near_duplicate_matches_correct_row(synthetic: list[LabeledEmail]) -> None:
    record = next(e for e in synthetic if e.email_id == "syn-special-near-duplicate")
    assert record.extracted.requisition_id == "AC-1002"
    assert record.matched_tracker_id == 4


def test_agency_unnamed_has_no_match(synthetic: list[LabeledEmail]) -> None:
    record = next(e for e in synthetic if e.email_id == "syn-special-agency-unnamed")
    assert record.category == "recruiter_outreach"
    assert record.matched_tracker_id is None


def test_timezone_captured(synthetic: list[LabeledEmail]) -> None:
    record = next(e for e in synthetic if e.email_id == "syn-special-timezone")
    assert record.extracted.timezone == "America/Los_Angeles"


def test_ambiguous_has_no_match_and_only_a_label(synthetic: list[LabeledEmail]) -> None:
    record = next(e for e in synthetic if e.email_id == "syn-special-ambiguous")
    assert record.category == "rejection"
    assert record.matched_tracker_id is None
    assert record.action == Action.LABEL


def test_labelled_actions_are_allowed(synthetic: list[LabeledEmail]) -> None:
    for record in synthetic:
        if record.action is not None:
            assert record.action in ALLOWED[record.category], record.email_id
