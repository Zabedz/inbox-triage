from __future__ import annotations

from conftest import make_app

from evals.providers.rules import RulesDecisionModel, RulesGenerativeModel, classify_text
from triage.graph.nodes.extract import Extraction
from triage.graph.nodes.match import MatchStep
from triage.graph.nodes.propose_action import Draft
from triage.models.base import choice

ROWS = [make_app(1, "Acme Corp", "AI Engineer"), make_app(2, "Globex", "Software Engineer")]


def test_classify_text_by_keywords() -> None:
    assert classify_text("We have decided not to move forward.")[0] == "rejection"
    assert classify_text("We have received your application.")[0] == "application_ack"
    assert classify_text("Hello there") == ("other", {"other": 1.0})


def test_action_is_category_default_when_allowed() -> None:
    question = choice("?", {"label": "l", "tracker_rejected": "r"})
    answer = RulesDecisionModel().decide("Unfortunately, no.", {"action": question})["action"]

    assert answer.best() == "tracker_rejected"
    assert answer.confidence == 1.0


def test_match_prefers_requisition_id() -> None:
    criteria = {
        "candidate_0": "#1 Acme Corp - AI Engineer (req AC-1001)",
        "candidate_1": "#4 Acme Corp - AI Engineer (req AC-1002)",
        "none_of_these": "none",
    }
    state = '["<DATA>\\nAcme Corp req AC-1002\\n</DATA>", "Tool search_tracker -> AC-1001"]'
    answer = RulesDecisionModel().decide(state, {"match": choice("?", criteria)})["match"]

    assert answer.best() == "candidate_1"


def test_extract_returns_verbatim_spans() -> None:
    model = RulesGenerativeModel(ROWS)
    text = "<DATA>\nThe AI Engineer role at Acme Corp (req AC-1001), on 2026-10-21 at 10:00.\n</DATA>"
    result = model.generate([{"role": "user", "content": text}], Extraction)

    assert result.company is not None and result.company.span == "Acme Corp"
    assert result.role is not None and result.role.span == "AI Engineer"
    assert result.requisition_id is not None and result.requisition_id.span == "AC-1001"
    assert result.date_time is not None and result.date_time.span == "2026-10-21 at 10:00"


def test_match_loop_searches_once_then_stops() -> None:
    model = RulesGenerativeModel(ROWS)
    first = model.generate([{"role": "user", "content": "<DATA>\nGlobex\n</DATA>"}], MatchStep)
    second = model.generate(
        [{"role": "user", "content": "<DATA>\nGlobex\n</DATA>\n\nTool search_tracker(...) -> []"}],
        MatchStep,
    )

    assert (first.action, first.company) == ("search_tracker", "Globex")
    assert second.action == "done"


def test_draft_is_fixed() -> None:
    draft = RulesGenerativeModel(ROWS).generate([{"role": "user", "content": "x"}], Draft)
    assert draft.body == "Thank you, I will reply shortly."
