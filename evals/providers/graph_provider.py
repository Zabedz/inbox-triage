"""promptfoo Python provider: classify / extract / match nodes."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from triage.config import load
from triage.graph.graph import sender_domain
from triage.graph.nodes.classify import build_classify
from triage.graph.nodes.extract import ExtractedField, Extraction, build_extract
from triage.graph.nodes.match import MatchStep, build_match
from triage.graph.state import State
from triage.models import build_decision_model
from triage.models.base import Answer
from triage.models.llm import build_generative_model
from triage.redact import redact
from triage.tracker.db import TrackerDB
from triage.tracker.schema import Application
from triage.tracker.snapshot import load_into_sqlite

_TRACKER_PATH = Path(__file__).resolve().parent.parent / "datasets" / "synthetic_tracker.json"


class _FakeDecisionModel:
    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Answer]:
        return {
            "category": Answer(
                type="choice", choice="rejection", probabilities={"rejection": 1.0}
            ),
            "needs_reply": Answer(type="choice", choice="no", probabilities={"no": 1.0}),
            "urgency": Answer(type="score", score="low", probabilities={"low": 1.0}),
        }


class _FakeExtractionLLM:
    def generate(self, messages: list[dict[str, str]], schema: Any) -> Any:
        return Extraction(
            company=ExtractedField(value="Acme Corp", span="Acme Corp"),
            role=ExtractedField(value="AI Engineer", span="AI Engineer"),
        )


class _FakeMatchLLM:
    def generate(self, messages: list[dict[str, str]], schema: Any) -> Any:
        return MatchStep(action="done")


class _FakeMatchDecision:
    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Answer]:
        options = list(questions["match"].criteria)
        probs = {opt: (1.0 if opt == "none_of_these" else 0.0) for opt in options}
        return {"match": Answer(type="choice", choice="none_of_these", probabilities=probs)}


_synthetic_db: TrackerDB | None = None


def _synthetic_tracker() -> TrackerDB:
    global _synthetic_db
    if _synthetic_db is None:
        if not _TRACKER_PATH.exists():
            raise FileNotFoundError(
                f"missing {_TRACKER_PATH}; run evals/datasets/generate.py first"
            )
        fd, path = tempfile.mkstemp(prefix="triage-synthetic-", suffix=".db")
        os.close(fd)
        rows = json.loads(_TRACKER_PATH.read_text())
        apps = [
            Application(
                id=row["id"],
                date_applied=date.fromisoformat(row["date_applied"]),
                company=row["company"],
                via=row["via"],
                role=row["role"],
                location="",
                score="",
                status="",
                pdf="",
                report="",
                notes="",
                url="",
                requisition_id=row["req"],
            )
            for row in rows
        ]
        load_into_sqlite(Path(path), apps, [], [])
        _synthetic_db = TrackerDB(path)
    return _synthetic_db


def _build_state(vars_: dict[str, Any], names: tuple[str, ...] = ()) -> State:
    subject = vars_.get("subject", "")
    from_ = vars_.get("from_", "")
    body = vars_.get("body", "")
    extracted = vars_.get("extracted") or {}
    if isinstance(extracted, str):
        extracted = json.loads(extracted)
    return {
        "subject_text": redact(subject, names=names).text,
        "sender_domain": sender_domain(from_),
        "body_text": redact(body, names=names).text,
        "extracted": extracted,
    }


def _call_classify(kind: str, state: State) -> dict[str, Any]:
    model = _FakeDecisionModel() if kind == "fake" else build_decision_model(load(), backend=kind)
    result = build_classify(model)(state)
    named_scores = {f"p_{option}": prob for option, prob in result["category_probs"].items()}
    return {"output": result["category"], "namedScores": named_scores}


def _call_extract(kind: str, state: State) -> dict[str, Any]:
    model: Any = _FakeExtractionLLM() if kind == "fake" else build_generative_model(load())
    result = build_extract(model)(state)
    return {"output": json.dumps(result["extracted"])}


def _call_match(kind: str, state: State) -> dict[str, Any]:
    if kind == "fake":
        llm: Any = _FakeMatchLLM()
        decision: Any = _FakeMatchDecision()
    else:
        llm = build_generative_model(load())
        decision = build_decision_model(load(), backend=kind)
    result = build_match(_synthetic_tracker(), llm, decision)(state)
    return {"output": json.dumps(result["matched_id"])}


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    config = (options or {}).get("config", {})
    task = config.get("task", "classify")
    kind = config.get("model", "jev")
    cfg = None if kind == "fake" else load()
    names = cfg.candidate_names if cfg else ()

    state = _build_state(context.get("vars", {}), names)

    if task == "extract":
        return _call_extract(kind, state)
    if task == "match":
        return _call_match(kind, state)
    return _call_classify(kind, state)
