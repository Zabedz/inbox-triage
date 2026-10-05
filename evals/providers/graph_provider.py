"""promptfoo Python provider: run the classify node, report category + probabilities."""

from __future__ import annotations

from typing import Any

from triage.config import load
from triage.graph.graph import sender_domain
from triage.graph.nodes.classify import build_classify
from triage.graph.state import State
from triage.models import build_decision_model
from triage.models.base import Answer, DecisionModel
from triage.redact import redact


class _FakeDecisionModel:
    """Deterministic stand-in for smoke-testing the harness without model keys."""

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Answer]:
        return {
            "category": Answer(
                type="choice", choice="rejection", probabilities={"rejection": 1.0}
            ),
            "needs_reply": Answer(type="choice", choice="no", probabilities={"no": 1.0}),
            "urgency": Answer(type="score", score="low", probabilities={"low": 1.0}),
        }


def _build_model(kind: str) -> DecisionModel:
    if kind == "fake":
        return _FakeDecisionModel()
    return build_decision_model(load(), backend=kind)


def _build_state(vars_: dict[str, Any], names: tuple[str, ...] = ()) -> State:
    subject = vars_.get("subject", "")
    from_ = vars_.get("from_", "")
    body = vars_.get("body", "")
    return {
        "subject_text": redact(subject, names=names).text,
        "sender_domain": sender_domain(from_),
        "body_text": redact(body, names=names).text,
    }


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    config = (options or {}).get("config", {})
    kind = config.get("model", "jev")
    cfg = None if kind == "fake" else load()
    names = cfg.candidate_names if cfg else ()

    vars_ = context.get("vars", {})
    state = _build_state(vars_, names)
    result = build_classify(_build_model(kind))(state)

    named_scores = {f"p_{option}": prob for option, prob in result["category_probs"].items()}
    return {"output": result["category"], "namedScores": named_scores}
