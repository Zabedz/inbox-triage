"""promptfoo Python provider: single nodes (classify, extract, match) or the full graph (e2e)."""

from __future__ import annotations

import atexit
import json
import os
import tempfile
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from evals.providers.metering import Ledger, MeteredDecision, MeteredGenerative, run_timed
from evals.providers.rules import RulesDecisionModel, RulesGenerativeModel
from triage.agent import assemble
from triage.config import load
from triage.gmail.client import Email
from triage.graph.graph import sender_domain
from triage.graph.nodes.classify import build_classify
from triage.graph.nodes.extract import ExtractedField, Extraction, build_extract
from triage.graph.nodes.match import MatchStep, build_match
from triage.graph.state import State
from triage.guard import Action, GuardThresholds
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
            fields=[
                ExtractedField(field="company", value="Acme Corp", span="Acme Corp"),
                ExtractedField(field="role", value="AI Engineer", span="AI Engineer"),
            ]
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

FILLER = "This paragraph is filler that makes the email long. "
SERDE = JsonPlusSerializer(allowed_msgpack_modules=[("triage.gmail.client", "Email")])


def _synthetic_tracker() -> TrackerDB:
    global _synthetic_db
    if _synthetic_db is None:
        if not _TRACKER_PATH.exists():
            raise FileNotFoundError(
                f"missing {_TRACKER_PATH}; run evals/datasets/generate.py first"
            )
        fd, path = tempfile.mkstemp(prefix="triage-synthetic-", suffix=".db")
        os.close(fd)
        atexit.register(os.unlink, path)
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


def _thresholds(config: dict[str, Any]) -> GuardThresholds:
    """From the provider config, so results are reproducible across machines."""
    defaults = GuardThresholds()
    return GuardThresholds(
        category=float(config.get("category_threshold", defaults.category)),
        match=float(config.get("match_threshold", defaults.match)),
    )


def _skip_redaction(state: State) -> dict[str, Any]:
    email = state["email"]
    return {"body_text": email.body_plain, "subject_text": email.subject}


def _names(config: dict[str, Any], default: tuple[str, ...]) -> tuple[str, ...]:
    if "candidate_names" not in config:
        return default
    return tuple(n.strip() for n in str(config["candidate_names"]).split(",") if n.strip())


def _email(vars_: dict[str, Any]) -> Email:
    body = vars_.get("body", "")
    if padding := int(vars_.get("body_padding") or 0):
        body = (FILLER * (padding // len(FILLER) + 1))[:padding] + "\n\n" + body
    return Email(
        id=str(vars_.get("email_id", "eval")),
        thread_id="eval",
        from_=vars_.get("from_", ""),
        to="",
        subject=vars_.get("subject", ""),
        date=None,
        body_plain=body,
        body_html=vars_.get("body_html", ""),
        snippet="",
    )


def _call_e2e(
    kind: str, vars_: dict[str, Any], config: dict[str, Any], names: tuple[str, ...]
) -> dict[str, Any]:
    real_tracker = config.get("tracker") == "real"
    tracker = TrackerDB(load().tracker.db) if real_tracker else _synthetic_tracker()
    try:
        return _run_e2e(kind, vars_, config, names, tracker)
    finally:
        if real_tracker:
            tracker.close()


def _run_e2e(
    kind: str,
    vars_: dict[str, Any],
    config: dict[str, Any],
    names: tuple[str, ...],
    tracker: TrackerDB,
) -> dict[str, Any]:
    ledger = Ledger()
    decision: Any
    generative: Any
    model = kind
    if kind == "rules":
        decision = RulesDecisionModel()
        generative = RulesGenerativeModel([r for i in tracker.ids() if (r := tracker.get(i))])
    else:
        cfg = load()
        if not cfg.llm_provider:
            raise ValueError(f"e2e/{kind} needs LLM_PROVIDER; extract and match would run as stubs")
        decision = build_decision_model(cfg, backend=kind)
        generative = build_generative_model(cfg)
        model = f"{kind}+{cfg.llm_provider}:{cfg.llm_model}"
    thresholds = _thresholds(config)
    redacted = config.get("redact", True) is not False
    redact_names = _names(config, names)

    applied: list[str] = []

    def record(action: Action, state: State) -> None:
        applied.append(action.value)

    graph = assemble(
        decision=MeteredDecision(decision, ledger),
        generative=MeteredGenerative(generative, ledger),
        tracker=tracker,
        thresholds=thresholds,
        apply_sink=record,
        log_sink=None,
        checkpointer=InMemorySaver(serde=SERDE),
        redact_names=redact_names,
        nodes=None if redacted else {"redact": _skip_redaction},
    )
    email = _email(vars_)
    # Spans redaction would mask, whether or not this run masks them; pairs raw with redacted.
    masked = sum(
        len(redact(text, names=redact_names).replacements)
        for text in (email.subject, email.body_plain)
    )
    thread = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}}
    timings, node_tokens = run_timed(graph, {"email_id": email.id, "email": email}, thread, ledger)
    result = graph.get_state(thread).values
    category = result.get("category")
    probs = result.get("category_probs", {})
    output = {
        "category": category,
        "category_prob": probs.get(category, 0.0),
        "extracted": result.get("extracted", {}),
        "matched_id": result.get("matched_id"),
        "match_prob": result.get("match_prob", 0.0),
        "action": result.get("action"),
        "action_prob": result.get("action_prob", 0.0),
        "guard_decision": result.get("guard_decision"),
        "applied": applied,
        "draft": result.get("draft"),
        "thresholds": [thresholds.category, thresholds.match],
        "model": model,
        "redacted": redacted,
        "masked_spans": masked,
        "timings_ms": timings,
        "node_tokens": node_tokens,
        "tokens": ledger.totals,
    }
    named_scores = {f"p_{option}": prob for option, prob in probs.items()}
    tokens_in, tokens_out = ledger.tokens()
    usage = {"prompt": tokens_in, "completion": tokens_out, "total": tokens_in + tokens_out}
    return {"output": json.dumps(output), "namedScores": named_scores, "tokenUsage": usage}


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    config = (options or {}).get("config", {})
    task = config.get("task", "classify")
    kind = config.get("model", "jev")
    cfg = None if kind in ("fake", "rules") else load()
    names = cfg.candidate_names if cfg else ()
    vars_ = context.get("vars", {})

    if task == "e2e":
        return _call_e2e(kind, vars_, config, names)

    state = _build_state(vars_, names)
    if task == "extract":
        return _call_extract(kind, state)
    if task == "match":
        return _call_match(kind, state)
    return _call_classify(kind, state)
