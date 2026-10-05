"""Model interfaces and provider implementations."""

from triage.config import Config
from triage.models.base import (
    Answer,
    ChoiceQuestion,
    DecisionModel,
    GenerativeModel,
    Question,
    ScoreQuestion,
    choice,
    score,
)
from triage.models.decision_llm import LLMDecisionModel
from triage.models.jev import (
    JevAPIError,
    JevClient,
    JevDecisionModel,
    JevError,
    JevResponse,
    JevRetryableError,
    JevUsage,
)
from triage.models.llm import (
    AnthropicGenerativeModel,
    GenerativeAPIError,
    GenerativeError,
    GenerativeRetryableError,
    ZenGenerativeModel,
    build_generative_model,
)


def build_decision_model(cfg: Config, backend: str | None = None) -> DecisionModel:
    """Build the configured decision model (Jev or an LLM-backed one)."""
    backend = backend or cfg.decision_backend
    if backend == "jev":
        return JevDecisionModel(JevClient(cfg.jev.endpoint, cfg.jev.api_key, cfg.jev.model))
    if backend == "llm":
        return LLMDecisionModel(build_generative_model(cfg))
    raise ValueError(
        f"unknown decision backend {backend!r}; set DECISION_BACKEND to 'jev' or 'llm'"
    )


__all__ = [
    "Answer",
    "ChoiceQuestion",
    "DecisionModel",
    "GenerativeModel",
    "Question",
    "ScoreQuestion",
    "choice",
    "score",
    "JevAPIError",
    "JevClient",
    "JevDecisionModel",
    "JevError",
    "JevRetryableError",
    "JevResponse",
    "JevUsage",
    "LLMDecisionModel",
    "AnthropicGenerativeModel",
    "ZenGenerativeModel",
    "GenerativeAPIError",
    "GenerativeError",
    "GenerativeRetryableError",
    "build_generative_model",
    "build_decision_model",
]
