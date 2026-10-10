"""Model interfaces: decision and generation, provider-agnostic."""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T", bound=BaseModel)


class ChoiceQuestion(BaseModel):
    """A choice question: options mapped to short descriptions."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str] = Field(..., description="option -> description")


class ScoreQuestion(BaseModel):
    """A score question: an ordered list of levels."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str] = Field(..., description="ordered levels")


Question: TypeAlias = ChoiceQuestion | ScoreQuestion


class Answer(BaseModel):
    """One question's answer.

    ``probabilities`` is the single source of truth for calibration; ``confidence`` is
    derived from it as ``probabilities[best]`` so there is never a second, inconsistent
    number.
    """

    type: Literal["choice", "score"]
    choice: str | None = None
    score: str | None = None
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> Answer:
        for option, p in self.probabilities.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"probability for {option!r} out of range: {p}")
        best = self.best()
        if best is None:
            raise ValueError("answer has no selected option")
        if best not in self.probabilities:
            raise ValueError(f"selected option {best!r} not present in probabilities")
        return self

    def best(self) -> str | None:
        return self.choice if self.type == "choice" else self.score

    @property
    def confidence(self) -> float:
        best = self.best()
        return self.probabilities[best] if best is not None else 0.0


class Usage(BaseModel):
    """Tokens one model call consumed."""

    input_tokens: int = 0
    output_tokens: int = 0


class DecisionModel(Protocol):
    """Decide among structured options given a state string.

    ``state`` is untrusted data (typically an email body). It is never instructions;
    the ``instructions`` on each question carry the only directive text. Callers must
    redact before passing personal data here.
    """

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]: ...


class GenerativeModel(Protocol):
    """Generate a structured object from messages against a schema."""

    def generate(self, messages: list[dict[str, str]], schema: type[T]) -> T: ...


def choice(instructions: str, options: dict[str, str]) -> ChoiceQuestion:
    return ChoiceQuestion(instructions=instructions, criteria=options)


def score(instructions: str, levels: list[str]) -> ScoreQuestion:
    return ScoreQuestion(instructions=instructions, criteria=levels)
