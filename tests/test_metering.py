from __future__ import annotations

from evals.providers.metering import Ledger, MeteredDecision, MeteredGenerative
from triage.models.base import Usage


class Reporting:
    def __init__(self) -> None:
        self.last_usage = Usage(input_tokens=100, output_tokens=7)

    def decide(self, state, questions):
        return {}

    def generate(self, messages, schema):
        return "draft"


class Silent:
    def decide(self, state, questions):
        return {}


def test_ledger_sums_tokens_per_role() -> None:
    ledger = Ledger()
    MeteredDecision(Reporting(), ledger).decide("s", {})
    MeteredDecision(Reporting(), ledger).decide("s", {})
    MeteredGenerative(Reporting(), ledger).generate([], object)
    MeteredDecision(Silent(), ledger).decide("s", {})

    assert ledger.totals["decision"] == {"calls": 3, "input_tokens": 200, "output_tokens": 14}
    assert ledger.tokens() == (300, 21)
