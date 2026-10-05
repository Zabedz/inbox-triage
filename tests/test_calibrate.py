from __future__ import annotations

import json

import pytest

from evals.calibrate import (
    brier,
    ece,
    load_promptfoo_results,
    recommend_threshold,
    reliability,
)


def test_ece_perfect() -> None:
    probs = [{"a": 1.0, "b": 0.0}, {"b": 1.0, "a": 0.0}]
    labels = ["a", "b"]
    assert ece(probs, labels) == 0.0


def test_ece_overconfident() -> None:
    probs = [{"a": 0.9, "b": 0.1}, {"a": 0.9, "b": 0.1}]
    labels = ["a", "b"]
    assert ece(probs, labels) == pytest.approx(0.4)


def test_brier_perfect() -> None:
    probs = [{"a": 1.0, "b": 0.0}, {"b": 1.0, "a": 0.0}]
    labels = ["a", "b"]
    assert brier(probs, labels) == 0.0


def test_brier_overconfident() -> None:
    probs = [{"a": 0.9, "b": 0.1}, {"a": 0.9, "b": 0.1}]
    labels = ["a", "b"]
    assert brier(probs, labels) == pytest.approx(0.82)


def test_reliability_rows() -> None:
    probs = [{"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}]
    labels = ["a", "b"]
    rows = reliability(probs, labels)
    assert sum(count for _, _, count in rows) == 2


def test_recommend_threshold_lowest_satisfying() -> None:
    probs = [{"a": 0.99}, {"a": 0.95}, {"a": 0.6}]
    labels = ["a", "a", "b"]
    assert recommend_threshold(probs, labels, target_failure=0.5) == 0.6


def test_recommend_threshold_skips_unreliable() -> None:
    probs = [{"a": 0.99}, {"a": 0.6}]
    labels = ["a", "b"]
    assert recommend_threshold(probs, labels, target_failure=0.4) == 0.99


def test_recommend_threshold_none_when_never_meets() -> None:
    probs = [{"a": 0.99}]
    labels = ["b"]
    assert recommend_threshold(probs, labels, target_failure=0.05) is None


def test_brier_accounts_for_missing_class() -> None:
    probs = [{"a": 1.0}]
    labels = ["b"]
    assert brier(probs, labels, options=["a", "b"]) == pytest.approx(2.0)


def test_load_promptfoo_results(tmp_path: pytest.TempPathFactory) -> None:
    payload = {
        "results": {
            "results": [
                {
                    "provider": {"id": "jev"},
                    "vars": {"category": "rejection"},
                    "response": {"namedScores": {"p_rejection": 0.9, "p_other": 0.1}},
                },
                {
                    "provider": {"id": "jev"},
                    "vars": {"category": "offer"},
                    "response": {"namedScores": {"p_rejection": 0.8, "p_other": 0.2}},
                },
                {
                    "provider": {"id": "llm"},
                    "vars": {"category": "offer"},
                    "response": {"namedScores": {"p_offer": 0.7, "p_other": 0.3}},
                },
            ]
        }
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))

    data = load_promptfoo_results(path)
    assert data["jev"]["labels"] == ["rejection", "offer"]
    assert data["jev"]["probs"][0] == {"rejection": 0.9, "other": 0.1}
    assert data["llm"]["probs"][0] == {"offer": 0.7, "other": 0.3}
