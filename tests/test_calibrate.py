from __future__ import annotations

import json

import pytest

from evals.calibrate import (
    brier,
    ece,
    guard_lines,
    load_promptfoo_results,
    recommend_guard_thresholds,
    recommend_threshold,
    reliability,
    round_up,
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


def test_guard_thresholds_searched_jointly() -> None:
    good = [(0.999, 0.999, True)] * 30 + [(0.999, 0.95, True)] * 20 + [(0.97, 0.999, True)] * 20
    bad = [(0.96, 0.999, False), (0.999, 0.90, False)]
    cat, match, admitted, bound = recommend_guard_thresholds(good + bad, target_failure=0.05)
    assert (cat, match, admitted) == (0.97, 0.95, 70)
    assert bound <= 0.05


def test_guard_thresholds_need_enough_evidence() -> None:
    assert recommend_guard_thresholds([(0.99, 0.99, True)] * 11, target_failure=0.05) is None


def test_guard_thresholds_tie_picks_tightest_pair() -> None:
    rows = [(0.9, 0.9, True)] * 60 + [(0.5, 0.1, False)]
    cat, match, admitted, _ = recommend_guard_thresholds(rows, target_failure=0.05)
    assert (cat, match, admitted) == (0.9, 0.9, 60)


@pytest.mark.parametrize(("value", "expected"), [(0.9949, 0.995), (0.29, 0.29), (0.99, 0.99)])
def test_round_up_never_rounds_down(value, expected) -> None:
    assert round_up(value) == expected


def test_printed_guard_thresholds_admit_exactly_the_bounded_rows() -> None:
    rows = [(0.9123 + i / 10_000, 0.95, True) for i in range(52)] + [(0.91, 0.99, False)]
    lines = guard_lines(rows, target_failure=0.05)
    values = dict(line.split("=", 1) for line in lines[1:])
    cat, match = float(values["GUARD_CATEGORY_THRESHOLD"]), float(values["GUARD_MATCH_THRESHOLD"])

    admitted = [r for r in rows if r[0] >= cat and r[1] >= match]
    assert lines[0].startswith(f"admitted={len(admitted)} ")
    assert all(r[2] for r in admitted)
