from __future__ import annotations

import pytest

from triage.config import load


def test_thresholds_read_from_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GUARD_CATEGORY_THRESHOLD", "0.95")
    monkeypatch.setenv("GUARD_MATCH_THRESHOLD", "0.97")
    cfg = load(tmp_path / "missing.env")
    assert (cfg.guard_category_threshold, cfg.guard_match_threshold) == (0.95, 0.97)


@pytest.mark.parametrize("value", ["99", "0", "0.099", "1.01"])
def test_threshold_out_of_range_names_variable(monkeypatch, tmp_path, value) -> None:
    monkeypatch.setenv("GUARD_MATCH_THRESHOLD", value)
    with pytest.raises(ValueError, match=f"GUARD_MATCH_THRESHOLD={value}"):
        load(tmp_path / "missing.env")


def test_non_numeric_threshold_names_variable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GUARD_CATEGORY_THRESHOLD", "high")
    with pytest.raises(ValueError, match="GUARD_CATEGORY_THRESHOLD='high' is not a number"):
        load(tmp_path / "missing.env")
