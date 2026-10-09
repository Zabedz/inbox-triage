from __future__ import annotations

import json
from pathlib import Path

import yaml

from triage.guard import ALLOWED, AUTO_ACTIONS, THRESHOLD_GATED_AUTO
from triage.labels import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "redteam" / "cases.yaml"
SPEC_ATTACKS = {"hijack", "indirect_body", "indirect_attachment", "data_extraction", "excessive_agency", "swaying"}
ATTACKS = SPEC_ATTACKS | {"tool_abuse", "spoofing", "obfuscation"}


def load() -> list[dict]:
    return yaml.safe_load(CASES.read_text())


def split(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def test_every_spec_attack_type_is_covered() -> None:
    attacks = {case["vars"]["attack"] for case in load()}
    assert SPEC_ATTACKS <= attacks <= ATTACKS


def test_cases_are_well_formed() -> None:
    cases = load()
    assert len({c["description"] for c in cases}) == len(cases)
    for case in cases:
        name, vars_ = case["description"], case["vars"]
        assert vars_["category"] in CATEGORIES, name
        for key in ("allowed_auto", "forbidden", "forbidden_echo"):
            assert isinstance(vars_.get(key, ""), str), f"{name}: list vars expand into separate tests"
        allowed = split(vars_["allowed_auto"])
        assert allowed <= {a.value for a in AUTO_ACTIONS | THRESHOLD_GATED_AUTO}, name
        if allowed & {a.value for a in THRESHOLD_GATED_AUTO}:
            assert vars_["matched_tracker_id"] != "none", f"{name}: a tracker write needs a labelled row"
        if vars_["action"] != "none":
            assert vars_["action"] in {a.value for a in ALLOWED[vars_["category"]]}, name


def test_forbidden_strings_are_absent_and_echoes_present() -> None:
    for case in load():
        vars_ = case["vars"]
        email = " ".join(vars_.get(k, "") for k in ("subject", "body", "body_html")).lower()
        for text in split(vars_["forbidden"]):
            assert text.lower() not in email, f"{case['description']}: {text!r} is in the email"
        for text in split(vars_.get("forbidden_echo", "")):
            assert text.lower() in email, f"{case['description']}: echo {text!r} not in the email"


def test_known_failures_name_real_cases() -> None:
    names = {case["description"] for case in load()}
    thresholds = json.loads((ROOT / "evals" / "thresholds.json").read_text())
    for limits in thresholds["redteam"].values():
        assert set(limits.get("known_failures") or {}) <= names
