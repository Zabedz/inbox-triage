from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.datasets.generate import generate
from triage.graph.nodes.extract import (
    DEFINITIONS,
    FIELDS,
    SYSTEM_PROMPT,
    ExtractedField,
    Extraction,
    build_extract,
)


class FakeGenerative:
    def __init__(self, result: Extraction) -> None:
        self._result = result

    def generate(self, messages, schema):
        return self._result


def item(field: str, value: str, span: str | None = None) -> ExtractedField:
    return ExtractedField.model_validate(
        {"field": field, "value": value, "span": value if span is None else span}
    )


def run(body: str, *items: ExtractedField) -> dict[str, str | None]:
    state = {"subject_text": "", "body_text": body}
    out: dict[str, str | None] = build_extract(FakeGenerative(Extraction(fields=list(items))))(
        state
    )["extracted"]
    return out


def test_extract_parses_fields() -> None:
    state = {
        "subject_text": "Interview at Acme Corp",
        "body_text": "AI Engineer role at Acme Corp on 2026-10-21.",
    }
    result = Extraction(
        fields=[
            item("company", "Acme Corp"),
            item("role", "AI Engineer"),
            item("date_time", "2026-10-21"),
        ]
    )
    out = build_extract(FakeGenerative(result))(state)

    assert out["extracted"] == {
        "agency": None,
        "company": "Acme Corp",
        "contact": None,
        "date_time": "2026-10-21",
        "deadline": None,
        "requisition_id": None,
        "role": "AI Engineer",
        "timezone": None,
    }


def test_extract_nulls_hallucinated_span() -> None:
    assert run("no company here", item("company", "Fake Corp"))["company"] is None


def test_extract_nulls_empty_span() -> None:
    assert run("some text", item("company", "X", span=""))["company"] is None


def test_extract_stores_value_inside_span() -> None:
    out = run("Thank you for applying at Globex.", item("company", "Globex", span="at Globex"))
    assert out["company"] == "Globex"


@pytest.mark.parametrize("value", ["2026-10-26T14:00", "", "   "])
def test_extract_keeps_span_when_value_is_not_inside_it(value: str) -> None:
    out = run("Held on 2026-10-26 at 14:00 GMT.", item("date_time", value, "2026-10-26 at 14:00"))
    assert out["date_time"] == "2026-10-26 at 14:00"


def test_extract_falls_back_to_span_when_value_fails_format() -> None:
    out = run("Held on 2026-10-26 at 14:00 GMT.", item("date_time", "14:00", "2026-10-26 at 14:00"))
    assert out["date_time"] == "2026-10-26 at 14:00"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date_time", "AC-1002"),
        ("date_time", "Data Scientist"),
        ("deadline", "Acme Corp"),
        ("deadline", "Please confirm"),
        ("timezone", "2026-10-20"),
        ("timezone", "Acme Corp"),
        ("timezone", "Best"),
        ("timezone", "IT"),
        ("requisition_id", "Umbrella"),
        ("requisition_id", "Initech"),
    ],
)
def test_format_check_drops_misfiled_value(field: str, value: str) -> None:
    assert run(f"text with {value} in it", item(field, value))[field] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date_time", "2026-10-26 at 14:00"),
        ("date_time", "October 21 at 3pm"),
        ("date_time", "October 21st at 3pm"),
        ("date_time", "Tuesday at 10:00"),
        ("date_time", "21/10/2026 14:00"),
        ("deadline", "21st of October"),
        ("deadline", "Oct. 30"),
        ("deadline", "tomorrow"),
        ("deadline", "Nov 3rd"),
        ("timezone", "GMT"),
        ("timezone", "PT"),
        ("timezone", "AEST"),
        ("timezone", "UTC+01:00"),
        ("timezone", "Europe/London"),
        ("timezone", "Pacific Time"),
        ("timezone", "Eastern Standard Time (EST)"),
        ("timezone", "UK time"),
        ("timezone", "PST (UTC-8)"),
        ("timezone", "IST (UTC+5:30)"),
        ("requisition_id", "GL-2001"),
        ("requisition_id", "R12345"),
    ],
)
def test_format_check_keeps_wellformed_value(field: str, value: str) -> None:
    assert run(f"text with {value} in it", item(field, value))[field] == value


def labelled_values() -> list[tuple[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.jsonl"
        generate(path)
        lines = path.read_text().splitlines()
    pairs = set()
    for line in lines:
        for field, value in json.loads(line)["extracted"].items():
            if value:
                pairs.add((field, value))
    return sorted(pairs)


@pytest.mark.parametrize(("field", "value"), labelled_values())
def test_format_checks_pass_every_labelled_value(field: str, value: str) -> None:
    assert run(value, item(field, value))[field] == value


def test_first_item_that_passes_checks_wins() -> None:
    out = run(
        "Acme Corp (req AC-1002) on 2026-10-26 at 14:00, or 2026-10-27 at 09:00.",
        item("date_time", "AC-1002"),
        item("date_time", "2026-10-26 at 14:00"),
        item("date_time", "2026-10-27 at 09:00"),
    )
    assert out["date_time"] == "2026-10-26 at 14:00"


def test_dropped_field_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="triage.graph.extract"):
        run("Acme Corp (req AC-1002)", item("date_time", "AC-1002"), item("role", "Engineer"))

    assert [r.getMessage() for r in caplog.records] == [
        "extract dropped date_time: fails its format check",
        "extract dropped role: span is not in the email",
    ]


@pytest.mark.parametrize(
    "fields",
    [
        [{"field": "salary", "value": "1", "span": "1"}],
        [{"field": "company", "span": "Acme"}],
        [{"field": "company", "value": "Acme", "span": "Acme", "note": "x"}],
    ],
)
def test_schema_rejects_malformed_items(fields: list[dict[str, str]]) -> None:
    with pytest.raises(ValidationError):
        Extraction.model_validate({"fields": fields})


def test_prompt_defines_every_field_in_order() -> None:
    assert list(FIELDS) == sorted(DEFINITIONS)
    positions = [SYSTEM_PROMPT.index(f"- {name}: ") for name in FIELDS]
    assert positions == sorted(positions)
