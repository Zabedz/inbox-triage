from __future__ import annotations

from triage.graph.nodes.extract import ExtractedField, Extraction, build_extract


class FakeGenerative:
    def __init__(self, result: Extraction) -> None:
        self._result = result

    def generate(self, messages, schema):
        return self._result


def test_extract_parses_fields() -> None:
    state = {
        "subject_text": "Interview at Acme Corp",
        "body_text": "AI Engineer role at Acme Corp on 2026-10-21.",
    }
    result = Extraction(
        company=ExtractedField(value="Acme Corp", span="Acme Corp"),
        role=ExtractedField(value="AI Engineer", span="AI Engineer"),
        date_time=ExtractedField(value="2026-10-21", span="2026-10-21"),
    )
    out = build_extract(FakeGenerative(result))(state)

    assert out["extracted"]["company"] == "Acme Corp"
    assert out["extracted"]["role"] == "AI Engineer"
    assert out["extracted"]["date_time"] == "2026-10-21"
    assert out["extracted"]["agency"] is None


def test_extract_nulls_hallucinated_span() -> None:
    state = {"subject_text": "", "body_text": "no company here"}
    result = Extraction(company=ExtractedField(value="Fake Corp", span="Fake Corp"))
    out = build_extract(FakeGenerative(result))(state)

    assert out["extracted"]["company"] is None


def test_extract_nulls_empty_span() -> None:
    state = {"subject_text": "", "body_text": "some text"}
    result = Extraction(company=ExtractedField(value="X", span=""))
    out = build_extract(FakeGenerative(result))(state)

    assert out["extracted"]["company"] is None
