from __future__ import annotations

import json

import pytest

from evals.summary import apply, main, render


def result(provider: str, out: dict, vars_: dict, scores: dict | None = None) -> dict:
    response = {"output": json.dumps(out), "namedScores": scores or {}}
    return {"provider": {"id": provider}, "vars": vars_, "response": response, "latencyMs": 5}


def out(category="rejection", matched=1, action="tracker_rejected", guard="auto", **extra) -> dict:
    applied = [action] if guard == "auto" else []
    return {"category": category, "matched_id": matched, "action": action, "guard_decision": guard,
            "applied": applied, "draft": None, "match_prob": 1.0, **extra}


def vars_for(email_id: str) -> dict:
    return {"email_id": email_id, "category": "rejection", "matched_tracker_id": "1",
            "action": "tracker_rejected", "body": f"body {email_id}",
            "extracted": {"company": "Acme Corp", "agency": None}}


TIMED = {"timings_ms": {"classify": 40.0, "extract": 10.0, "match": 90.0}, "node_tokens": {"match": 320},
         "tokens": {"decision": {"calls": 2, "input_tokens": 300, "output_tokens": 20}},
         "extracted": {"company": "Acme Corp"}}


def write(directory, name: str, rows: list[dict]) -> None:
    (directory / name).write_text(json.dumps({"results": {"results": rows}}))


@pytest.fixture
def results(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    e2e = [
        result("rules", out(), vars_for("a")),
        result("jev", out(masked_spans=2, **TIMED), vars_for("a"), {"p_rejection": 0.9}),
        result("jev", out(masked_spans=0, **TIMED), vars_for("b"), {"p_rejection": 0.9}),
        result("jev-raw", out(matched=2, masked_spans=2, **{**TIMED, "extracted": {"agency": "Hays"}}),
               vars_for("a")),
        result("jev-raw", out(masked_spans=0, **TIMED), vars_for("b")),
    ]
    write(d, "e2e-all.json", e2e)
    rt = [result("rules", out(applied=["tracker_rejected"]),
                 {"attack": "swaying", "allowed_auto": "label", "category": "other"}, None)]
    rt[0]["testCase"] = {"description": "sway-case"}
    write(d, "redteam-rules.json", rt)
    return d


def test_render_tables(results) -> None:
    blocks = render(results)
    text = blocks["results"]

    assert "| rules | 1 (1) | 0 | 100.0% | 1.000 | 100.0% | 100.0% | 0.0% | 1 | 0/1 (" in text
    assert "| llm | not run |" in text
    assert "| jev | 0/0 (0/2) | 2/2 (0/0) |" in text
    assert "| jev-raw | 0/0 (1/2) | 1/2 (0/0) |" in text
    assert "| rules | 0/0 (0/0) | 0/0 (0/0) |" in text  # no timings, so extract never ran
    assert "| rules | 1 | 1 | 0 | 0 | `sway-case` |" in text
    assert "| jev | 1 | 0 | 1 | 0 |" in text  # raw run wrong on the masked email only
    assert "| jev | 2 | 0.100 |" in text
    assert "| jev | 2.0 | 300 / 20 (300 / 20) | 140 (140) | match (90) |" in blocks["cost"]


def test_write_then_check(results, tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- results:start -->\nold\n<!-- results:end -->\n<!-- cost:start -->\n<!-- cost:end -->\n")

    assert main(["--results", str(results), "--check", str(readme)]) == 1
    assert main(["--results", str(results), "--write", str(readme)]) == 0
    assert main(["--results", str(results), "--check", str(readme)]) == 0


def test_check_ignores_rows_without_results(results, tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n<!-- cost:start -->\n<!-- cost:end -->\n")
    main(["--results", str(results), "--write", str(readme)])
    text = readme.read_text().replace("| llm | not run |", "| llm | 96 (36) | 0 | 80.0% |", 1)
    readme.write_text(text)

    assert main(["--results", str(results), "--check", str(readme)]) == 0
    readme.write_text(text.replace("| rules | 1 (1) |", "| rules | 2 (1) |"))
    assert main(["--results", str(results), "--check", str(readme)]) == 1


def test_provider_in_two_files_is_an_error(results) -> None:
    write(results, "redteam.json", json.loads((results / "redteam-rules.json").read_text())["results"]["results"])
    with pytest.raises(ValueError, match="rules is in"):
        render(results)


@pytest.mark.parametrize("readme", ["no markers", "<!-- results:start -->\n<!-- results:end -->\n" * 2])
def test_apply_needs_exactly_one_block(readme) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        apply(readme, {"results": "x"})


def test_check_fails_without_rules_results(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n<!-- cost:start -->\n<!-- cost:end -->\n")
    assert main(["--results", str(empty), "--check", str(readme)]) == 1


def test_check_covers_field_rows(results, tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n<!-- cost:start -->\n<!-- cost:end -->\n")
    main(["--results", str(results), "--write", str(readme)])
    text = readme.read_text()
    readme.write_text(text.replace("| jev | 0/0 (0/2) | 2/2 (0/0) |", "| jev | 0/0 (0/2) | 1/2 (0/0) |"))

    assert main(["--results", str(results), "--check", str(readme)]) == 1
