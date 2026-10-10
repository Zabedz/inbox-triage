"""README results tables from promptfoo result files in evals/results.

The ``results`` block is deterministic for the rules baseline and is checked; rows that read
"not run" match anything, so CI can check the rules rows alone. The ``cost`` block holds
wall-clock and token numbers and is only written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from evals import calibrate, report, scoring

RESULTS = Path("evals/results")
PROVIDERS = ["rules", "jev", "jev-raw", "llm", "llm-raw"]
BACKENDS = ["jev", "llm"]
NOT_RUN = "not run"

Rows = dict[str, list[dict[str, Any]]]


def load(paths: list[Path]) -> Rows:
    """Rows per provider; raises ValueError when a provider appears in two files."""
    rows: Rows = {}
    source: dict[str, Path] = {}
    for path in sorted(paths):
        for row in json.loads(path.read_text())["results"]["results"]:
            provider = scoring.provider_id(row)
            if source.setdefault(provider, path) != path:
                raise ValueError(f"{provider} is in {source[provider]} and {path}; delete one")
            rows.setdefault(provider, []).append(row)
    return rows


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _not_run(provider: str, columns: int) -> str:
    return f"| {provider} | {NOT_RUN} |" + " |" * (columns - 2)


def e2e_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Provider | n (distinct) | Errors | Category | Macro F1 | Match | Action "
        "| Sent to review | Auto-applied | Silent failures (95% upper) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for provider in PROVIDERS:
        m = metrics.get(provider)
        if m is None:
            lines.append(_not_run(provider, 10))
            continue
        lines.append(
            f"| {provider} | {m['n']} ({m['distinct_bodies']}) | {m['errors']} "
            f"| {_pct(m['category_accuracy'])} | {m['macro_f1']:.3f} "
            f"| {_pct(m['match_accuracy'])} | {_pct(m['action_accuracy'])} "
            f"| {_pct(m['approval_rate'])} | {m['auto_applied']} "
            f"| {m['silent_failures']} ({_pct(m['silent_failure_upper95'])}) |"
        )
    return lines


def field_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    fields = sorted({f for m in metrics.values() for f in m["field_accuracy"]})
    lines = [
        "| Provider | " + " | ".join(fields) + " |",
        "|---|" + "---|" * len(fields),
    ]
    for provider in PROVIDERS:
        m = metrics.get(provider)
        if m is None:
            lines.append(_not_run(provider, len(fields) + 1))
            continue
        cells = [_pct(m["field_accuracy"].get(f, 0.0)) for f in fields]
        lines.append(f"| {provider} | " + " | ".join(cells) + " |")
    return lines


def redteam_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Provider | Cases | Unapproved actions | Errors | Hijacked, caught by review "
        "| Failing cases |",
        "|---|---|---|---|---|---|",
    ]
    for provider in PROVIDERS:
        if provider.endswith("-raw"):
            continue
        m = metrics.get(provider)
        if m is None:
            lines.append(_not_run(provider, 6))
            continue
        failing = ", ".join(f"`{case}`" for case in m["failures"]) or "-"
        lines.append(
            f"| {provider} | {m['n']} | {m['unapproved']} | {m['errors']} "
            f"| {m['hijacked_caught']} | {failing} |"
        )
    return lines


def _case_ok(row: dict[str, Any]) -> tuple[bool, int] | None:
    out = report._output(row)
    if out is None:
        return None
    vars_ = row.get("vars") or {}
    ok = scoring.category_ok(out, vars_) and scoring.match_ok(out, vars_)
    return ok and scoring.action_ok(out, vars_), int(out.get("masked_spans") or 0)


def redaction_table(rows: Rows) -> list[str]:
    """Paired by email: discordant outcomes where redaction changed the text, against the
    same count where it changed nothing (the noise floor)."""
    lines = [
        "| Backend | Emails redaction changes | Wrong only redacted | Wrong only raw "
        "| Discordant where unchanged (noise) |",
        "|---|---|---|---|---|",
    ]
    for backend in BACKENDS:
        redacted, raw = rows.get(backend), rows.get(f"{backend}-raw")
        if not redacted or not raw:
            lines.append(_not_run(backend, 5))
            continue
        raw_by_id = {(r.get("vars") or {}).get("email_id"): r for r in raw}
        changed = only_redacted = only_raw = noise = 0
        for row in redacted:
            pair = raw_by_id.get((row.get("vars") or {}).get("email_id"))
            a, b = _case_ok(row), _case_ok(pair) if pair else None
            if a is None or b is None:
                continue
            if a[1]:
                changed += 1
                only_redacted += int(not a[0] and b[0])
                only_raw += int(a[0] and not b[0])
            else:
                noise += int(a[0] != b[0])
        lines.append(f"| {backend} | {changed} | {only_redacted} | {only_raw} | {noise} |")
    return lines


def calibration_table(rows: Rows) -> list[str]:
    lines = [
        "| Backend | n | Category ECE | Category Brier | Match ECE (top label) |",
        "|---|---|---|---|---|",
    ]
    for backend in BACKENDS:
        backend_rows = rows.get(backend)
        if not backend_rows:
            lines.append(_not_run(backend, 5))
            continue
        probs, labels, confidences, correct = [], [], [], []
        for row in backend_rows:
            out = report._output(row)
            if out is None:
                continue
            vars_ = row.get("vars") or {}
            named = (row.get("response") or {}).get("namedScores") or {}
            if category_probs := {k[2:]: v for k, v in named.items() if k.startswith("p_")}:
                probs.append(category_probs)
                labels.append(vars_.get("category", ""))
            match_prob = float(out.get("match_prob") or 0.0)
            confidences.append(match_prob if out.get("matched_id") is not None else 1 - match_prob)
            correct.append(scoring.match_ok(out, vars_))
        options = sorted({o for p in probs for o in p} | set(labels))
        category = (
            f"{calibrate.ece(probs, labels):.3f} | {calibrate.brier(probs, labels, options):.3f}"
            if probs
            else "- | -"
        )
        lines.append(
            f"| {backend} | {len(confidences)} | {category} "
            f"| {calibrate.top_label_ece(confidences, correct):.3f} |"
        )
    return lines


def cost_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Backend | Model calls per email | Tokens in / out per email (mean, p95) "
        "| Graph ms per email (mean, p95) | Slowest node (p95 ms) |",
        "|---|---|---|---|---|",
    ]
    for backend in BACKENDS:
        m = metrics.get(backend)
        if m is None:
            lines.append(_not_run(backend, 5))
            continue
        t, nodes = m["tokens_per_email"], m["nodes"]
        slowest = max(nodes, key=lambda n: nodes[n]["p95_ms"]) if nodes else None
        slow = f"{slowest} ({nodes[slowest]['p95_ms']:.0f})" if slowest else "-"
        lines.append(
            f"| {backend} | {t['calls_mean']:.1f} "
            f"| {t['input_mean']:.0f} / {t['output_mean']:.0f} "
            f"({t['input_p95']:.0f} / {t['output_p95']:.0f}) "
            f"| {m['graph_latency_mean_ms']:.0f} ({m['graph_latency_p95_ms']:.0f}) | {slow} |"
        )
    return lines


def render(results: Path = RESULTS) -> dict[str, str]:
    e2e_rows = load(list(results.glob("e2e*.json")))
    redteam_rows = load(list(results.glob("redteam*.json")))
    e2e = {p: report.e2e_metrics(rows) for p, rows in e2e_rows.items()}
    redteam = {p: report.redteam_metrics(rows) for p, rows in redteam_rows.items()}
    sections = [
        ("### End to end (synthetic set)", e2e_table(e2e)),
        ("### Extracted fields", field_table(e2e)),
        ("### Red team", redteam_table(redteam)),
        ("### Redaction cost (paired by email)", redaction_table(e2e_rows)),
        ("### Calibration", calibration_table(e2e_rows)),
    ]
    body = "\n\n".join(f"{title}\n\n" + "\n".join(table) for title, table in sections)
    return {"results": body, "cost": "\n".join(cost_table(e2e))}


def _block(readme: str, name: str) -> re.Match[str]:
    pattern = re.compile(rf"(<!-- {name}:start -->\n)(.*?)(<!-- {name}:end -->)", re.DOTALL)
    found = list(pattern.finditer(readme))
    if len(found) != 1:
        raise ValueError(f"README needs exactly one <!-- {name}:start --> ... end block")
    return found[0]


def apply(readme: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        found = _block(readme, name)
        readme = readme[: found.start(2)] + body + "\n" + readme[found.end(2) :]
    return readme


def stale_lines(readme: str, rendered: str) -> list[str]:
    """Rendered lines missing from the README block; "not run" rows match anything."""
    present = set(_block(readme, "results")[2].splitlines())
    if not any(line.startswith("| rules |") for line in rendered.splitlines()):
        return ["no rules results in evals/results; run npm run eval:ci"]
    return [line for line in rendered.splitlines() if NOT_RUN not in line and line not in present]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.summary")
    parser.add_argument("--results", type=Path, default=RESULTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", type=Path, help="rewrite the README blocks")
    mode.add_argument("--check", type=Path, help="exit 1 if rows with results are stale")
    args = parser.parse_args(argv)

    blocks = render(args.results)
    if args.write:
        args.write.write_text(apply(args.write.read_text(), blocks))
        return 0
    if args.check:
        stale = stale_lines(args.check.read_text(), blocks["results"])
        for line in stale:
            print(f"stale or missing in {args.check}: {line}")
        return 1 if stale else 0
    print(blocks["results"])
    print()
    print(blocks["cost"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
