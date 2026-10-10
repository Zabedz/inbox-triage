"""Metrics per provider from a promptfoo results file, optionally gated on recorded thresholds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from evals import scoring

FLOORS = {"n", "category_accuracy", "match_accuracy", "action_accuracy"}
CEILINGS = {"silent_failure_rate", "invented_details", "errors", "leaks", "policy_violations"}


def _results(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(Path(path).read_text())["results"]["results"]
    return rows


def _output(row: dict[str, Any]) -> dict[str, Any] | None:
    output = (row.get("response") or {}).get("output")
    if output is None:
        return None
    try:
        return scoring.parse_output(output)
    except json.JSONDecodeError:
        return None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)]


def _share(count: int, n: int) -> float:
    return count / n if n else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def node_profile(outputs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per graph node: mean and p95 milliseconds, and mean tokens."""
    per_node: dict[str, list[float]] = {}
    tokens: dict[str, list[float]] = {}
    for out in outputs:
        for node, ms in (out.get("timings_ms") or {}).items():
            per_node.setdefault(node, []).append(float(ms))
            tokens.setdefault(node, []).append(float((out.get("node_tokens") or {}).get(node, 0)))
    return {
        node: {
            "mean_ms": round(_mean(ms), 2),
            "p95_ms": round(_p95(ms), 2),
            "mean_tokens": round(_mean(tokens[node]), 1),
        }
        for node, ms in per_node.items()
    }


def _total(out: dict[str, Any], key: str) -> float:
    return float(sum(role.get(key, 0) for role in (out.get("tokens") or {}).values()))


def token_usage(outputs: list[dict[str, Any]]) -> dict[str, float]:
    """Mean and p95 tokens and model calls per email."""
    stats: dict[str, float] = {}
    for key, name in (("input_tokens", "input"), ("output_tokens", "output"), ("calls", "calls")):
        values = [_total(out, key) for out in outputs]
        stats[f"{name}_mean"] = round(_mean(values), 1)
        stats[f"{name}_p95"] = _p95(values) if values else 0.0
    return stats


def class_f1(pairs: list[tuple[str | None, str]]) -> dict[str, float]:
    """F1 per labelled category; ``pairs`` are (predicted, expected), None for an error."""
    scores = {}
    for label in sorted({expected for _, expected in pairs}):
        tp = sum(1 for p, e in pairs if p == label and e == label)
        fp = sum(1 for p, e in pairs if p == label and e != label)
        fn = sum(1 for p, e in pairs if p != label and e == label)
        scores[label] = round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0
    return scores


def field_accuracy(
    outputs: list[tuple[dict[str, Any] | None, dict[str, Any]]],
) -> dict[str, float]:
    """Share of emails whose extracted field matches the label, per field; an errored
    email counts as wrong on every field."""
    per_field: dict[str, list[bool]] = {}
    for out, vars_ in outputs:
        expected = vars_.get("extracted") or {}
        if isinstance(expected, str):
            expected = json.loads(expected)
        predicted = (out or {}).get("extracted") or {}
        for field, value in expected.items():
            ok = out is not None and scoring.field_ok(field, predicted.get(field), value)
            per_field.setdefault(field, []).append(ok)
    return {f: round(_share(sum(ok), len(ok)), 3) for f, ok in sorted(per_field.items())}


def e2e_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy is over every row, so an errored email counts as wrong."""
    outputs = [(_output(row), row.get("vars") or {}) for row in rows]
    scored = [(out, v) for out, v in outputs if out is not None]
    n = len(rows)
    auto = [(o, v) for o, v in scored if o.get("applied")]
    silent = sum(1 for o, v in auto if scoring.silent_failures(o, v))
    graph_ms = [sum((o.get("timings_ms") or {}).values()) for o, _ in scored]
    f1 = class_f1([(o.get("category") if o else None, v.get("category", "")) for o, v in outputs])
    return {
        "n": n,
        "distinct_bodies": len({v.get("body") for _, v in outputs}),
        "errors": n - len(scored),
        "category_accuracy": _share(sum(scoring.category_ok(o, v) for o, v in scored), n),
        "macro_f1": round(_mean(list(f1.values())), 3),
        "class_f1": f1,
        "match_accuracy": _share(sum(scoring.match_ok(o, v) for o, v in scored), n),
        "action_accuracy": _share(sum(scoring.action_ok(o, v) for o, v in scored), n),
        "field_accuracy": field_accuracy(outputs),
        "approval_rate": _share(sum(o.get("guard_decision") == "approval" for o, _ in scored), n),
        "auto_applied": len(auto),
        "silent_failures": f"{silent}/{len(auto)}",
        "silent_failure_rate": _share(silent, len(auto)),
        "silent_failure_upper95": scoring.wilson_upper(silent, len(auto)),
        "policy_violations": sum(len(scoring.policy_violations(o)) for o, _ in scored),
        "invented_details": sum(
            bool(scoring.invented_details(o.get("draft"), v.get("body", ""))) for o, v in scored
        ),
        "graph_latency_mean_ms": round(_mean(graph_ms), 2),
        "graph_latency_p95_ms": round(_p95(graph_ms), 2) if graph_ms else 0.0,
        "tokens_per_email": token_usage([o for o, _ in scored]),
        "nodes": node_profile([o for o, _ in scored]),
    }


def redteam_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: dict[str, list[str]] = {}
    errors: list[str] = []
    leaks = hijacked = caught = violations = 0
    for row in rows:
        vars_ = row.get("vars") or {}
        name = str((row.get("testCase") or {}).get("description") or vars_.get("email_id"))
        out = _output(row)
        if out is None:
            errors.append(name)
            continue
        unapproved = scoring.unapproved(out, vars_)
        if unapproved:
            failures[name] = sorted(unapproved)
        swayed = scoring.hijacked(out, vars_)
        leaks += bool(scoring.leaks(out, vars_))
        hijacked += swayed
        caught += swayed and not unapproved
        violations += len(scoring.policy_violations(out))
    return {
        "n": len(rows),
        "errors": len(errors),
        "error_cases": sorted(errors),
        "unapproved": len(failures),
        "failures": dict(sorted(failures.items())),
        "leaks": leaks,
        "policy_violations": violations,
        "hijacked": hijacked,
        "hijacked_caught": caught,
    }


def metrics_by_provider(path: str | Path, suite: str) -> dict[str, dict[str, Any]]:
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for row in _results(path):
        by_provider.setdefault(scoring.provider_id(row), []).append(row)
    compute = e2e_metrics if suite == "e2e" else redteam_metrics
    return {provider: compute(rows) for provider, rows in sorted(by_provider.items())}


def _known_failures(metrics: dict[str, Any], known: dict[str, list[str]]) -> list[str]:
    problems = []
    for case, applied in metrics["failures"].items():
        if case not in known:
            problems.append(f"new red-team failure {case}: {', '.join(applied)}")
        elif sorted(known[case]) != applied:
            problems.append(f"red-team failure {case} changed: {applied} (recorded {known[case]})")
    return problems


def check(metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Return gate violations. A null floor or ceiling is not yet recorded and is skipped;
    a null ``known_failures`` means none are accepted."""
    problems: list[str] = []
    for name, limit in thresholds.items():
        if name == "known_failures" and "failures" in metrics:
            problems += _known_failures(metrics, limit or {})
        elif limit is None:
            continue
        elif name not in metrics:
            problems.append(f"threshold {name!r} is not reported for this suite")
        elif name in FLOORS:
            if metrics[name] < limit:
                problems.append(f"{name}={metrics[name]} below floor {limit}")
        elif name in CEILINGS:
            if metrics[name] > limit:
                problems.append(f"{name}={metrics[name]} above ceiling {limit}")
        else:
            problems.append(f"unknown threshold {name!r}")
    return problems


def _format(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.report")
    parser.add_argument("results")
    parser.add_argument("--suite", choices=["e2e", "redteam"], required=True)
    parser.add_argument("--check", type=Path, help="thresholds JSON; exit 1 on a violation")
    parser.add_argument("--provider", help="with --check: the provider that must be present")
    args = parser.parse_args(argv)
    if args.check and not args.provider:
        parser.error("--check needs --provider")

    by_provider = metrics_by_provider(args.results, args.suite)
    for provider, metrics in by_provider.items():
        print(f"== {args.suite} / {provider} ==")
        for name, value in metrics.items():
            print(f"  {name}: {_format(value)}")
    if not args.check:
        return 0

    limits = json.loads(args.check.read_text())[args.suite].get(args.provider)
    problems: list[str] = []
    if args.provider not in by_provider:
        problems.append(f"no results for provider {args.provider!r}")
    elif limits is None:
        problems.append(f"no thresholds recorded for {args.provider!r}")
    else:
        problems = check(by_provider[args.provider], limits)
        if args.suite == "redteam":
            seen = by_provider[args.provider]
            fixed = set(limits.get("known_failures") or {}) - set(seen["failures"])
            fixed -= set(seen["error_cases"])
            for case in sorted(fixed):
                print(f"  now passing, remove from known_failures: {case}")
    for problem in problems:
        print(f"  FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
