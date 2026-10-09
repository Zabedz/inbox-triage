"""Metrics per provider from a promptfoo results file, optionally gated on recorded thresholds.

python -m evals.report RESULTS --suite e2e|redteam [--check THRESHOLDS --provider NAME]
"""

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


def e2e_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [(out, row.get("vars") or {}) for row in rows if (out := _output(row)) is not None]
    n = len(scored)
    auto = [(o, v) for o, v in scored if o.get("applied")]
    silent = sum(1 for o, v in auto if scoring.silent_failures(o, v))
    latencies = [float(row.get("latencyMs") or 0) for row in rows]
    return {
        "n": n,
        "distinct_bodies": len({v.get("body") for _, v in scored}),
        "errors": len(rows) - n,
        "category_accuracy": _share(sum(scoring.category_ok(o, v) for o, v in scored), n),
        "match_accuracy": _share(sum(scoring.match_ok(o, v) for o, v in scored), n),
        "action_accuracy": _share(sum(scoring.action_ok(o, v) for o, v in scored), n),
        "approval_rate": _share(sum(o.get("guard_decision") == "approval" for o, _ in scored), n),
        "auto_applied": len(auto),
        "silent_failures": f"{silent}/{len(auto)}",
        "silent_failure_rate": _share(silent, len(auto)),
        "silent_failure_upper95": scoring.wilson_upper(silent, len(auto)),
        "policy_violations": sum(len(scoring.policy_violations(o)) for o, _ in scored),
        "invented_details": sum(
            bool(scoring.invented_details(o.get("draft"), v.get("body", ""))) for o, v in scored
        ),
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95_ms": _p95(latencies) if latencies else 0.0,
    }


def redteam_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: dict[str, list[str]] = {}
    errors: list[str] = []
    leaks = hijacked = violations = 0
    for row in rows:
        vars_ = row.get("vars") or {}
        name = str((row.get("testCase") or {}).get("description") or vars_.get("email_id"))
        out = _output(row)
        if out is None:
            errors.append(name)
            continue
        if unapproved := scoring.unapproved(out, vars_):
            failures[name] = sorted(unapproved)
        leaks += bool(scoring.leaks(out, vars_))
        hijacked += scoring.hijacked(out, vars_)
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
