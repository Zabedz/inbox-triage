"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from triage.config import load
from triage.gmail.ingest import auth, ingest
from triage.label_cli import label, load_emails
from triage.review import review
from triage.run import run
from triage.stats import render, stats
from triage.tracker.db import TrackerDB
from triage.tracker.snapshot import snapshot


def _cmd_snapshot(args: argparse.Namespace) -> int:
    cfg = load()
    count = snapshot(cfg)
    print(f"snapshot: {count} applications written to {cfg.tracker.db}")
    return 0


def _cmd_auth(args: argparse.Namespace) -> int:
    cfg = load()
    auth(cfg)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg = load()
    ingest(cfg, dry_run=args.dry_run, out=args.out)
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    cfg = load()
    emails = load_emails(args.in_file)
    tracker = TrackerDB(cfg.tracker.db)
    try:
        count = label(emails, tracker, args.out)
    finally:
        tracker.close()
    print(f"labelled {count} emails (total now in {args.out})")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load()
    pending = run(cfg)
    print(f"run complete: {len(pending)} emails awaiting approval")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    cfg = load()
    handled = review(cfg)
    print(f"reviewed {handled} approvals")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    cfg = load()
    summary = stats(cfg.agent_db, days=args.days)
    if not summary["emails"] and not summary["queued"] and not summary["error_rows"]:
        print(f"no decisions in {cfg.agent_db}; run triage run first")
        return 1
    print(render(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="copy tracker into local SQLite")
    snap.set_defaults(func=_cmd_snapshot)

    auth_p = sub.add_parser("auth", help="authorise Gmail (gmail.modify)")
    auth_p.set_defaults(func=_cmd_auth)

    ing = sub.add_parser("ingest", help="list/fetch job-related Gmail")
    ing.add_argument("--dry-run", action="store_true", help="print message IDs only")
    ing.add_argument("--out", type=Path, help="write emails to JSONL")
    ing.set_defaults(func=_cmd_ingest)

    lab = sub.add_parser("label", help="label emails against the tracker")
    lab.add_argument("in_file", type=Path, help="JSONL of emails (from ingest --out)")
    lab.add_argument("--out", type=Path, required=True, help="output JSONL of labels")
    lab.set_defaults(func=_cmd_label)

    run_p = sub.add_parser("run", help="run the triage graph over new email")
    run_p.set_defaults(func=_cmd_run)

    rev_p = sub.add_parser("review", help="review pending approvals")
    rev_p.set_defaults(func=_cmd_review)

    stats_p = sub.add_parser("stats", help="summarise real use from the decision log")
    stats_p.add_argument("--days", type=int, help="only decisions logged in the last N days")
    stats_p.set_defaults(func=_cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
