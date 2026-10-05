"""Command-line entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from triage.config import load
from triage.gmail.ingest import ingest
from triage.tracker.snapshot import snapshot


def _cmd_snapshot(args: argparse.Namespace) -> int:
    cfg = load()
    count = snapshot(cfg)
    print(f"snapshot: {count} applications written to {cfg.tracker.db}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg = load()
    ingest(cfg, dry_run=args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="copy tracker into local SQLite")
    snap.set_defaults(func=_cmd_snapshot)

    ing = sub.add_parser("ingest", help="list/fetch job-related Gmail")
    ing.add_argument("--dry-run", action="store_true", help="print message IDs only")
    ing.set_defaults(func=_cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
