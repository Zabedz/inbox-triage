"""Configuration, loaded from environment (optionally a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_JEV_ENDPOINT = "https://opencode.ai/zen/v1/systemone"
DEFAULT_JEV_MODEL = "jev-1.13"

DEFAULT_TRACKER_APPLICATIONS = "~/Projects/career-ops/data/applications.md"
DEFAULT_TRACKER_STATUS_LOG = "~/Projects/career-ops/data/status-log.tsv"
DEFAULT_TRACKER_ACTIVE_INTERVIEWS = "~/Projects/career-ops/data/active-interviews.md"

DEFAULT_DATA_DIR = "./data"
DEFAULT_GMAIL_TOKEN = "./data/gmail-token.json"  # noqa: S105

DEFAULT_QUERY = "is:unread newer_than:7d"


@dataclass(frozen=True)
class JevConfig:
    endpoint: str
    api_key: str
    model: str
    timeout_s: float = 30.0


@dataclass(frozen=True)
class TrackerConfig:
    source_applications: Path
    source_status_log: Path
    source_active_interviews: Path
    data_dir: Path
    snapshot_md: Path
    db: Path


@dataclass(frozen=True)
class GmailConfig:
    token_path: Path
    scopes: tuple[str, ...]
    query: str


@dataclass(frozen=True)
class Config:
    jev: JevConfig
    tracker: TrackerConfig
    gmail: GmailConfig
    llm_provider: str = ""
    anthropic_model: str = ""
    anthropic_api_key: str = ""


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def load(env_path: str | Path | None = None) -> Config:
    if env_path is not None:
        load_dotenv(env_path)
    else:
        load_dotenv()

    data_dir = _path("DATA_DIR", DEFAULT_DATA_DIR)

    jev = JevConfig(
        endpoint=os.environ.get("JEV_ENDPOINT", DEFAULT_JEV_ENDPOINT),
        api_key=os.environ.get("OPENCODE_API_KEY", ""),
        model=os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL),
    )

    tracker = TrackerConfig(
        source_applications=_path("TRACKER_APPLICATIONS", DEFAULT_TRACKER_APPLICATIONS),
        source_status_log=_path("TRACKER_STATUS_LOG", DEFAULT_TRACKER_STATUS_LOG),
        source_active_interviews=_path(
            "TRACKER_ACTIVE_INTERVIEWS", DEFAULT_TRACKER_ACTIVE_INTERVIEWS
        ),
        data_dir=data_dir,
        snapshot_md=data_dir / "applications.snapshot.md",
        db=data_dir / "tracker.db",
    )

    gmail = GmailConfig(
        token_path=_path("GMAIL_TOKEN", DEFAULT_GMAIL_TOKEN),
        scopes=("https://www.googleapis.com/auth/gmail.readonly",),
        query=os.environ.get("GMAIL_QUERY", DEFAULT_QUERY),
    )

    return Config(
        jev=jev,
        tracker=tracker,
        gmail=gmail,
        llm_provider=os.environ.get("LLM_PROVIDER", ""),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
