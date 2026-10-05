"""Gmail OAuth service construction and ingestion entry points."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from triage.config import Config
from triage.gmail.client import Email, GmailClient

logger = logging.getLogger("triage.gmail.ingest")

CLIENT_SECRETS = "client_secrets.json"


def build_service(cfg: Config) -> object:
    """Build a gmail.readonly service using an installed-app OAuth flow.

    Reads `client_secrets.json` from the data dir and stores the refreshed token at
    `cfg.gmail.token_path`. Never requests a send scope.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    token_path: Path = cfg.gmail.token_path
    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(  # type: ignore[no-untyped-call]
            str(token_path), cfg.gmail.scopes
        )
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # type: ignore[no-untyped-call]
        else:
            secrets = cfg.tracker.data_dir / CLIENT_SECRETS
            if not secrets.exists():
                raise FileNotFoundError(
                    f"missing OAuth client secrets at {secrets}; see README setup"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), cfg.gmail.scopes)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        os.chmod(token_path, 0o600)

    return build("gmail", "v1", credentials=creds)


def auth(cfg: Config) -> None:
    build_service(cfg)
    print(f"authorised; token written to {cfg.gmail.token_path}")


def ingest(
    cfg: Config, dry_run: bool = False, out: str | Path | None = None
) -> list[Email]:
    service = build_service(cfg)
    client = GmailClient(service)
    ids = client.list_job_messages(cfg.gmail.query)
    if dry_run:
        for mid in ids:
            print(mid)
        return []
    emails = [client.get_email(mid) for mid in ids]
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for email in emails:
                fh.write(email.model_dump_json() + "\n")
        print(f"wrote {len(emails)} emails to {out}")
    else:
        for email in emails:
            print(f"{email.id}\t{email.date}\t{email.from_}\t{email.subject}")
    return emails
