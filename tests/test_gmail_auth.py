from __future__ import annotations

import json

from triage.config import GMAIL_MODIFY
from triage.gmail.ingest import load_token

READONLY = "https://www.googleapis.com/auth/gmail.readonly"


def write_token(path, scopes: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "token": "t",
                "refresh_token": "r",
                "client_id": "c",
                "client_secret": "s",
                "scopes": scopes,
                "expiry": "2999-01-01T00:00:00Z",
            }
        )
    )


def test_readonly_token_is_rejected(tmp_path) -> None:
    token = tmp_path / "token.json"
    write_token(token, [READONLY])
    assert load_token(token, (GMAIL_MODIFY,)) is None


def test_token_with_scope_is_loaded(tmp_path) -> None:
    token = tmp_path / "token.json"
    write_token(token, [GMAIL_MODIFY])
    creds = load_token(token, (GMAIL_MODIFY,))
    assert creds is not None and creds.valid


def test_missing_token(tmp_path) -> None:
    assert load_token(tmp_path / "none.json", (GMAIL_MODIFY,)) is None
