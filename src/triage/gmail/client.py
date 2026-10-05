"""Thin Gmail client (read-only) over an injected google service resource."""

from __future__ import annotations

import base64
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from pydantic import BaseModel


class Email(BaseModel):
    id: str
    thread_id: str
    from_: str
    to: str
    subject: str
    date: datetime | None
    body_plain: str
    body_html: str
    snippet: str


def _decode(data: str | None) -> str:
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return ""


def _find_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        parts.append(payload)
    if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data"):
        parts.append(payload)
    for child in payload.get("parts", []):
        parts.extend(_find_parts(child))
    return parts


def _parse_message(raw: dict[str, Any]) -> Email:
    payload = raw.get("payload", {})
    headers = {h.get("name"): h.get("value", "") for h in payload.get("headers", [])}

    plain, html = "", ""
    for part in _find_parts(payload):
        data = _decode(part.get("body", {}).get("data"))
        if part.get("mimeType") == "text/plain" and not plain:
            plain = data
        elif part.get("mimeType") == "text/html" and not html:
            html = data

    date: datetime | None = None
    if headers.get("Date"):
        try:
            date = parsedate_to_datetime(headers["Date"])
        except (TypeError, ValueError):
            date = None

    return Email(
        id=raw.get("id", ""),
        thread_id=raw.get("threadId", ""),
        from_=headers.get("From", ""),
        to=headers.get("To", ""),
        subject=headers.get("Subject", ""),
        date=date,
        body_plain=plain,
        body_html=html,
        snippet=raw.get("snippet", ""),
    )


class GmailClient:
    def __init__(self, service: object) -> None:
        self._service = service

    def list_job_messages(self, query: str, after: datetime | None = None) -> list[str]:
        q = query
        if after is not None:
            q = f"{q} after:{int(after.timestamp())}"
        ids: list[str] = []
        page_token: str | None = None
        while True:
            request = self._service.users().messages().list(  # type: ignore[attr-defined]
                userId="me", q=q, pageToken=page_token
            )
            result = request.execute()
            ids.extend(m["id"] for m in result.get("messages", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return ids

    def get_email(self, message_id: str) -> Email:
        raw = (
            self._service.users()  # type: ignore[attr-defined]
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return _parse_message(raw)
