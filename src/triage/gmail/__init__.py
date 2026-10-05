"""Read-only Gmail access."""

from triage.gmail.client import Email, GmailClient
from triage.gmail.ingest import build_service, ingest

__all__ = ["Email", "GmailClient", "build_service", "ingest"]
