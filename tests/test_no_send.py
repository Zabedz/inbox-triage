from __future__ import annotations

import ast
import re
from pathlib import Path

from triage.gmail.client import GmailClient
from triage.gmail.labels import apply_label

ROOT = Path(__file__).resolve().parent.parent

# gmail.modify permits these; nothing here may call them, directly or by name.
FORBIDDEN_NAMES = {
    "send",
    "drafts",
    "trash",
    "untrash",
    "batchDelete",
    "batchModify",
    "import_",
    "sendmail",
    "send_message",
    "forwardingAddresses",
    "autoForwarding",
}
# Gmail delete and insert; no other object in src or evals uses these names.
METHOD_NAMES = {"delete", "insert"}
FORBIDDEN_TEXT = re.compile(
    r"gmail\.send|gmail\.compose|mail\.google\.com|/send\b|/drafts\b|\bTRASH\b|\bSPAM\b"
)
ALLOWED_CALLS = {
    ("users", "messages", "list"),
    ("users", "messages", "get"),
    ("users", "messages", "modify"),
    ("users", "labels", "list"),
    ("users", "labels", "create"),
}


def _python_hits(path: Path) -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NAMES | METHOD_NAMES:
                hits.append(f"{path.relative_to(ROOT)}:{line}: .{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in FORBIDDEN_NAMES or FORBIDDEN_TEXT.search(node.value):
                hits.append(f"{path.relative_to(ROOT)}:{line}: {node.value[:60]!r}")
    return hits


def test_no_mail_sending_or_deleting_code() -> None:
    hits = [hit for top in ("src", "evals") for p in (ROOT / top).rglob("*.py") for hit in _python_hits(p)]
    assert hits == []


def test_no_mail_sending_in_configs_or_scripts() -> None:
    paths = [*(ROOT / "evals").rglob("*.yaml"), *(ROOT / "evals").rglob("*.sh"), ROOT / "package.json"]
    hits = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in paths
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if FORBIDDEN_TEXT.search(line) or re.search(r"\bcurl\b|https?://[^\s\"']*googleapis", line)
    ]
    assert hits == []


class RecordingService:
    """Fake Gmail resource that records every call path and modify body."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.modify_bodies: list[dict] = []
        self._path: tuple[str, ...] = ()

    def __getattr__(self, name: str):
        def step(**kwargs):
            self._path = (*self._path, name)
            if name in {"list", "get", "modify", "create"}:
                self.calls.append(self._path)
                if name == "modify":
                    self.modify_bodies.append(kwargs["body"])
                self._path = ()
                return _Result(name)
            return self

        return step


class _Result:
    def __init__(self, name: str) -> None:
        self._name = name

    def execute(self) -> dict:
        if self._name == "list":
            return {"labels": [{"id": "L1", "name": "Job Applications"}], "messages": [{"id": "m1"}]}
        if self._name == "get":
            return {"id": "m1", "threadId": "t1", "payload": {"headers": []}}
        return {"id": "L1"}


def test_runtime_gmail_calls_are_read_and_add_label_only() -> None:
    service = RecordingService()
    client = GmailClient(service)
    client.get_email(client.list_job_messages("q")[0])
    apply_label(service, "m1")

    assert set(service.calls) <= ALLOWED_CALLS
    assert service.modify_bodies == [{"addLabelIds": ["L1"]}]
