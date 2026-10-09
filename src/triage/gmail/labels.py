"""Apply a Gmail label to a message (gmail.modify scope, no send)."""

from __future__ import annotations

LABEL_NAME = "Job Applications"


def _find_or_create_label(service: object, name: str) -> str:
    result = service.users().labels().list(userId="me").execute()  # type: ignore[attr-defined]
    for label in result.get("labels", []):
        if label["name"] == name:
            return str(label["id"])
    created = (
        service.users()  # type: ignore[attr-defined]
        .labels()
        .create(userId="me", body={"name": name, "labelListVisibility": "labelShow"})
        .execute()
    )
    return str(created["id"])


def apply_label(service: object, email_id: str, name: str = LABEL_NAME) -> None:
    label_id = _find_or_create_label(service, name)
    service.users().messages().modify(  # type: ignore[attr-defined]
        userId="me", id=email_id, body={"addLabelIds": [label_id]}
    ).execute()
