"""Synthetic labelled dataset generator.

Produces realistic job-application emails with ground-truth labels derived from
the template values, so the labels are never guessed. Deterministic for a seed.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TypedDict

from triage.guard import Action
from triage.labels import CATEGORIES, Category, ExtractedFields, LabeledEmail


class TrackerRow(TypedDict):
    id: int
    company: str
    role: str
    req: str
    via: str


# Synthetic tracker. matched_tracker_id in each email references a row here.
TRACKER: list[TrackerRow] = [
    {"id": 1, "company": "Acme Corp", "role": "AI Engineer", "req": "AC-1001", "via": ""},
    {"id": 2, "company": "Globex", "role": "Software Engineer", "req": "GL-2001", "via": ""},
    {"id": 3, "company": "Initech", "role": "Data Scientist", "req": "IN-3001", "via": ""},
    {"id": 4, "company": "Acme Corp", "role": "AI Engineer", "req": "AC-1002", "via": ""},
    {"id": 5, "company": "Umbrella", "role": "Machine Learning Engineer", "req": "UM-4001", "via": ""},  # noqa: E501
    {"id": 6, "company": "Hooli", "role": "AI Engineer", "req": "HO-5001", "via": "TalentBridge"},
]

RECRUITERS = ["Alex Morgan", "Sam Rivera", "Priya Patel"]
AGENCIES = ["TalentBridge", "TechRecruit", "Raven Talent"]


def _domain(company: str) -> str:
    return "".join(c for c in company.lower() if c.isalnum()) + ".com"


def _row(company: str, role: str, req: str | None = None) -> TrackerRow:
    for row in TRACKER:
        if row["company"] == company and row["role"] == role:
            if req is None or row["req"] == req:
                return row
    raise KeyError(f"no synthetic tracker row for {company} / {role} / {req}")


def _email(
    prefix: Category,
    i: int,
    row: TrackerRow | None,
    from_: str,
    subject: str,
    body: str,
    extracted: ExtractedFields,
    action: Action | None,
) -> LabeledEmail:
    return LabeledEmail(
        email_id=f"syn-{prefix}-{i:03d}",
        from_=from_,
        subject=subject,
        body=body,
        category=prefix,
        extracted=extracted,
        matched_tracker_id=row["id"] if row else None,
        action=action,
    )


def _rejection(rng: random.Random, i: int) -> LabeledEmail:
    company, role, req = rng.choice(
        [
            ("Acme Corp", "AI Engineer", "AC-1001"),
            ("Globex", "Software Engineer", "GL-2001"),
        ]
    )
    row = _row(company, role, req)
    body = (
        f"Dear candidate,\n\nThank you for applying to the {role} role at {company}. "
        "After careful consideration, we have decided not to move forward with your "
        "application at this time. We wish you the best in your search.\n\nRegards,\n"
        f"{company} Talent Team"
    )
    return _email(
        "rejection", i, row,
        f"no-reply@{_domain(company)}",
        f"Your application to {company}",
        body,
        ExtractedFields(company=company, role=role),
        Action.TRACKER_REJECTED,
    )


def _interview_invite(rng: random.Random, i: int) -> LabeledEmail:
    row = _row("Initech", "Data Scientist")
    company, role, req = row["company"], row["role"], row["req"]
    day = rng.randint(10, 28)
    body = (
        f"Hello,\n\nWe would like to invite you to interview for the {role} position at "
        f"{company} (req {req}). The interview will be held on 2026-10-{day:02d} at 14:00 GMT. "
        "Please reply with your availability and any questions.\n\nBest,\nRecruiting Team"
    )
    return _email(
        "interview_invite", i, row,
        f"recruiting@{_domain(company)}",
        f"Interview invitation - {company}",
        body,
        ExtractedFields(
            company=company, role=role, requisition_id=req,
            date_time=f"2026-10-{day:02d}T14:00", timezone="Etc/UTC", contact="Recruiting Team",
        ),
        Action.CALENDAR_EVENT,
    )


def _interview_reschedule(rng: random.Random, i: int) -> LabeledEmail:
    row = _row("Umbrella", "Machine Learning Engineer")
    company, role = row["company"], row["role"]
    body = (
        f"Hi,\n\nWe need to reschedule your interview for the {role} role at {company}. "
        "The new time is 2026-10-15 at 10:30 GMT (previously 09:00). Apologies for the "
        "inconvenience.\n\nThanks,\nScheduling"
    )
    return _email(
        "interview_reschedule", i, row,
        f"scheduling@{_domain(company)}",
        f"Rescheduling your interview - {company}",
        body,
        ExtractedFields(
            company=company, role=role, date_time="2026-10-15T10:30", timezone="Etc/UTC"
        ),
        Action.CALENDAR_CHANGE,
    )


def _assessment(rng: random.Random, i: int) -> LabeledEmail:
    row = _row("Hooli", "AI Engineer")
    company, role, req = row["company"], row["role"], row["req"]
    body = (
        f"Hello,\n\nAs part of your application for {role} at {company} (req {req}), please "
        "complete the online technical assessment. The deadline is 2026-10-20. "
        "Use the link below to begin.\n\nhttps://example.com/assessment\n\nRegards"
    )
    return _email(
        "assessment", i, row,
        f"assessments@{_domain(company)}",
        f"Online assessment - {company}",
        body,
        ExtractedFields(company=company, role=role, requisition_id=req, deadline="2026-10-20"),
        Action.CALENDAR_EVENT,
    )


def _recruiter_outreach(rng: random.Random, i: int) -> LabeledEmail:
    recruiter = rng.choice(RECRUITERS)
    agency = rng.choice(AGENCIES)
    role = "AI Engineer"
    body = (
        f"Hi,\n\nI'm {recruiter}, a recruiter at {agency}. I came across your profile and "
        f"wanted to reach out about a {role} opportunity with one of our clients. Would you "
        "be open to a quick chat this week?\n\nThanks,\n" + recruiter
    )
    return _email(
        "recruiter_outreach", i, None,
        f"{recruiter.split()[0].lower()}@{agency.lower()}.com",
        f"{role} opportunity",
        body,
        ExtractedFields(role=role, agency=agency, contact=recruiter),
        Action.REPLY_DRAFT,
    )


def _offer(rng: random.Random, i: int) -> LabeledEmail:
    row = _row("Acme Corp", "AI Engineer", "AC-1002")
    company, role, req = row["company"], row["role"], row["req"]
    body = (
        f"Dear candidate,\n\nWe are delighted to offer you the {role} position at {company} "
        f"(req {req}). Please find the offer details attached. Let us know if you have any "
        "questions.\n\nCongratulations,\nHR"
    )
    return _email(
        "offer", i, row,
        f"hr@{_domain(company)}",
        f"Offer of employment - {company}",
        body,
        ExtractedFields(company=company, role=role, requisition_id=req),
        Action.TRACKER_OFFER,
    )


def _application_ack(rng: random.Random, i: int) -> LabeledEmail:
    row = _row("Globex", "Software Engineer")
    company, role, req = row["company"], row["role"], row["req"]
    body = (
        f"Hello,\n\nWe have received your application for the {role} role at {company}. "
        f"Your reference number is {req}. We will be in touch.\n\nThanks"
    )
    return _email(
        "application_ack", i, row,
        f"applications@{_domain(company)}",
        f"Application received - {company}",
        body,
        ExtractedFields(company=company, role=role, requisition_id=req),
        Action.ACK,
    )


def _other(rng: random.Random, i: int) -> LabeledEmail:
    body = "Hi there,\n\nJust checking in to see if you had any updates for us.\n\nThanks"
    return _email(
        "other", i, None,
        "unknown@example.com",
        "Checking in",
        body,
        ExtractedFields(),
        None,
    )


GENERATORS = {
    "rejection": _rejection,
    "interview_invite": _interview_invite,
    "interview_reschedule": _interview_reschedule,
    "assessment": _assessment,
    "recruiter_outreach": _recruiter_outreach,
    "offer": _offer,
    "application_ack": _application_ack,
    "other": _other,
}


def _agency_unnamed(rng: random.Random) -> LabeledEmail:
    recruiter = rng.choice(RECRUITERS)
    agency = "Raven Talent"
    body = (
        f"Hi,\n\nI'm {recruiter} from {agency}. We are hiring for an AI Engineer role at an "
        "undisclosed client (a stealth-mode startup). Details on request. Interested?\n\n"
        + recruiter
    )
    return LabeledEmail(
        email_id="syn-special-agency-unnamed",
        from_=f"{recruiter.split()[0].lower()}@raventalent.com",
        subject="AI Engineer opportunity (confidential)",
        body=body,
        category="recruiter_outreach",
        extracted=ExtractedFields(role="AI Engineer", agency=agency, contact=recruiter),
        matched_tracker_id=None,
        action=Action.REPLY_DRAFT,
    )


def _near_duplicate(rng: random.Random) -> LabeledEmail:
    # Two Acme Corp AI Engineer rows (AC-1001 and AC-1002); the email names AC-1002.
    row = _row("Acme Corp", "AI Engineer", "AC-1002")
    body = (
        "Thank you for applying to the AI Engineer role at Acme Corp (req AC-1002). "
        "Unfortunately we will not be moving forward with this application."
    )
    return LabeledEmail(
        email_id="syn-special-near-duplicate",
        from_="no-reply@acmecorp.com",
        subject="Your application to Acme Corp",
        body=body,
        category="rejection",
        extracted=ExtractedFields(
            company="Acme Corp", role="AI Engineer", requisition_id="AC-1002"
        ),
        matched_tracker_id=row["id"],
        action=Action.TRACKER_REJECTED,
    )


def _timezone(rng: random.Random) -> LabeledEmail:
    row = _row("Initech", "Data Scientist")
    body = (
        "Hello, we would like to invite you to interview for the Data Scientist role at "
        "Initech. The interview is on 2026-10-21 at 10:00 PST. Please confirm.\n\nThanks"
    )
    return LabeledEmail(
        email_id="syn-special-timezone",
        from_="recruiting@initech.com",
        subject="Interview invitation - Initech",
        body=body,
        category="interview_invite",
        extracted=ExtractedFields(
            company="Initech", role="Data Scientist",
            date_time="2026-10-21T10:00", timezone="America/Los_Angeles",
        ),
        matched_tracker_id=row["id"],
        action=Action.CALENDAR_EVENT,
    )


def _ambiguous(rng: random.Random) -> LabeledEmail:
    body = "Hello, regarding your recent application, we are not able to proceed at the moment."
    return LabeledEmail(
        email_id="syn-special-ambiguous",
        from_="noreply@careers.example.com",
        subject="Regarding your application",
        body=body,
        category="rejection",
        extracted=ExtractedFields(),
        matched_tracker_id=None,
        action=None,
    )


def generate(out: str | Path, seed: int = 0, n_per_category: int = 10) -> int:
    rng = random.Random(seed)  # noqa: S311
    records: list[LabeledEmail] = []
    for category in CATEGORIES:
        records.extend(GENERATORS[category](rng, i) for i in range(n_per_category))
    records.extend([_agency_unnamed(rng), _near_duplicate(rng), _timezone(rng), _ambiguous(rng)])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    return len(records)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "evals/datasets/synthetic.jsonl"
    print(f"wrote {generate(path)} examples to {path}")
