"""Own parsers for the tracker markdown/TSV/markdown data files."""

from __future__ import annotations

from datetime import date, datetime

from triage.tracker.schema import ActiveInterview, Application, StatusLogRow

APPLICATION_COLUMNS = [
    "#",
    "Date",
    "Company",
    "Via",
    "Role",
    "Location",
    "Score",
    "Status",
    "PDF",
    "Report",
    "Notes",
    "URL",
]

INTERVIEW_COLUMNS = ["Company", "Role", "Round", "Date/Time", "Interviewer", "Status", "Notes"]


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(set(cell) <= {"-", ":", " "} for cell in cells if cell)


def parse_applications(markdown: str) -> list[Application]:
    apps: list[Application] = []
    header_seen = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if not header_seen:
            header_seen = True
            continue
        if _is_separator(cells):
            continue
        if len(cells) != len(APPLICATION_COLUMNS):
            raise ValueError(
                f"tracker row has {len(cells)} cells, expected {len(APPLICATION_COLUMNS)}: "
                f"{cells[:3]!r}"
            )
        values = dict(zip(APPLICATION_COLUMNS, cells, strict=True))
        apps.append(
            Application(
                id=int(values["#"]),
                date_applied=date.fromisoformat(values["Date"]),
                company=values["Company"],
                via=values["Via"],
                role=values["Role"],
                location=values["Location"],
                score=values["Score"],
                status=values["Status"],
                pdf=values["PDF"],
                report=values["Report"],
                notes=values["Notes"],
                url=values["URL"],
            )
        )
    return apps


def parse_status_log(text: str) -> list[StatusLogRow]:
    rows: list[StatusLogRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"status-log row has {len(parts)} fields, expected 6: {line[:60]!r}")
        tracker_id, d, from_, to, source, note = parts
        rows.append(
            StatusLogRow(
                tracker_id=int(tracker_id),
                date=date.fromisoformat(d),
                from_=from_,
                to=to,
                source=source,
                note=note,
            )
        )
    return rows


def _parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    if not value or value == "TBC":
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_active_interviews(markdown: str) -> list[ActiveInterview]:
    rows: list[ActiveInterview] = []
    header_seen = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if not header_seen:
            header_seen = True
            continue
        if _is_separator(cells):
            continue
        if len(cells) != len(INTERVIEW_COLUMNS):
            raise ValueError(
                f"active-interview row has {len(cells)} cells, expected {len(INTERVIEW_COLUMNS)}: "
                f"{cells[:3]!r}"
            )
        values = dict(zip(INTERVIEW_COLUMNS, cells, strict=True))
        rows.append(
            ActiveInterview(
                company=values["Company"],
                role=values["Role"],
                round=values["Round"],
                date_time=_parse_datetime(values["Date/Time"]),
                interviewer=values["Interviewer"],
                status=values["Status"],
                notes=values["Notes"],
            )
        )
    return rows
