"""Deterministic redaction of personal data before external model calls."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\b")
_PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class Redaction(BaseModel):
    text: str
    replacements: dict[str, str]


def _find_phones(text: str) -> list[re.Match[str]]:
    matches = []
    for match in _PHONE_CANDIDATE.finditer(text):
        span = match.group(0)
        if sum(c.isdigit() for c in span) < 10:
            continue
        if _ISO_DATE_RE.match(span):
            continue
        matches.append(match)
    return matches


def _mask_spans(
    text: str, matches: list[re.Match[str]], prefix: str, replacements: dict[str, str]
) -> str:
    result = text
    tokens = [f"[{prefix}_{i + 1}]" for i in range(len(matches))]
    for match, token in zip(matches, tokens, strict=True):
        replacements[token] = match.group(0)
    for match, token in zip(reversed(matches), reversed(tokens), strict=True):
        result = result[: match.start()] + token + result[match.end() :]
    return result


def _non_overlapping(matches: Sequence[re.Match[str]]) -> list[re.Match[str]]:
    ordered = sorted(matches, key=lambda m: (m.start(), -(m.end() - m.start())))
    kept: list[re.Match[str]] = []
    last_end = -1
    for match in ordered:
        if match.start() < last_end:
            continue
        kept.append(match)
        last_end = match.end()
    return kept


def redact(
    text: str, names: Sequence[str] = (), keep: Sequence[str] = ()
) -> Redaction:
    """Mask emails, phone numbers, and known personal names.

    Emails and phones are masked first, so a ``keep`` literal that happens to be a
    substring of an address can never leak it. ``keep`` then protects company names
    from name-masking. Sender domains survive separately in graph state.
    """
    replacements: dict[str, str] = {}
    result = _mask_spans(text, list(EMAIL_RE.finditer(text)), "EMAIL", replacements)
    result = _mask_spans(result, _find_phones(result), "PHONE", replacements)

    protected = result
    sentinels: dict[str, str] = {}
    for i, literal in enumerate(keep):
        if literal:
            sentinel = f"\x00KEEP{i}\x00"
            sentinels[sentinel] = literal
            protected = protected.replace(literal, sentinel)

    name_matches = [
        match
        for name in names
        if name
        for match in re.finditer(rf"\b{re.escape(name)}\b", protected, re.IGNORECASE)
    ]
    result = _mask_spans(protected, _non_overlapping(name_matches), "PERSON", replacements)

    for sentinel, original in sentinels.items():
        result = result.replace(sentinel, original)

    return Redaction(text=result, replacements=replacements)
