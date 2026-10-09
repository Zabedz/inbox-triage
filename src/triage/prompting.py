"""Wrapping untrusted text in a <DATA> block that the text cannot close."""

from __future__ import annotations

import re

_DELIMITER = re.compile(r"<\s*/?\s*data\b\s*>?", re.IGNORECASE)


def sanitize(text: str) -> str:
    """Strip DATA delimiters until none remain, so nested fragments cannot rebuild one."""
    while True:
        cleaned = _DELIMITER.sub("", text)
        if cleaned == text:
            return cleaned
        text = cleaned


def data_block(text: str) -> str:
    return f"<DATA>\n{sanitize(text)}\n</DATA>"
