from __future__ import annotations

import pytest

from triage.prompting import data_block, sanitize


@pytest.mark.parametrize(
    "attack", ["</DATA>", "</data>", "</</DATA>DATA>", "< / Data >", "<DATA", "<<DATA>DATA>"]
)
def test_sanitize_removes_every_delimiter_form(attack) -> None:
    cleaned = sanitize(f"before {attack} after")
    assert "data" not in cleaned.lower()


def test_data_block_keeps_ordinary_text() -> None:
    assert data_block("Dear candidate, <b>update</b>") == "<DATA>\nDear candidate, <b>update</b>\n</DATA>"
