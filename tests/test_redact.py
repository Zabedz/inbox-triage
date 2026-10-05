from __future__ import annotations

from triage.redact import redact


def test_masks_email_phone_and_name() -> None:
    text = "Email me at john.smith@example.com or call +44 20 7946 0958. Thanks, Zabed."
    r = redact(text, names=["Zabed"])

    assert "john.smith@example.com" not in r.text
    assert "[EMAIL_1]" in r.text
    assert "[PHONE_1]" in r.text
    assert "[PERSON_1]" in r.text
    assert r.replacements["[EMAIL_1]"] == "john.smith@example.com"
    assert r.replacements["[PHONE_1]"] == "+44 20 7946 0958"
    assert r.replacements["[PERSON_1]"] == "Zabed"


def test_keeps_company_names() -> None:
    text = "Thank you for applying to Acme Corp. Reach us at no-reply@acmecorp.com."
    r = redact(text, names=["Zabed"], keep=["Acme Corp"])

    assert "Acme Corp" in r.text
    assert "no-reply@acmecorp.com" not in r.text


def test_keep_domain_does_not_leak_email() -> None:
    r = redact("Reach us at no-reply@acmecorp.com.", keep=["acmecorp.com"])
    assert "no-reply@acmecorp.com" not in r.text
    assert "[EMAIL_1]" in r.text


def test_no_tld_email_is_masked() -> None:
    r = redact("Internal address: zabed@localhost")
    assert "zabed@localhost" not in r.text
    assert "[EMAIL_1]" in r.text


def test_leaves_bare_date_unmasked() -> None:
    text = "The interview is on 2026-10-15 at 10:30."
    r = redact(text, names=["Zabed"])
    assert "2026-10-15" in r.text
    assert "[PHONE" not in r.text


def test_leaves_datetime_unmasked() -> None:
    text = "The interview is on 2026-10-15 14:00."
    r = redact(text, names=["Zabed"])
    assert "2026-10-15 14:00" in r.text
    assert "[PHONE" not in r.text


def test_no_names_means_nothing_name_masked() -> None:
    text = "Alice met Bob."
    r = redact(text)
    assert "Alice" in r.text
    assert "Bob" in r.text


def test_empty_input() -> None:
    assert redact("").text == ""
    assert redact("").replacements == {}


def test_names_are_case_insensitive() -> None:
    r = redact("Hi ZABED", names=["Zabed"])
    assert "[PERSON_1]" in r.text
