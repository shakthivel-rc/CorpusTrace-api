"""Unit tests for the password policy — the single validator behind all four
password entry points. Mirrors nexarag-app/src/lib/passwordRules.ts rule-for-rule."""
import pytest

from utils.password import hash_password, pwd_context, validate_password_policy

pytestmark = pytest.mark.unit


class TestValidatePasswordPolicy:
    def test_accepts_a_fully_compliant_password(self):
        assert validate_password_policy("Abcdef123!@#") == []

    def test_flags_a_password_that_is_too_short(self):
        errors = validate_password_policy("Abc1!")
        assert any("at least" in e for e in errors)

    def test_requires_an_uppercase_letter(self):
        assert any("uppercase" in e for e in validate_password_policy("abcdef123!@#"))

    def test_requires_a_lowercase_letter(self):
        assert any("lowercase" in e for e in validate_password_policy("ABCDEF123!@#"))

    def test_requires_a_digit(self):
        assert any("number" in e for e in validate_password_policy("Abcdefghi!@#"))

    def test_requires_a_special_character(self):
        assert any("special" in e for e in validate_password_policy("Abcdef1234567"))

    def test_rejects_whitespace(self):
        assert any("whitespace" in e for e in validate_password_policy("Abcdef 123!@#"))

    def test_reports_every_failing_rule_at_once(self):
        # "short" fails length, uppercase, digit and special simultaneously.
        assert len(validate_password_policy("short")) >= 4


def test_hash_password_is_verifiable_and_not_plaintext():
    hashed = hash_password("Abcdef123!@#")
    assert hashed != "Abcdef123!@#"
    assert pwd_context.verify("Abcdef123!@#", hashed)
    assert not pwd_context.verify("wrong-password", hashed)
