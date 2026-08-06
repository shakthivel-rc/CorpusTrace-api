"""Unit tests for the Fernet token helpers (used for LLM credential encryption
and verify/reset links). Importing this module at all proves SECRET_KEY is a valid
Fernet key in the test environment."""
import pytest

from utils.token import decrypt_string, encrypt_string, generate_random_code

pytestmark = pytest.mark.unit


def test_encrypt_decrypt_round_trip():
    ciphertext = encrypt_string("super-secret-value")
    assert ciphertext != b"super-secret-value"
    assert decrypt_string(ciphertext) == "super-secret-value"


def test_encryption_is_non_deterministic():
    # Fernet embeds a random IV + timestamp, so two encryptions differ.
    assert encrypt_string("same") != encrypt_string("same")


def test_generate_random_code_returns_hex_of_expected_length():
    code = generate_random_code(16)
    assert len(code) == 32  # token_hex(n) → 2 hex chars per byte
    assert all(c in "0123456789abcdef" for c in code)
