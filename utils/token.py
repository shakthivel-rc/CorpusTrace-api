import os
import secrets
import base64
from cryptography.fernet import Fernet

def generate_random_code(length: int) -> str:
    """Generate a random hexadecimal string of the specified byte length."""
    return secrets.token_hex(length)

# Generate a key (do this once and store it securely)
key = os.getenv("SECRET_KEY")
cipher_suite = Fernet(key)

def encrypt_string(plain_text: str) -> bytes:
    """Encrypts a string using Fernet encryption."""
    return cipher_suite.encrypt(plain_text.encode())

def decrypt_string(encrypted_text: bytes) -> str:
    """Decrypts a string using Fernet encryption."""
    return cipher_suite.decrypt(encrypted_text).decode()

def encode_to_base64(text: str) -> str:
    """Encodes a given string into Base64 format."""
    encoded_bytes = base64.b64encode(text)
    return encoded_bytes.decode("utf-8")

def decode_base64(encoded_text: str) -> str:
    """Decodes a Base64-encoded string."""
    decoded_bytes = base64.b64decode(encoded_text)  # Decode Base64 to bytes
    return decoded_bytes.decode("utf-8")  # Convert bytes back to string