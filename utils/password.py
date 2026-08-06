from passlib.context import CryptContext

from core.config import get_settings


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def validate_password_policy(password: str) -> list[str]:
    settings = get_settings()
    errors: list[str] = []

    if len(password) < settings.password_min_length:
        errors.append(f"Password must be at least {settings.password_min_length} characters long.")
    if not any(char.isupper() for char in password):
        errors.append("Password must include at least one uppercase letter.")
    if not any(char.islower() for char in password):
        errors.append("Password must include at least one lowercase letter.")
    if not any(char.isdigit() for char in password):
        errors.append("Password must include at least one number.")
    if not any(not char.isalnum() for char in password):
        errors.append("Password must include at least one special character.")
    if any(char.isspace() for char in password):
        errors.append("Password cannot contain whitespace.")

    return errors