from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv


load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def _optional_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def _split_csv(name: str, default: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, default)
    values = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not values:
        raise RuntimeError(f"Environment variable {name} must contain at least one value")
    return values


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    database_url: str
    secret_key: str
    algorithm: str
    jwt_expiry_hours: int
    refresh_token_expiry_days: int
    cors_origins: tuple[str, ...]
    log_level: str
    auth_max_failed_login_attempts: int
    auth_lockout_minutes: int
    password_min_length: int
    rag_upload_dir: str
    llm_credential_encryption_key: str | None
    llm_model_cache_ttl_hours: int
    llm_request_timeout_seconds: int
    llm_first_token_timeout_seconds: int
    llm_max_retries: int
    llm_retry_backoff_seconds: int
    ollama_base_url: str

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"prod", "production"}


@lru_cache
def get_settings() -> Settings:
    settings = Settings(
        app_name=os.getenv("APP_NAME", "CorpusTrace API"),
        environment=os.getenv("ENVIRONMENT", "development"),
        database_url=_required("DATABASE_URL"),
        secret_key=_required("SECRET_KEY"),
        algorithm=os.getenv("ALGORITHM", "HS256"),
        jwt_expiry_hours=_optional_int("JWT_EXPIRY_VALUE", 1),
        refresh_token_expiry_days=_optional_int("REFRESH_TOKEN_EXPIRY_DAYS", 30),
        cors_origins=_split_csv("CORS_ORIGINS", "http://localhost:5173"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        auth_max_failed_login_attempts=_optional_int("AUTH_MAX_FAILED_LOGIN_ATTEMPTS", 5),
        auth_lockout_minutes=_optional_int("AUTH_LOCKOUT_MINUTES", 15),
        password_min_length=_optional_int("PASSWORD_MIN_LENGTH", 12),
        rag_upload_dir=os.getenv("RAG_UPLOAD_DIR", "uploads/rag"),
        llm_credential_encryption_key=os.getenv("LLM_CREDENTIAL_ENCRYPTION_KEY"),
        llm_model_cache_ttl_hours=_optional_int("LLM_MODEL_CACHE_TTL_HOURS", 24),
        llm_request_timeout_seconds=_optional_int("LLM_REQUEST_TIMEOUT_SECONDS", 45),
        # Wall-clock budget for the *first* token of a stream, which
        # LLM_REQUEST_TIMEOUT_SECONDS cannot bound: that is a per-socket-read timeout, and a
        # provider's keep-alive comment lines are successful reads that reset it. OpenRouter
        # emits `: OPENROUTER PROCESSING` precisely to defeat proxy timeouts, so a queued
        # free-tier model held a request open for 124 s under a 45 s timeout and showed the
        # user nothing the whole time. Generation itself stays unbounded — a slow answer is
        # fine; silence is not.
        llm_first_token_timeout_seconds=_optional_int("LLM_FIRST_TOKEN_TIMEOUT_SECONDS", 60),
        llm_max_retries=_optional_int("LLM_MAX_RETRIES", 2),
        llm_retry_backoff_seconds=_optional_int("LLM_RETRY_BACKOFF_SECONDS", 1),
        # Where Ollama is, when the user has saved no base URL of their own.
        #
        # It needs to be configurable because the right answer differs by how the API is
        # running, and neither answer works for the other: a native process reaches Ollama
        # on the host at localhost, while inside a container localhost IS the container, so
        # the default silently points the embedding call at the API itself. docker-compose
        # sets this to the service name. Nothing about Ollama is required — an unset value
        # and no Ollama installed simply means embeddings are not offered through it.
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        or "http://localhost:11434",
    )

    if settings.is_production and "*" in settings.cors_origins:
        raise RuntimeError("CORS_ORIGINS cannot contain '*' in production")

    return settings
