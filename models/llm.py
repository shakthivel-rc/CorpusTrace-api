import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.mysql import CHAR

from db.session import Base


class LlmProviderCredential(Base):
    __tablename__ = "llm_provider_credentials"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(CHAR(36), ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String(50), nullable=False, index=True)
    encrypted_api_key = Column(Text, nullable=True)
    api_key_hint = Column(String(64), nullable=True)
    base_url = Column(String(512), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    validation_status = Column(String(50), nullable=False, default="stored")
    validation_error = Column(String(512), nullable=True)
    last_validated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    deleted_at = Column(DateTime, nullable=True)


class LlmModelCatalog(Base):
    __tablename__ = "llm_model_catalogs"
    __table_args__ = (UniqueConstraint("provider", "model_id", name="uq_llm_model_provider_model"),)

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    provider = Column(String(50), nullable=False, index=True)
    model_id = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    capabilities_json = Column(Text, nullable=True)
    raw_json = Column(Text, nullable=True)
    source = Column(String(50), nullable=False, default="seed")
    is_active = Column(Boolean, nullable=False, default=True)
    fetched_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class LlmUserPreference(Base):
    __tablename__ = "llm_user_preferences"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(CHAR(36), ForeignKey("users.id"), nullable=False, unique=True, index=True)
    provider = Column(String(50), nullable=False)
    model_id = Column(String(255), nullable=False)
    # Generation tuning; NULL means "use the service defaults" (temperature 0.2, provider default limits).
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    top_p = Column(Float, nullable=True)
    # The off switch for LLM synthesis. NOT NULL with a server default so every existing
    # row keeps working: absent this, users who set a preference before the column existed
    # would read back NULL and silently lose their model.
    llm_enabled = Column(Boolean, nullable=False, server_default=text("1"), default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
