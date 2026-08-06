from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
import uuid

from db.session import Base

class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False) 
    user_id = Column(String,nullable=False)
    access_token = Column(String(1024), nullable=False)
    refresh_token = Column(String(1024), nullable=True)
    refresh_token_hash = Column(String(128), nullable=True, index=True)
    refresh_token_expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    user_agent = Column(String(512), nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
