import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import relationship

from db.session import Base


class Resource(Base):
    __tablename__ = "resources"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    resource_name = Column(String(255), nullable=False)
    resource_type = Column(String(50), nullable=False, default="knowledge_base")
    upload_status = Column(Boolean, nullable=False, default=False)
    user_id = Column(CHAR(36), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    files = relationship("File", back_populates="resource")
    chunks = relationship("DocumentChunk", back_populates="resource")
