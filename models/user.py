from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.mysql import CHAR
from db.session import Base
from datetime import datetime, timezone
import uuid
from models.user_roles import UserRole
from models.roles import Role


class User(Base):
    __tablename__ = "users"
    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String(100), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    organization = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    password = Column(String(512), nullable=True)
    status = Column(Boolean, nullable=False, default=False)
    fp_token = Column(String(255), nullable=True)
    fp_token_created_at = Column(DateTime, nullable=True)
    verify_token = Column(String(255), nullable=True)
    otp_code = Column(String(255), nullable=True)
    otp_created_at = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime, nullable=True)
    last_failed_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    deleted_at = Column(DateTime, nullable=True)  # Allow NULL values for soft deletion
    deleted = Column(Integer, nullable=False, default=0)  # 0 for active, 1 for deleted

    roles = relationship("Role", secondary="user_roles", back_populates="users")

    @validates('deleted')
    def validate_deleted(self, key, value):
        if value == 1:
            self.deleted_at = datetime.now(timezone.utc)
        elif value == 0:
            self.deleted_at = None  # Reset deleted_at if user is marked active again
        return value


# Event listener to automatically update the updated_at column
from sqlalchemy import event

@event.listens_for(User, 'before_update')
def receive_before_update(mapper, connection, target):
    target.updated_at = datetime.now(timezone.utc)