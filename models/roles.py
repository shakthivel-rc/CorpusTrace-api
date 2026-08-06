from sqlalchemy import Column, Integer, String, DateTime, Boolean
from db.session import Base
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import relationship, validates
from datetime import datetime, timezone
from models.role_permissions import RolePermission
from models.permissions import Permission
from models.user_roles import UserRole
import uuid
class Role(Base):
    __tablename__ = "roles"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    deleted_at = Column(DateTime, nullable=True)  # Allow NULL values for soft deletion
    deleted = Column(Integer, nullable=False, default=0)  # 0 for active, 1 for deleted


    permissions = relationship("Permission", secondary="role_permissions", back_populates="roles")
    users = relationship("User", secondary="user_roles", back_populates="roles")

    @validates('deleted')
    def validate_deleted(self, key, value):
        if value == 1:
            self.deleted_at = datetime.now(timezone.utc)
        elif value == 0:
            self.deleted_at = None  # Reset deleted_at if role is marked active again
        return value

# Event listener to automatically update the updated_at column
from sqlalchemy import event

@event.listens_for(Role, 'before_update')
def receive_before_update(mapper, connection, target):
    target.updated_at = datetime.now(timezone.utc)