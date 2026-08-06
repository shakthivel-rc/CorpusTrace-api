import uuid
from sqlalchemy import Column, Integer, ForeignKey
from db.session import Base
from sqlalchemy.dialects.mysql import CHAR

class RolePermission(Base):
    __tablename__ = "role_permissions"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    role_id = Column(CHAR(36), ForeignKey("roles.id"), nullable=False)
    permission_id = Column(CHAR(36), ForeignKey("permissions.id"), nullable=False)