from sqlalchemy import Column, Integer, ForeignKey
from db.session import Base
from sqlalchemy.dialects.mysql import CHAR
import uuid

class UserRole(Base):
    __tablename__ = "user_roles"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(CHAR(36), ForeignKey("users.id"), nullable=False)
    role_id = Column(CHAR(36), ForeignKey("roles.id"), nullable=False)