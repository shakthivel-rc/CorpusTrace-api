from pydantic import BaseModel

class PermissionCreate(BaseModel):
    permission_name: str

class GetAllPermissionSchema(BaseModel):
    permission_name: str
    machine_name: str

    class Config:
        from_attributes = True  # Ensures SQLAlchemy compatibility