from pydantic import BaseModel
from typing import List

class RolePermissionUpdateRequest(BaseModel):
    name: str
    permission_ids: List[str]


class RoleUpdateRequest(BaseModel):
    new_name: str


class RolePermissionDeleteRequest(BaseModel):
    permission_ids: List[str]


class RoleCreateRequest(BaseModel):
    name: str
    permission_ids: List[str]  # List of permission IDs to assign to the role


class GetAllRoles(BaseModel):
    roles : str
    