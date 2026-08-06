from pydantic import BaseModel
from typing import List
 
class UserRoleAssignRequest(BaseModel):
    role_id: List[str]
    
 
class UserRoleResponse(BaseModel):
    status: str
    status_code: int
    message: str
    data: dict
