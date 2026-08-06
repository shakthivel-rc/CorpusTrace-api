from models.roles import Role
from fastapi import HTTPException
from models.permissions import Permission
from models.role_permissions import RolePermission
from schemas.response import SuccessResponse,ErrorResponse



def permission_to_role(role_id,permission_ids,db):
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        return ErrorResponse(
            status_code=404,
            message="Role not found"
        )
    permissions = db.query(Permission).filter(Permission.id.in_(permission_ids)).all()
    
    if len(permissions) != len(permission_ids):
        return ErrorResponse(
            status_code=404,
            message="One or more permissions not found")
            
    role_permissions = [RolePermission(role_id=role_id, permission_id=perm.id) for perm in permissions]

    db.add_all(role_permissions)
    db.commit()
