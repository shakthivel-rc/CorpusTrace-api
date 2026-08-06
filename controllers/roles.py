from sqlalchemy.orm import Session
from models.role_permissions import RolePermission
from services.role import fetch_multiple_permission, remove_permission_from_role, delete_role, fetch_permissions_on_role, get_role_by_id, get_role_by_rolename, get_user_roles_by_id, get_roles, remove_role_permission_by_role_id, role_info
from models.roles import Role
from sqlalchemy import asc,desc
from schemas.response import SuccessResponse,ErrorResponse
from fastapi.responses import JSONResponse
from schemas.roles import RoleCreateRequest, RolePermissionUpdateRequest
from services.activity_log import log_activity

def remove_role(role_id: str, db: Session):
    try:
        # Check if the role exists
        role = get_role_by_id(db, role_id)
        if not role:
            error_response = ErrorResponse(status_code=404, message="Role not found")
            return JSONResponse(status_code=404, content=error_response.model_dump())  

        user_role = get_user_roles_by_id(db, role_id)
        if user_role:
            error_response = ErrorResponse(status_code=404, message="Role Already assigned to User, So unable to delete the role.")
            return JSONResponse(status_code=404, content=error_response.model_dump())
        
        # Delete the specified permissions from the role
        remove_permission_from_role(db, role_id)

        # Delete the specified role
        delete_role(db, role_id)

        db.commit()
        return SuccessResponse(
            status_code=200,
            message = f"Role deleted successfully",
            data= {
                "role_id": role_id,
            }
        )

    except Exception as e:
        db.rollback()
        error_response = ErrorResponse(status_code=404, message="The user does not have these permissions and cannot delete them.",
            data = {
                "role_id": role_id,
            })
        return JSONResponse(status_code=404, content=error_response.model_dump())  
 
def add_role(role: RoleCreateRequest, db: Session):
    try:
        if len(role.permission_ids) == 0:
            error_response = ErrorResponse(status_code=404, message="Role must have at least one permission.")
            return JSONResponse(status_code=404, content=error_response.model_dump())

        # Check if role already exists
        existing_role = get_role_by_rolename(db, role)
        if existing_role:
            error_response = ErrorResponse(status_code=400, message="Role already exists")
            return JSONResponse(status_code=400, content=error_response.model_dump())  
       
        # Fetch all permissions from the database based on provided IDs
        permissions = fetch_permissions_on_role(db, role)

        # Validate that all requested permissions exist
        existing_permission_ids = {permission.id for permission in permissions}
        missing_permissions = set(role.permission_ids) - existing_permission_ids

        if missing_permissions:
            error_response = ErrorResponse(status_code = 400, message= f"Permissions not found: {list(missing_permissions)}")
            return JSONResponse(status_code = 400, content= error_response.model_dump())  
            
        # Create the new role
        new_role = Role(name=role.name)
        db.add(new_role)
        db.commit()
        db.refresh(new_role)

        # Assign permissions to the role
        role_permissions = [RolePermission(role_id=new_role.id, permission_id=perm.id) for perm in permissions]
        db.add_all(role_permissions)
        db.commit()

    except Exception as e:
        db.rollback()
        error_response = ErrorResponse(status_code=500, message=f"An error occurred while creating the new role: {str(e)}")
        return JSONResponse(status_code=500, content=error_response.model_dump())  
            
    return SuccessResponse(
        status_code=201,
        message="Role created successfully",
        data= {
            "role_id": new_role.id,
            "name": new_role.name,
            "assigned_permissions": [perm.id for perm in permissions]
        })

def fetch_all_roles(db, page: int, limit: int, sort_by: str, order: str, search: str):
    skip = (page - 1) * limit
    allowed_sort_fields = ["name"]
    if sort_by not in allowed_sort_fields:
        error_response = ErrorResponse(status_code=400, message="Invalid sort field")
        return JSONResponse(status_code=400, content=error_response.model_dump())  
        
    sort_column = getattr(Role, sort_by)
    order_by = asc(sort_column) if order.lower() == "asc" else desc(sort_column)
    
    roles,total_roles = get_roles(db, skip, limit, order_by, search)

    roles_data =[{
        "id": role.id,
        "name": role.name,
    } for role in roles] 
    
    return SuccessResponse(
        status_code=200,
        message="Roles fetched successfully",
        data={
            "records": roles_data,
            "total": total_roles
        }
    )


def fetch_role(role_id, db):
    try:
        role = role_info(role_id, db)
        
        if not role:
            error_response = ErrorResponse(status_code= 404, message= "Role not found")
            return JSONResponse(status_code= 404, content= error_response.model_dump()) 
 
        role_data = {
            "id":role.id,
            "name": role.name,
            "permissions": [
                {
                    "id": permission.id,
                    "permission_name": permission.name
                }
                for permission in role.permissions  # Ensure relationship exists in models
            ]
        }
    except Exception as e:
        error_response = ErrorResponse(status_code= 400, message= f"Failed to fetch role details: {str(e)}")
        return JSONResponse(status_code= 400, content= error_response.model_dump()) 
        
    return SuccessResponse(
        status_code= 200,
        message= "Role fetched successfully",
        data= role_data
    )
    
def update_role_controller(role_id: str, request: RolePermissionUpdateRequest, db: Session, http_request=None):
    try:
        permission_ids = request.permission_ids
        role_name = request.name
        # return update_exisiting_role(role_id, role_name, permission_ids, db)
        role = get_role_by_id(db, role_id)
        role.name = role_name
        if not role:
            error_response = ErrorResponse(status_code=404, message="Role not found")
            return JSONResponse(status_code=404, content=error_response.model_dump())  
        
        permissions = fetch_multiple_permission(db, permission_ids)
        if not permissions:
            error_response = ErrorResponse(status_code=404, message="No valid permissions found")
            return JSONResponse(status_code=404, content=error_response.model_dump())  
        # Remove existing permissions before inserting the received permissions
        remove_role_permission_by_role_id(db, role_id)

        # Insert the received permissions
        for permission_id in permission_ids:
            role_permission = RolePermission(role_id=role_id, permission_id=permission_id)
            db.add(role_permission)

        # Log activity
        if http_request:
            current_user = getattr(http_request.state, "user", None)
            if current_user:
                from models.user import User
                ip_address = http_request.client.host if http_request.client else None
                actor_id = current_user.get("sub")
                actor = db.query(User).filter(User.id == actor_id).first()
                actor_name = f"{actor.first_name} {actor.last_name}" if actor else ""
                log_activity(db, actor_id, actor_name, "UPDATE", entity_type="role", entity_id=role_id, details=f"Updated role {role_name}", ip_address=ip_address)

        db.commit()
        return SuccessResponse(
            status_code=200,
            message="Role updated successfully",
            data={"role_id": role_id},
        )
 
 
    except Exception as e:
        db.rollback()
        error_response = ErrorResponse(status_code=500, message="User already have this Permissions")
        return JSONResponse(status_code=500, content=error_response.model_dump())  
 