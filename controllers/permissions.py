from sqlalchemy.orm import Session
from services.permissions import fetch_permission, get_permission, get_role_permissions
from schemas.permissions import PermissionCreate
from models.permissions import Permission
from sqlalchemy import asc,desc
from schemas.response import SuccessResponse, ErrorResponse
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from utils.common import to_snake_case
from services.activity_log import log_activity

def add_permission(permission: PermissionCreate, db: Session, request=None):
    try:
        # Check if permission already exists
        existing_permission = db.query(Permission).filter(Permission.name == permission.permission_name).first()
        if existing_permission:
            error_response = ErrorResponse(status_code=409, message="Permission already exists")
            return JSONResponse(status_code=409, content=error_response.model_dump())
            
        # Create a new permission
        new_permission = Permission(
            name = permission.permission_name,
            machine_name = to_snake_case(permission.permission_name), 
        )
        db.add(new_permission)

        # Log activity
        if request:
            current_user = getattr(request.state, "user", None)
            if current_user:
                from models.user import User
                ip_address = request.client.host if request.client else None
                actor_id = current_user.get("sub")
                actor = db.query(User).filter(User.id == actor_id).first()
                actor_name = f"{actor.first_name} {actor.last_name}" if actor else ""
                log_activity(db, actor_id, actor_name, "CREATE", entity_type="permission", entity_id=new_permission.id, details=f"Created permission {permission.permission_name}", ip_address=ip_address)

        db.commit()
        db.refresh(new_permission)
        return SuccessResponse(
            status_code=201,
            message="Permission created successfully",
            data={
                "permission_id": new_permission.id,
                "permission_name": new_permission.name
            }
        )

    except Exception as e:
        db.rollback()  # Rollback transaction on failure
        error_response = ErrorResponse(status_code=500, message="Failed to create permission")
        return JSONResponse(status_code=500, content=error_response.model_dump())


def retrieve_permissions(db,page:int,
                        limit:int,
                        sort_by:str,
                        order:str,
                        search: str):
    skip = (page - 1) * limit
    allowed_sort_fields = ["id", "name", "machine_name"]
    if sort_by not in allowed_sort_fields:
        error_response = ErrorResponse(status_code=400, message="Invalid sort field.")
        return JSONResponse(status_code=400, content=error_response.model_dump())

    sort_column = getattr(Permission, sort_by)
    order_by = asc(sort_column) if order.lower() == "asc" else desc(sort_column)

    permissions,total_permissions = fetch_permission(db, skip, limit, order_by, search)
    
    permission_data = [
        {
            "permission_id": permission.id,
            "permission_name": permission.name,
            "machine_name": permission.machine_name
        }
        for permission in permissions
    ]
    
    return SuccessResponse(
        status_code=200,
        message="Permissions fetched successfully.",
        data={
            "records": permission_data,
            "total": total_permissions
        })

def retrieve_permission(permission_id, db):
    try:
        permission = get_permission(permission_id, db)
        if not permission:
            error_response = ErrorResponse(status_code= 404, message= "Permission not found.")
            return JSONResponse(status_code= 404, content= error_response.model_dump())

        permission_data = {
            "id": permission.id,
            "permission_name": permission.name
        }
    except Exception as e:
        error_response = ErrorResponse(status_code= 404, message= "Permission not found.")
        return JSONResponse(status_code= 404, content= error_response.model_dump())

    return SuccessResponse(
        status_code = 200,
        message = "Permission info fetched successfully.",
        data = permission_data
        )

def modify_permission(permission_id: str, permission: PermissionCreate, db: Session):
    try:
        # Fetch the existing permission
        existing_permission = get_permission(db= db, permission_id= permission_id)
        if not existing_permission:
            error_response = ErrorResponse(status_code= 404, message= "Permission not found")
            return JSONResponse(status_code= 404, content= error_response.model_dump())    

        # Update only the name field (machine_name remains unchanged)
        existing_permission.name = permission.permission_name
        existing_permission.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(existing_permission)
        return SuccessResponse(
            status_code=200,
            message="Permission updated successfully",
            data={"permission_id": existing_permission.id})

    except Exception as e:
        db.rollback()  # Rollback transaction on failure
        error_response = ErrorResponse(status_code= 500, message= "Failed to update permission name")
        return JSONResponse(status_code= 500, content= error_response.model_dump())    

def delete_permission_by_id(permission_id, db):
    try:
        permission = get_permission(permission_id, db)
    
        if not permission:
            error_response = ErrorResponse(status_code=404, message="Permission not found")
            return JSONResponse(status_code=404, content=error_response.model_dump())  
        
        role_assign = get_role_permissions(permission_id, db)
        if role_assign:
            error_response = ErrorResponse(
                status_code=409,
                message="Cannot delete permission as it is assigned to a role."
            )
            return JSONResponse(status_code=409, content=error_response.model_dump())

        db.delete(permission)
        db.commit()
        
        return SuccessResponse(
            status_code=200,    
            message="Permission deleted successfully",
            data={}
        )

    except Exception as e:
        db.rollback()  # Rollback transaction on failure
        error_response = ErrorResponse(status_code=500, message="Failed to update permission name")
        return JSONResponse(status_code=500, content=error_response.model_dump())