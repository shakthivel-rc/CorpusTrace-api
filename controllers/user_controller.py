import os
from fastapi import Depends, HTTPException, Response, status,Request 
from fastapi.responses import JSONResponse
from models.user_session import UserSession
from schemas.user import UserSignupReqeust, UserUpdateRequest
from schemas.email_schema import EmailTemplate
from sqlalchemy.orm import Session
from db.session import get_db
from models.user import User
from services.user import assign_role_for_user, create_new_user, delete_user_session, fetch_user_profile, fetch_user, get_user_by_id,get_user_by_username, user_role_deletion
from integrations.aws_ses import send_email
from utils.token import generate_random_code, encrypt_string, encode_to_base64
from services.user_role import add_roles_to_user
from datetime import datetime,timezone
from sqlalchemy import asc, desc
from typing import Dict, Any
from schemas.response import SuccessResponse,ErrorResponse
from fastapi.responses import JSONResponse
from utils.constants import login_email_template
from services.activity_log import log_activity
from schemas.user import ChangePasswordSchema
from utils.password import hash_password, validate_password_policy
import bcrypt

def get_user_profile(request: Request, db: Session):
    user = getattr(request.state, "user", None)
    
    if not user:
        error_response = ErrorResponse(status_code=401, message="User not authenticated")
        return JSONResponse(status_code=401, content=error_response.model_dump())  

    user_id = user.get("sub")
    
    return fetch_user_profile(user_id, db)


def add_user_controller(response: Response, user: UserSignupReqeust, db: Session = Depends(get_db), request: Request = None):
    if not user.email or not user.first_name or not user.last_name or not user.organization or not user.role_id or not user.department or not user.username:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(
            status="error",
            status_code=400,
            message="Some required fields are missing",
            data={}
        )
    username = user.username
    username_valid = get_user_by_username(db=db,username=username)
        
    if username_valid:
        error_response = ErrorResponse(status_code=400, message="Username already exists")
        return JSONResponse(status_code=400, content=error_response.model_dump())   

    web_base_url = os.getenv("APP_BASE_URL")
    token = generate_random_code(16)
    encoded_token = encode_to_base64(encrypt_string(token))
    verify_url = f'{web_base_url}/verify-account/{encoded_token}'
    new_user = create_new_user(user,db, token)
    add_roles_to_user(user, new_user, db)
    
    # setting up the email
    email_template = EmailTemplate(subject="Welcome to NexaRAG - Your Account is Ready",body=login_email_template(user.first_name, verify_url))
    # sending the email
    email_response = send_email(recipient_email=user.email,email_template=email_template)
    # throw error when email can be sent
    if not email_response["status"]:
        db.rollback()
        error_response = ErrorResponse(status_code=500, message="Error occurred while create a user. check the email you entered is correct")
        return JSONResponse(status_code=500, content=error_response.model_dump())
    # Log activity
    if request:
        current_user = getattr(request.state, "user", None)
        if current_user:
            ip_address = request.client.host if request.client else None
            actor_id = current_user.get("sub")
            actor = db.query(User).filter(User.id == actor_id).first()
            actor_name = f"{actor.first_name} {actor.last_name}" if actor else ""
            log_activity(db, actor_id, actor_name, "CREATE", entity_type="user", entity_id=new_user.id, details=f"Created user {user.first_name} {user.last_name}", ip_address=ip_address)

    db.commit()
    response.status_code = status.HTTP_201_CREATED
    return SuccessResponse(status="success", status_code=201, message="User added successfully", data=new_user)

def update_user_controller(user_id: str, update_data: UserUpdateRequest, db: Session, request: Request = None):
    try:
        user = get_user_by_id(db, user_id)
        if user.deleted == 1 or user.deleted_at is not None:
            error_response = ErrorResponse(status_code=404, message="User is deleted and cannot be updated")
            return JSONResponse(status_code=404, content=error_response.model_dump())  
        
        update_values = update_data.dict(exclude_unset=True)
         # Handle role assignment separately
        role_id = update_values.pop("role_id", None)
        if role_id:
            assign_role_for_user(user_id, role_id, db)
        # Update remaining user fields
        update_values["updated_at"] = datetime.now(timezone.utc)
        
        db.query(User).filter(User.id == user_id).update(update_values)

        # Log activity
        if request:
            current_user = getattr(request.state, "user", None)
            if current_user:
                ip_address = request.client.host if request.client else None
                actor_id = current_user.get("sub")
                actor = db.query(User).filter(User.id == actor_id).first()
                actor_name = f"{actor.first_name} {actor.last_name}" if actor else ""
                log_activity(db, actor_id, actor_name, "UPDATE", entity_type="user", entity_id=user_id, details=f"Updated user {user.first_name} {user.last_name}", ip_address=ip_address)

        db.commit()

        return SuccessResponse(
            status_code=200,
            message="User updated successfully",
            data={"user_id": user.id}
        )
    except Exception as e:
        db.rollback()
        error_response = ErrorResponse(status_code=500, message=f"Failed to update user :{e}")
        return JSONResponse(status_code=500, content=error_response.model_dump())   
    

def delete_user_controller(user_id: str, db: Session, request: Request = None):
    try:
        user = get_user_by_id(db, user_id)
   
        if user.deleted == 1:
            error_response = ErrorResponse(status_code=400, message="User is already deleted")
            return JSONResponse(status_code=400, content=error_response.model_dump())   
        delete_user_session(user_id,db)
        
        user_role_deletion(db, user_id)
        # Soft delete: update deleted_at and deleted
        user.deleted = 1
        user.deleted_at = datetime.now(timezone.utc)

        # Log activity
        if request:
            current_user = getattr(request.state, "user", None)
            if current_user:
                ip_address = request.client.host if request.client else None
                actor_id = current_user.get("sub")
                actor = db.query(User).filter(User.id == actor_id).first()
                actor_name = f"{actor.first_name} {actor.last_name}" if actor else ""
                log_activity(db, actor_id, actor_name, "DELETE", entity_type="user", entity_id=user_id, details=f"Deleted user {user.first_name} {user.last_name}", ip_address=ip_address)

        db.commit()
        return SuccessResponse(
            status_code=200,
            message="User deleted successfully",
            data={"user_id": user.id}
        )
    
    except Exception as e:
        db.rollback()
        error_response = ErrorResponse(status_code=500, message="Failed to delete user")
        return JSONResponse(status_code=500, content=error_response.model_dump())

def get_user(user_id: str, request: Request, db: Session):
    try:
        user: User | None = get_user_by_id(db, user_id)
        if not user:
            err_response =  ErrorResponse(status_code=500, message="User not found")
            return JSONResponse(status_code="User not found", content=err_response)
        user_role = { "id": user.roles[0].id, "name": user.roles[0].name } if len(user.roles) > 0 else []
        response_data = {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "department": user.department,
            "organization": user.organization,
            "role": user_role,
        }
        return SuccessResponse(status_code=200, message="User Data fetched successfully", data=response_data)
    except Exception as e:
        err_response = ErrorResponse(status_code=500, message="Error occurred while fetching users")
        return JSONResponse(content=err_response.model_dump(), status_code=500)

def change_password_controller(request: Request, data: ChangePasswordSchema, db: Session):
    user_payload = getattr(request.state, "user", None)
    if not user_payload:
        return JSONResponse(status_code=401, content=ErrorResponse(status_code=401, message="Not authenticated").model_dump())

    user_id = user_payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return JSONResponse(status_code=404, content=ErrorResponse(status_code=404, message="User not found").model_dump())

    # Verify current password
    if not bcrypt.checkpw(data.current_password.encode('utf-8'), user.password.encode('utf-8')):
        return JSONResponse(status_code=400, content=ErrorResponse(status_code=400, message="Current password is incorrect").model_dump())

    if data.new_password != data.confirm_password:
        return JSONResponse(status_code=400, content=ErrorResponse(status_code=400, message="New passwords do not match").model_dump())

    password_errors = validate_password_policy(data.new_password)
    if password_errors:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                status_code=400,
                message="Password does not meet policy requirements",
                data={"errors": password_errors},
            ).model_dump(),
        )

    user.password = hash_password(data.new_password)
    user.updated_at = datetime.now(timezone.utc)

    ip_address = request.client.host if request.client else None
    user_name = f"{user.first_name} {user.last_name}"
    log_activity(db, user_id, user_name, "SECURITY", details="Password changed", ip_address=ip_address)

    db.commit()
    return SuccessResponse(status_code=200, message="Password changed successfully", data={})


def get_users_info(db: Session, page: int, limit: int, sort_by: str, order: str,search:str) -> Dict[str, Any]:
    skip = (page - 1) * limit
    allowed_sort_fields = ["first_name","last_name","email"]
    
    if sort_by not in allowed_sort_fields:
        error_response = ErrorResponse(status_code=400, message="Invalid sort_by field")
        return JSONResponse(status_code=400, content=error_response.model_dump())

    sort_column = getattr(User, sort_by)  # Get column dynamically
    order_by = asc(sort_column) if order.lower() == "asc" else desc(sort_column)
        
    users,total_users= fetch_user(db, skip, limit, order_by, search)
    user_data = [
        {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "username":user.username,
            "department": user.department,
            "status": user.status,
            "organization": user.organization,
            "roles": [
                {
                    "role_id": role.id,
                    "role_name": role.name
                }
                for role in user.roles
            ],
            "permissions": [
                {
                    "permission_name": permission.name,
                    "machine_name": permission.machine_name
                }
                for role in user.roles
                for permission in role.permissions 
            ]
        }
        for user in users
    ]
    
    return SuccessResponse(
        status_code=200,
        message="Users fetched successfully",
        data={
            "records":user_data,
            "total":total_users}
    )
