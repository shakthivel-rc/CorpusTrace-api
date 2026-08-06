from sqlalchemy.orm import Session, selectinload
from models.user import User
from models.roles import Role
from models.user_roles import UserRole
from sqlalchemy.exc import SQLAlchemyError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from db.session import get_db
from schemas.user import UserUpdateRequest,UserProfileResponse,UserProfileResponseWrapper, UserUpdateResponse, UserRoleSchema, UserPermissionSchema
from datetime import datetime, timezone
from schemas.response import SuccessResponse, ErrorResponse
from sqlalchemy import or_, and_
from fastapi.responses import JSONResponse
from models.user_session import UserSession

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username, User.deleted != True).first()

def get_user_by_id(db: Session, user_id: str):
    return db.query(User).filter(User.id == user_id).first()

def fetch_user_profile(user_id: str, db: Session) -> UserProfileResponseWrapper:
    user_record = get_id_by_token(user_id, db)

    if not user_record:
        error_response = ErrorResponse(status_code=404, message="User not found")
        return JSONResponse(status_code=404, content=error_response.model_dump())  

    user_roles = [UserRoleSchema(role_id=role.id, role_name=role.name) for role in user_record.roles]
    user_permissions = [
        UserPermissionSchema(permission_name=permission.name, machine_name=permission.machine_name)
        for role in user_record.roles
        for permission in role.permissions
    ]
    user_profile = UserProfileResponse(
        id=user_record.id,
        username=user_record.username,
        first_name=user_record.first_name,
        last_name=user_record.last_name,
        email=user_record.email,
        organization=user_record.organization,
        department=user_record.department,
        created_at=user_record.created_at,
        updated_at=user_record.updated_at,
        roles=user_roles,
        permissions=user_permissions,
    )
    return UserProfileResponseWrapper(
        status="success",
        status_code=200,
        message="User profile fetched successfully",
        data=user_profile,
    )

def create_new_user(user: User, db: Session, token: str = ""):
    new_user = User(
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                organization = user.organization,
                department = user.department,
                verify_token = token,
                deleted=0,  # Ensure user is active by default
                deleted_at=None 
                )
    db.add(new_user)
    db.flush()
    db.refresh(new_user)
    
    # Convert SQLAlchemy object to dictionary (excluding private attributes)
    user_dict = {key: value for key, value in vars(new_user).items() if not key.startswith("_")}
    return user_dict  
    
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email, User.deleted != True).first()        

def get_id_by_token(user_id,db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()

def assign_role_for_user(user_id, role_id, db):
    try:
        db.query(UserRole).filter(UserRole.user_id == user_id).delete()
        db.flush()
        # Assign the new role
        new_role = UserRole(user_id=user_id, role_id=role_id)
        db.add(new_role)
        db.flush()
            
    except Exception as e:
        db.rollback()
        error_response = ErrorResponse(status_code=401, message="Error Occured : {e}")
        return JSONResponse(status_code=401, content=error_response.model_dump())

def delete_user_session(user_id: str, db: Session):
    return db.query(UserSession).filter(UserSession.user_id == user_id).delete()

def fetch_user(db: Session, skip, limit, order_by, search):
    # The caller renders every user's roles AND every role's permissions. Lazy-loading walks
    # that as one query per user plus one per role — 2N+1 for a page of N. selectinload
    # collapses it to three regardless of page size, which matters because this endpoint
    # defaults to no limit at all.
    query = (
        db.query(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .filter(User.deleted != 1)  # Ensure deleted users are excluded globally
    )

    if search:
        search_terms = search.strip().split()  
        num_words = len(search_terms)
        if num_words == 1:
            query = query.filter(
                or_(
                    User.first_name.ilike(f"%{search}%"),
                    User.last_name.ilike(f"%{search}%"),
                    User.email.ilike(f"%{search}%"),
                    User.username.ilike(f"%{search}%")

                )
            )

        elif num_words == 2:
            query = query.filter(
                and_(
                    User.first_name.ilike(f"%{search_terms[0]}%"),
                    User.last_name.ilike(f"%{search_terms[1]}%")
                )
            )

        elif num_words >= 3:
            first_name_part = f"{search_terms[0]} {search_terms[1]}"
            last_name_part = search_terms[-1]  # Last word maps to last_name

            query = query.filter(
                and_(
                    User.first_name.ilike(f"%{first_name_part}%"),
                    User.last_name.ilike(f"%{last_name_part}%")
                )
            )
    
    if limit != 0:
        users = query.order_by(order_by).offset(skip).limit(limit).all()   
        total_users = len(users)
    else:
        users = query.order_by(order_by).offset(skip).all()
        total_users = len(users)
    return users,total_users

def user_role_deletion(db, user_id):
    return db.query(UserRole).filter(UserRole.user_id == user_id).delete()
        