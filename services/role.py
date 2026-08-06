from models.role_permissions import RolePermission
from fastapi import HTTPException
from models.roles import Role
from models.permissions import Permission
from sqlalchemy.orm import Session
from schemas.roles import RoleCreateRequest
from schemas.response import SuccessResponse,ErrorResponse
from fastapi.responses import JSONResponse
from models.user import User
from models.user_roles import UserRole


def get_role_by_id(db, role_id):
    return db.query(Role).filter(Role.id == role_id).first()

def get_user_roles_by_id(db, role_id):
    return db.query(UserRole).filter(UserRole.role_id == role_id).first()

def remove_permission_from_role(db, role_id):
    return db.query(RolePermission).filter(
            RolePermission.role_id == role_id,
        ).delete(synchronize_session=False)
    
def delete_role(db, role_id):
    return db.query(Role).filter(Role.id == role_id).delete()

def get_role_by_rolename(db, role):
    return db.query(Role).filter(Role.name == role.name).first()

def fetch_permissions_on_role(db, role):
    return db.query(Permission).filter(Permission.id.in_(role.permission_ids)).all()

def get_roles(db:Session, skip:int, limit:int, order_by, search:str):
    if limit == 0:
        roles = db.query(Role).filter(Role.name.ilike(f"%{search}%")).order_by(order_by).offset(skip).all()
        total_roles = len(roles)
        return roles, total_roles
    else:
        roles = db.query(Role).filter(Role.name.ilike(f"%{search}%")).order_by(order_by).offset(skip).limit(limit).all()
        total_roles = len(roles)
        return roles,total_roles

def role_info(role_id,db):
    return db.query(Role).filter(Role.id == role_id).first()


def fetch_multiple_permission(db, permission_ids):
    return db.query(Permission).filter(Permission.id.in_(permission_ids)).all()

def remove_role_permission_by_role_id(db, role_id):
        return db.query(RolePermission).filter(RolePermission.role_id == role_id).delete()
    