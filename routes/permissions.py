from fastapi import APIRouter, HTTPException, Depends, Response, Query, Request
from schemas.user import UserSignupReqeust
from sqlalchemy.orm import Session
from db.session import get_db
from dependencies.auth import check_permissions
from fastapi import Depends, HTTPException, Request
from schemas.permissions import PermissionCreate,GetAllPermissionSchema
from models.permissions import Permission
from controllers.permissions import delete_permission_by_id, retrieve_permissions,retrieve_permission, add_permission, modify_permission
from schemas.user_role import UserRoleAssignRequest, UserRoleResponse
from typing import Dict, Any, List,Union
from schemas.response import ErrorResponse,SuccessResponse
from fastapi.responses import JSONResponse
from models.role_permissions import RolePermission

router = APIRouter(
    prefix="/permissions",
    tags=["Permissions"]
)

@router.post("", response_model=Union[SuccessResponse, ErrorResponse], dependencies=[Depends(check_permissions(["manage_users"]))])
def create_permission(permission: PermissionCreate, request: Request, db: Session = Depends(get_db)):
    return add_permission(permission, db, request=request)

@router.put("/{permission_id}", response_model= Union[SuccessResponse, ErrorResponse], dependencies=[Depends(check_permissions(["manage_users"]))])
def update_permission(permission_id: str, permission: PermissionCreate, db: Session = Depends(get_db)):
    return modify_permission(permission_id, permission, db)

@router.get("")
def get_permissions(db: Session = Depends(get_db),
                    page:int = Query(1,ge=1),
                    limit : int= Query(0), 
                    sort_by: str = Query("name"), 
                    order:str = Query("asc"), 
                    search: str = Query("")):
    return retrieve_permissions(db, page, limit, sort_by, order, search)
    
@router.delete("/{permission_id}", response_model=Union[SuccessResponse, ErrorResponse], dependencies=[Depends(check_permissions(["manage_users"]))])
def delete_permission(permission_id: str, db: Session = Depends(get_db)):
    return delete_permission_by_id(permission_id, db)

@router.get("/{permission_id}")
def get_permission(permission_id: str, db: Session = Depends(get_db)):
    return retrieve_permission(permission_id, db)
