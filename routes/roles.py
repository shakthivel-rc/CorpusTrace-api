from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from db.session import get_db
from dependencies.auth import check_permissions
from schemas.roles import RolePermissionUpdateRequest, RoleUpdateRequest, RolePermissionDeleteRequest, RoleCreateRequest
from controllers.roles import fetch_all_roles, fetch_role, remove_role, add_role, update_role_controller
from schemas.response import SuccessResponse,ErrorResponse
from typing import Union


router = APIRouter(
    prefix="/roles",
    tags=["Roles"]
)

@router.delete("/{role_id}", response_model=Union[SuccessResponse, ErrorResponse], dependencies=[Depends(check_permissions(["manage_users"]))])
def delete_role(role_id: str, db: Session = Depends(get_db)):
    return remove_role(role_id, db)

@router.post("", dependencies=[Depends(check_permissions(["manage_users"]))])
def create_role(role: RoleCreateRequest, db: Session = Depends(get_db)):
    return add_role(role, db)

@router.get("")
def get_roles(db:Session=Depends(get_db),page: int=Query(1,ge=1), limit: int=Query(0), sort_by: str=Query("name"), order: str= Query("asc"), search: str=Query("")):
    return fetch_all_roles(db, page, limit, sort_by, order, search)

@router.get("/{role_id}")
def get_role(role_id: str, db: Session = Depends(get_db)):
    return fetch_role(role_id, db)

@router.put("/{role_id}", response_model=Union[SuccessResponse, ErrorResponse], dependencies=[Depends(check_permissions(["manage_users"]))])
def add_permissions(role_id: str, request: RolePermissionUpdateRequest, http_request: Request, db: Session = Depends(get_db)):
    return update_role_controller(role_id, request, db, http_request=http_request)
 