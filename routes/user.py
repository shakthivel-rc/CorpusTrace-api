from fastapi import APIRouter, Depends, Request, Response, Query
from schemas.user import UserSignupReqeust, UserUpdateRequest, ChangePasswordSchema
from sqlalchemy.orm import Session
from controllers.user_controller import add_user_controller, delete_user_controller, get_user_profile, update_user_controller,get_users_info, get_user, change_password_controller
from db.session import get_db
from dependencies.auth import check_permissions
from models.user import User
from schemas.user_role import UserRoleAssignRequest, UserRoleResponse
from typing import Dict, Any,Union
from schemas.response import SuccessResponse,ErrorResponse
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

@router.post("", dependencies=[Depends(check_permissions(["manage_users"]))])
def add_user(user: UserSignupReqeust, response: Response, request: Request, db: Session = Depends(get_db)):
    return add_user_controller(response, user, db, request=request)

@router.get("/profile")
def get_profile(request: Request,db: Session = Depends(get_db)):
    return get_user_profile(request, db)

@router.post("/change-password")
def change_password(request: Request, data: ChangePasswordSchema, db: Session = Depends(get_db)):
    return change_password_controller(request, data, db)

@router.put("/{user_id}", response_model=Union[SuccessResponse, ErrorResponse], dependencies=[Depends(check_permissions(["manage_users"]))])
def update_user(user_id: str, update_data: UserUpdateRequest, request: Request, db: Session = Depends(get_db)):
    return update_user_controller(user_id, update_data, db, request=request)

@router.delete("/{user_id}", response_model=Union[SuccessResponse, ErrorResponse], dependencies=[Depends(check_permissions(["manage_users"]))])
def delete_user(user_id: str, request: Request, db: Session = Depends(get_db)):
    return delete_user_controller(user_id, db, request=request)

@router.get("/{user_id}")
def get_user_info(user_id: str, request: Request,db: Session = Depends(get_db)):
    return get_user(user_id, request, db)

@router.get("")
def get_users(db: Session = Depends(get_db), page: int = Query(1,ge=1), limit:int=Query(0), sort_by:str=Query("first_name"), order:str=Query("asc"), search:str=Query("")):
    return get_users_info(db, page, limit, sort_by, order, search)