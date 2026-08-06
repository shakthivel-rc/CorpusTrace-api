from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from models.user import User
from services.user import get_user_by_id
from sqlalchemy.orm import Session
from db.session import get_db
from schemas.response import ErrorResponse
from core.permissions import permission_is_satisfied

def check_permissions(required_permissions: list[str]):
    def role_checker(request: Request, db: Session = Depends(get_db)):
        user = getattr(request.state, "user", None)
        if not user:
            return ErrorResponse(status="error",
                                status_code=401,
                                message="Not authenticated",
                                data={})
     
        user_id = user.get("sub")
        user = get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=401,
                detail=ErrorResponse(
                    status_code=401,
                    message="Not authenticated",
                    data={},
                ).model_dump(),
            )
        table_user_permissions = []
        for role in user.roles:
            for permission in role.permissions:
                table_user_permissions.append(permission.machine_name)
        user_permissions = set(table_user_permissions)
        if all(permission_is_satisfied(user_permissions, permission) for permission in required_permissions):
            return user
        
        error_response = ErrorResponse(
            status="error",
            status_code=403,
            message="Insufficient permissions",
            
        )
        raise HTTPException(status_code=403, detail=error_response.model_dump())

    return role_checker
