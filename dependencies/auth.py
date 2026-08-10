from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from models.user import User
from services.user import get_user_by_id
from sqlalchemy.orm import Session
from db.session import get_db
from schemas.response import ErrorResponse
from core.permissions import permission_is_satisfied

def permission_slugs(user: User) -> set[str]:
    """Every permission slug this user's roles grant.

    Extracted so the chat WebSocket enforces `ai_access` with the same walk this
    dependency does. A WS connection cannot go through FastAPI's dependency system — there
    is no `Request`, and `JWTAuthMiddleware` never sees a websocket scope — so the only
    alternative was a second copy of the role→permission traversal, which is exactly the
    kind of authorization logic that must never exist twice.
    """
    return {permission.machine_name for role in user.roles for permission in role.permissions}


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
        user_permissions = permission_slugs(user)
        if all(permission_is_satisfied(user_permissions, permission) for permission in required_permissions):
            return user
        
        error_response = ErrorResponse(
            status="error",
            status_code=403,
            message="Insufficient permissions",
            
        )
        raise HTTPException(status_code=403, detail=error_response.model_dump())

    return role_checker
