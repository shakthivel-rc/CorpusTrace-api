from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session
from db.session import get_db
from controllers.activity_log_controller import get_activity_logs
from dependencies.auth import check_permissions

router = APIRouter(
    prefix="/activity-logs",
    tags=["Activity Logs"]
)


@router.get("")
def fetch_activity_logs(
    request: Request,
    db: Session = Depends(get_db),
    user_id: str = Query(None),
    action: str = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(25),
):
    """
    Get activity logs.
    Users with manage_users permission see all logs.
    Other users only see their own logs.
    """
    current_user = getattr(request.state, "user", None)
    if not current_user:
        from fastapi.responses import JSONResponse
        from schemas.response import ErrorResponse
        error_response = ErrorResponse(status_code=401, message="Not authenticated")
        return JSONResponse(status_code=401, content=error_response.model_dump())

    current_user_id = current_user.get("sub")

    # Check if user has manage_users permission
    # If not admin, force filter to own logs only
    has_manage_users = _has_permission(db, current_user_id, "manage_users")

    if not has_manage_users:
        user_id = current_user_id

    return get_activity_logs(db, user_id=user_id, action=action, page=page, limit=limit)


def _has_permission(db: Session, user_id: str, permission_machine_name: str) -> bool:
    """Check if a user has a specific permission via their roles."""
    from models.user import User
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return False
    for role in user.roles:
        for perm in role.permissions:
            if perm.machine_name == permission_machine_name:
                return True
    return False
