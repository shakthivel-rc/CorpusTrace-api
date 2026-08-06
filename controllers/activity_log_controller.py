from sqlalchemy.orm import Session
from sqlalchemy import desc
from models.activity_log import ActivityLog
from schemas.response import SuccessResponse, ErrorResponse
from fastapi.responses import JSONResponse


def get_activity_logs(
    db: Session,
    user_id: str = None,
    action: str = None,
    page: int = 1,
    limit: int = 25,
):
    """Retrieve activity logs with optional filtering."""
    try:
        query = db.query(ActivityLog)

        if user_id:
            query = query.filter(ActivityLog.user_id == user_id)
        if action:
            query = query.filter(ActivityLog.action == action.upper())

        total = query.count()
        query = query.order_by(desc(ActivityLog.created_at))

        if limit > 0:
            skip = (page - 1) * limit
            query = query.offset(skip).limit(limit)

        logs = query.all()

        records = [
            {
                "id": log.id,
                "userId": log.user_id,
                "userName": log.user_name or "",
                "action": log.action,
                "type": log.action.lower() if log.action else "login",
                "details": log.details or "",
                "ipAddress": log.ip_address or "",
                "timestamp": log.created_at.isoformat() if log.created_at else "",
            }
            for log in logs
        ]

        return SuccessResponse(
            status_code=200,
            message="Activity logs fetched successfully",
            data={"records": records, "total": total},
        )

    except Exception as e:
        error_response = ErrorResponse(status_code=500, message=f"Failed to fetch activity logs: {str(e)}")
        return JSONResponse(status_code=500, content=error_response.model_dump())
