from sqlalchemy.orm import Session
from models.activity_log import ActivityLog


def log_activity(
    db: Session,
    user_id: str,
    user_name: str,
    action: str,
    entity_type: str = None,
    entity_id: str = None,
    details: str = None,
    ip_address: str = None
):
    """Log an activity entry. Fire-and-forget — errors are silently ignored."""
    try:
        entry = ActivityLog(
            user_id=user_id,
            user_name=user_name,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
            ip_address=ip_address,
        )
        db.add(entry)
        db.flush()
    except Exception:
        pass  # Never let logging failures break business logic
