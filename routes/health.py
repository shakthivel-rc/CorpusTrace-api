from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.response import SuccessResponse


router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get("/live")
def live():
    return SuccessResponse(status_code=200, message="NexaRAG API is live", data={"status": "live"})


@router.get("/ready")
def ready(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return SuccessResponse(status_code=200, message="NexaRAG API is ready", data={"database": "ok"})


@router.get("/dependencies")
def dependencies(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return SuccessResponse(
        status_code=200,
        message="Dependencies are healthy",
        data={"database": "ok"},
    )
