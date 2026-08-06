import jwt
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session

from core.config import get_settings
from db.session import SessionLocal
from models.user_session import UserSession
from schemas.response import ErrorResponse


settings = get_settings()

UNAUTHENTICATED_ROUTES = [
    "/user/login",
    "/user/register",
    "/user/refresh",
    "/user/verify-token",
    "/user/set-password",
    "/user/forgot-password",
    "/user/verify-forgot-password-token",
    "/user/reset-password",
    "/user/verify-otp",
    "/user/resend-otp",
    "/user/set-password-registration",
]
UNAUTHENTICATED_PREFIXES = (
    "/api/v1/health",
)

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in UNAUTHENTICATED_ROUTES:
            return await call_next(request)
        if request.url.path.startswith(UNAUTHENTICATED_PREFIXES):
            return await call_next(request)
        if request.url.path in {"/docs", "/openapi.json"}:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            error_response = ErrorResponse(
                    status_code=401, message="Missing or invalid token", data={}
                )
            return JSONResponse(status_code=401, content=error_response.model_dump())

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            if payload.get("type", "access") != "access":
                error_response = ErrorResponse(
                    status_code=401, message="Invalid token type", data={}
                )
                return JSONResponse(status_code=401, content=error_response.model_dump())

            db: Session = SessionLocal()
            try:
                user_session = db.query(UserSession).filter(
                    UserSession.access_token == token,
                    UserSession.revoked_at.is_(None),
                ).first()
            finally:
                db.close()
            if not user_session:
                error_response = ErrorResponse(
                    status_code=401, message="Invalid session or token revoked", data={}
                )
                response = JSONResponse(status_code=401, content=error_response.model_dump())
                return response

            if payload.get("exp", 0) < time.time():
                error_response = ErrorResponse(
                    status_code=401, message="Token has expired", data={}
                )
                return JSONResponse(status_code=401, content=error_response.model_dump())

            request.state.user = payload  # Attach user info to request
        except jwt.ExpiredSignatureError:
            error_response = ErrorResponse(
                status_code=401, message="Token has expired", data={}
            )
            return JSONResponse(status_code=401, content=error_response.model_dump())
        except jwt.InvalidTokenError:
            error_response = ErrorResponse(
                status_code=401,
                message="Invalid token",
                data={}
            )
            return JSONResponse(status_code=401, content=error_response.model_dump())

        return await call_next(request)
