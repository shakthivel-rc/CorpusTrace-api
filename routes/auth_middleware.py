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

class TokenRejected(Exception):
    """An access token that cannot authenticate a caller.

    The message is the one reported to the client verbatim, so the wording lives here
    rather than at each call site — the middleware and the chat WebSocket must not drift
    into telling a user two different things about the same token.
    """


def decode_access_token(token: str) -> dict:
    """The token's claims, or `TokenRejected`. Touches no database.

    Split from the session lookup deliberately: an unauthenticated caller with a garbage
    token must not cost a connection checkout, and `user_sessions.access_token` is an
    unindexed VARCHAR(1024) that every authenticated request already full-scans once.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except jwt.ExpiredSignatureError:
        raise TokenRejected("Token has expired") from None
    except jwt.InvalidTokenError:
        raise TokenRejected("Invalid token") from None

    if payload.get("type", "access") != "access":
        raise TokenRejected("Invalid token type")
    return payload


def assert_session_is_live(db: Session, token: str, payload: dict) -> None:
    """Raise `TokenRejected` unless a live `user_sessions` row still backs this token.

    This is what makes logout and revocation take effect: a signed, unexpired JWT whose
    session row is gone must stop working immediately (business rule 3).
    """
    user_session = db.query(UserSession).filter(
        UserSession.access_token == token,
        UserSession.revoked_at.is_(None),
    ).first()
    if not user_session:
        raise TokenRejected("Invalid session or token revoked")

    if payload.get("exp", 0) < time.time():
        raise TokenRejected("Token has expired")


def verify_access_token(db: Session, token: str) -> dict:
    """Both halves of the check, for callers that already hold a session.

    The middleware does not use this — it opens its session only once the decode has
    passed — but the WebSocket handler runs entirely on one worker thread and has no
    reason to split them.
    """
    payload = decode_access_token(token)
    assert_session_is_live(db, token, payload)
    return payload


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Authenticates every HTTP request outside the allow-list.

    `BaseHTTPMiddleware` only ever sees `scope["type"] == "http"`, so a WebSocket
    connection passes straight through it untouched — which is why `/chat/ws`
    authenticates itself from its first frame using the helpers above rather than
    assuming `request.state.user` was populated for it.
    """

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
            payload = decode_access_token(token)
        except TokenRejected as exc:
            error_response = ErrorResponse(status_code=401, message=str(exc), data={})
            return JSONResponse(status_code=401, content=error_response.model_dump())

        db: Session = SessionLocal()
        try:
            assert_session_is_live(db, token, payload)
        except TokenRejected as exc:
            error_response = ErrorResponse(status_code=401, message=str(exc), data={})
            return JSONResponse(status_code=401, content=error_response.model_dump())
        finally:
            db.close()

        request.state.user = payload  # Attach user info to request
        return await call_next(request)
