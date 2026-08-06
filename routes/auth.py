from fastapi import APIRouter, HTTPException, Depends, Response,Request
from schemas.user import UserLoginRequest, UserRegisterRequest, PasswordSchema, ForgotPasswordSchema, VerifyOTPRequest, ResendOTPRequest, SetPasswordAfterRegRequest, RefreshTokenRequest
from controllers.auth_controller import login_controller, register_controller, verify_forgot_password_token, verify_token, set_user_password, handle_forgot_password, handle_reset_password, logout_controller, verify_otp_controller, resend_otp_controller, set_password_after_registration_controller, refresh_token_controller
from sqlalchemy.orm import Session
from db.session import get_db
from dependencies.auth import check_permissions

router = APIRouter(
    tags=["Authentication"]
)

@router.post("/user/login")
def login_user(user: UserLoginRequest, request: Request, db: Session = Depends(get_db)):
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    return login_controller(user, db, ip_address=ip_address, user_agent=user_agent)

@router.post("/user/refresh")
def refresh_user_token(schema: RefreshTokenRequest, db: Session = Depends(get_db)):
    return refresh_token_controller(schema, db)

@router.get("/user/verify-token")
def verify_account(response: Response, token: str, db: Session=Depends(get_db)):
    return verify_token(response, token, db)

@router.post("/user/set-password")
def set_password(response: Response, token: str, password_schema: PasswordSchema, db: Session=Depends(get_db)):
    return set_user_password(response, token, password_schema, db)

@router.post("/user/forgot-password")
def forgot_password(response: Response, schema: ForgotPasswordSchema, db: Session=Depends(get_db)):
    return handle_forgot_password(response, schema, db)

@router.get("/user/verify-forgot-password-token")
def verify_forgot_password(response: Response, token: str, db: Session=Depends(get_db)):
    return verify_forgot_password_token(response, token, db)

@router.post("/user/reset-password")
def reset_password(response: Response, token: str, password_schema: PasswordSchema, db: Session=Depends(get_db)):
    return handle_reset_password(response, token, password_schema, db)

@router.post("/user/register")
def register_user(response: Response, data: UserRegisterRequest, db: Session = Depends(get_db)):
    return register_controller(response, data, db)

@router.post("/user/verify-otp")
def verify_otp(response: Response, data: VerifyOTPRequest, db: Session = Depends(get_db)):
    return verify_otp_controller(response, data, db)

@router.post("/user/resend-otp")
def resend_otp(response: Response, data: ResendOTPRequest, db: Session = Depends(get_db)):
    return resend_otp_controller(response, data, db)

@router.post("/user/set-password-registration")
def set_password_registration(response: Response, data: SetPasswordAfterRegRequest, db: Session = Depends(get_db)):
    return set_password_after_registration_controller(response, data, db)

@router.post("/user/logout")
def logout(request:Request,db:Session=Depends(get_db)):
    return logout_controller(request,db)