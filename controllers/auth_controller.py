from fastapi import Response, status
from fastapi.responses import JSONResponse
from schemas.user import UserLoginRequest, UserRegisterRequest, PasswordSchema, ForgotPasswordSchema, VerifyOTPRequest, ResendOTPRequest, SetPasswordAfterRegRequest, RefreshTokenRequest
from schemas.response import SuccessResponse, ErrorResponse
from schemas.email_schema import EmailTemplate
from sqlalchemy.orm import Session
from db.session import get_db
from services.user import get_user_by_username, get_user_by_email
import datetime
import hashlib
import random
import bcrypt
import jwt
import os
import uuid
from dotenv import load_dotenv
from core.config import get_settings
from utils.token import decode_base64, decrypt_string, generate_random_code, encode_to_base64, encrypt_string
from utils.password import hash_password, validate_password_policy
from models.user import User
from integrations.aws_ses import send_email
from models.user_session import UserSession
from utils.constants import reset_password_email_template, otp_email_template
from services.activity_log import log_activity

load_dotenv()

settings = get_settings()
SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
JWT_EXPIRY_VALUE = settings.jwt_expiry_hours


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _password_policy_error(password: str):
    errors = validate_password_policy(password)
    if errors:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                status_code=400,
                message="Password does not meet policy requirements",
                data={"errors": errors},
            ).model_dump(),
        )
    return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify if the password matches the stored hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_jwt_token(user_id: str) -> dict:
    """Generate distinct access and refresh JWTs."""
    now = _utcnow()
    access_expires_at = now + datetime.timedelta(hours=JWT_EXPIRY_VALUE)
    refresh_expires_at = now + datetime.timedelta(days=settings.refresh_token_expiry_days)
    access_payload = {
        "sub": user_id,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": access_expires_at,
    }
    refresh_payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": refresh_expires_at,
    }
    access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
    refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "refresh_token_hash": _hash_token(refresh_token),
        "refresh_token_expires_at": refresh_expires_at.replace(tzinfo=None),
    }
    
def login_controller(user: UserLoginRequest, db: Session, ip_address: str = None, user_agent: str = None) -> SuccessResponse:
    # Try lookup by email first, then by username
    user_details = get_user_by_email(db, user.username)
    if not user_details:
        user_details = get_user_by_username(db=db, username=user.username)
    if not user_details:
        error_response = ErrorResponse(status_code=401, message="Invalid username or password")
        return JSONResponse(status_code=401, content=error_response.model_dump())
    if not user_details.status:
        error_response = ErrorResponse(status_code=401, message="Verify your account to login")
        return JSONResponse(status_code=401, content=error_response.model_dump())

    now = datetime.datetime.utcnow()
    if user_details.locked_until and user_details.locked_until > now:
        error_response = ErrorResponse(
            status_code=423,
            message="Account is temporarily locked. Please try again later.",
        )
        return JSONResponse(status_code=423, content=error_response.model_dump())

    if not verify_password(user.password, user_details.password):
        user_details.failed_login_attempts = (user_details.failed_login_attempts or 0) + 1
        user_details.last_failed_login_at = now
        if user_details.failed_login_attempts >= settings.auth_max_failed_login_attempts:
            user_details.locked_until = now + datetime.timedelta(minutes=settings.auth_lockout_minutes)
            db.commit()
            error_response = ErrorResponse(
                status_code=423,
                message="Account is temporarily locked. Please try again later.",
            )
            return JSONResponse(status_code=423, content=error_response.model_dump())
        db.commit()
        error_response = ErrorResponse(status_code=401, message="Invalid username or password")
        return JSONResponse(status_code=401, content=error_response.model_dump())
       
    token = create_jwt_token(user_details.id)
    #Generate the JWT with access and refresh token
        
    user_session = UserSession(
        user_id=user_details.id,
        access_token = token["access_token"],
        refresh_token=None,
        refresh_token_hash=token["refresh_token_hash"],
        refresh_token_expires_at=token["refresh_token_expires_at"],
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(user_session)

    user_details.failed_login_attempts = 0
    user_details.locked_until = None
    user_details.last_failed_login_at = None

    user_display_name = f"{user_details.first_name} {user_details.last_name}"
    log_activity(db, user_details.id, user_display_name, "LOGIN", details="Successful login", ip_address=ip_address)

    db.commit()
    db.refresh(user_session)

    return SuccessResponse(status="success", status_code=200, message="Login successful", data={"access_token": token["access_token"], "refresh_token": token["refresh_token"]})


def refresh_token_controller(schema: RefreshTokenRequest, db: Session) -> SuccessResponse:
    try:
        payload = jwt.decode(schema.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        error_response = ErrorResponse(status_code=401, message="Refresh token has expired")
        return JSONResponse(status_code=401, content=error_response.model_dump())
    except jwt.InvalidTokenError:
        error_response = ErrorResponse(status_code=401, message="Invalid refresh token")
        return JSONResponse(status_code=401, content=error_response.model_dump())

    if payload.get("type") != "refresh":
        error_response = ErrorResponse(status_code=401, message="Invalid token type")
        return JSONResponse(status_code=401, content=error_response.model_dump())

    refresh_token_hash = _hash_token(schema.refresh_token)
    session = db.query(UserSession).filter(
        UserSession.refresh_token_hash == refresh_token_hash,
        UserSession.revoked_at.is_(None),
    ).first()
    if not session:
        error_response = ErrorResponse(status_code=401, message="Invalid or revoked refresh token")
        return JSONResponse(status_code=401, content=error_response.model_dump())

    if session.refresh_token_expires_at and session.refresh_token_expires_at < datetime.datetime.utcnow():
        session.revoked_at = datetime.datetime.utcnow()
        db.commit()
        error_response = ErrorResponse(status_code=401, message="Refresh token has expired")
        return JSONResponse(status_code=401, content=error_response.model_dump())

    token = create_jwt_token(session.user_id)
    session.access_token = token["access_token"]
    session.refresh_token = None
    session.refresh_token_hash = token["refresh_token_hash"]
    session.refresh_token_expires_at = token["refresh_token_expires_at"]
    db.commit()

    return SuccessResponse(
        status="success",
        status_code=200,
        message="Token refreshed successfully",
        data={"access_token": token["access_token"], "refresh_token": token["refresh_token"]},
    )

# verify if the token is valid
def verify_token(response: Response, token: str, db: Session):
    decrypted_token = ""
    try:
        decoded_token = decode_base64(token)
        decrypted_token = decrypt_string(decoded_token)
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Invalid Token", data={})
    user = db.query(User).filter(User.verify_token == decrypted_token).first()
    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=400, message="User doesn't exists with this token")

    # check if the token is expired
    if datetime.datetime.now() > (user.created_at + datetime.timedelta(hours=48)):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="The verification token is expired")
       
    return SuccessResponse(status="success", status_code=200, message="user verified successfully", data={})

# verify if the token is valid
def verify_forgot_password_token(response: Response, token: str, db: Session):
    decrypted_token = ""
    try:
        decoded_token = decode_base64(token)
        decrypted_token = decrypt_string(decoded_token)
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Invalid Token", data={})
    user = db.query(User).filter(User.fp_token == decrypted_token).first()
    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=400, message="User doesn't exists with this token")

    # check if the token is expired
    if datetime.datetime.now() > (user.fp_token_created_at + datetime.timedelta(hours=48)):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="The verification token is expired")
       
    return SuccessResponse(status="success", status_code=200, message="user verified successfully", data={})

# to set the password
def set_user_password(response: Response, token: str, password_schema: PasswordSchema, db: Session):
    decrypted_token = ""
    try:
        decoded_token = decode_base64(token)
        decrypted_token = decrypt_string(decoded_token)
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Invalid Token", data={})
    user = db.query(User).filter(User.verify_token == decrypted_token).first()
    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=400, message="User doesn't exists with this token")

    if datetime.datetime.now() > (user.created_at + datetime.timedelta(hours=48)):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="The verification token is expired")

    # check if new password and confirm password are the same
    if password_schema.new_password != password_schema.confirm_password:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Passwords don't match", data=None)

    password_error = _password_policy_error(password_schema.new_password)
    if password_error:
        return password_error

    user.password = hash_password(password_schema.new_password) 
    user.status = True
    user.verify_token = None
    db.commit()

    response.status_code = status.HTTP_201_CREATED
    return SuccessResponse(status="success", message="Password set successfully", status_code=201, data={})

# send email for the user to reset password
def handle_forgot_password(response: Response, schema: ForgotPasswordSchema, db: Session):
    user = get_user_by_email(db, schema.email)
    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=response.status_code, message="No user found with this email")
    
    if not user.status:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=response.status_code, message="Unverified users cannot reset their password")
    
    # setting up the email
    web_base_url = os.getenv("APP_BASE_URL")
    token = generate_random_code(16)
    encoded_token = encode_to_base64(encrypt_string(token))
    reset_password_url = f'{web_base_url}/reset-password/{encoded_token}'
    email_template = EmailTemplate(subject="Reset Your CorpusTrace Password",body=reset_password_email_template(user.first_name, reset_password_url))
    # sending the email
    email_response = send_email(recipient_email=user.email,email_template=email_template)
    # throw error when email can not be sent
    if not email_response["status"]:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ErrorResponse(status="error", status_code=500, message="Error occurred sending the email. make sure the email you entered is correct")

    user.fp_token = token
    user.fp_token_created_at = datetime.datetime.now()
    db.commit()
    response.status_code = status.HTTP_200_OK
    return SuccessResponse(status="success", status_code=response.status_code, message="Reset password email sent successfully", data={})

# To reset the password
def handle_reset_password(response: Response, token: str, schema: PasswordSchema, db:Session):
    decrypted_token = ""
    try:
        decoded_token = decode_base64(token)
        decrypted_token = decrypt_string(decoded_token)
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Invalid Token", data={})
    user = db.query(User).filter(User.fp_token == decrypted_token).first()
    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=400, message="User doesn't exists with this token")

    # check if the token is expired
    if datetime.datetime.now() > (user.fp_token_created_at + datetime.timedelta(hours=48)):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="The verification token is expired")

    # check if new password and confirm password are the same
    if schema.new_password != schema.confirm_password:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Passwords don't match", data=None)

    password_error = _password_policy_error(schema.new_password)
    if password_error:
        return password_error

    user.password = hash_password(schema.new_password) 
    user.fp_token = None
    user.fp_token_created_at = None
    db.query(UserSession).filter(UserSession.user_id == user.id).update(
        {UserSession.revoked_at: datetime.datetime.utcnow()},
        synchronize_session=False,
    )
    db.commit()

    response.status_code = status.HTTP_201_CREATED
    return SuccessResponse(status="success", message="Password set successfully", status_code=201, data={})

def logout_controller(request, db):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return ErrorResponse(status="error", status_code=401, message="Missing or invalid token")

    token = auth_header.split(" ")[1]
    session = db.query(UserSession).filter(UserSession.access_token == token).first()

    if not session:
        return ErrorResponse(status_code=400, message="Invalid session or already logged out")

    # Log activity before deleting session
    user_payload = getattr(request.state, "user", None)
    if user_payload:
        user_id = user_payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()
        user_name = f"{user.first_name} {user.last_name}" if user else ""
        ip_address = request.client.host if request.client else None
        log_activity(db, user_id, user_name, "LOGOUT", details="User logged out", ip_address=ip_address)

    # Delete the session from the database
    db.delete(session)
    db.commit()

    return SuccessResponse(
        status="success",
        message="Logged out successfully",
        status_code=200,
        data={}
    )


def _generate_otp() -> str:
    """Generate a 6-digit OTP."""
    return str(random.randint(100000, 999999))


def _hash_otp(otp: str) -> str:
    """Hash OTP using bcrypt."""
    return bcrypt.hashpw(otp.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def _verify_otp(plain_otp: str, hashed_otp: str) -> bool:
    """Verify OTP against its hash."""
    return bcrypt.checkpw(plain_otp.encode('utf-8'), hashed_otp.encode('utf-8'))


def register_controller(response: Response, data: UserRegisterRequest, db: Session):
    """Self-registration step 1: collect info, generate OTP, send email."""
    # Check if email exists and is already verified
    existing_email = get_user_by_email(db, data.email)
    if existing_email and existing_email.status:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Email already registered")

    # If unverified user exists with same email, remove it to allow re-registration
    if existing_email and not existing_email.status:
        db.delete(existing_email)
        db.flush()

    # Check username uniqueness (excluding soft-deleted and the just-deleted unverified)
    existing_username = get_user_by_username(db, data.username)
    if existing_username:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Username already taken")

    # Generate OTP
    otp = _generate_otp()
    hashed_otp = _hash_otp(otp)

    # Create user without password, status=False (unverified)
    new_user = User(
        email=data.email,
        first_name=data.first_name,
        last_name=data.last_name,
        username=data.username,
        password=None,
        organization=data.organization,
        department=data.department,
        status=False,
        otp_code=hashed_otp,
        otp_created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db.add(new_user)
    db.flush()

    # Send OTP email
    email_template = EmailTemplate(
        subject="Your CorpusTrace Verification Code",
        body=otp_email_template(data.first_name, otp)
    )
    email_response = send_email(recipient_email=data.email, email_template=email_template)
    if not email_response["status"]:
        db.rollback()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ErrorResponse(status="error", status_code=500, message="Failed to send verification email. Please try again.")

    db.commit()

    response.status_code = status.HTTP_201_CREATED
    return SuccessResponse(
        status="success",
        status_code=201,
        message="OTP sent to your email. Please verify to continue.",
        data={"email": data.email}
    )


def verify_otp_controller(response: Response, data: VerifyOTPRequest, db: Session):
    """Verify the 6-digit OTP sent during registration."""
    user = db.query(User).filter(
        User.email == data.email,
        User.status == False,
        User.deleted == 0
    ).first()

    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=404, message="No pending registration found for this email")

    if not user.otp_code or not user.otp_created_at:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="No OTP found. Please request a new one.")

    # Check OTP expiry (10 minutes)
    now = datetime.datetime.now(datetime.timezone.utc)
    otp_created = user.otp_created_at.replace(tzinfo=datetime.timezone.utc) if user.otp_created_at.tzinfo is None else user.otp_created_at
    if now > (otp_created + datetime.timedelta(minutes=10)):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="OTP has expired. Please request a new one.")

    # Verify OTP
    if not _verify_otp(data.otp, user.otp_code):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Invalid OTP. Please try again.")

    return SuccessResponse(
        status="success",
        status_code=200,
        message="OTP verified successfully. Please set your password.",
        data={"verified": True, "email": data.email}
    )


def resend_otp_controller(response: Response, data: ResendOTPRequest, db: Session):
    """Resend OTP for an unverified registration."""
    user = db.query(User).filter(
        User.email == data.email,
        User.status == False,
        User.deleted == 0
    ).first()

    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=404, message="No pending registration found for this email")

    # Rate limit: 60 seconds between resends
    if user.otp_created_at:
        now = datetime.datetime.now(datetime.timezone.utc)
        otp_created = user.otp_created_at.replace(tzinfo=datetime.timezone.utc) if user.otp_created_at.tzinfo is None else user.otp_created_at
        elapsed = (now - otp_created).total_seconds()
        if elapsed < 60:
            remaining = int(60 - elapsed)
            response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
            return ErrorResponse(status="error", status_code=429, message=f"Please wait {remaining} seconds before requesting a new OTP")

    # Generate and store new OTP
    otp = _generate_otp()
    user.otp_code = _hash_otp(otp)
    user.otp_created_at = datetime.datetime.now(datetime.timezone.utc)

    # Send OTP email
    email_template = EmailTemplate(
        subject="Your CorpusTrace Verification Code",
        body=otp_email_template(user.first_name, otp)
    )
    email_response = send_email(recipient_email=user.email, email_template=email_template)
    if not email_response["status"]:
        db.rollback()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ErrorResponse(status="error", status_code=500, message="Failed to send verification email. Please try again.")

    db.commit()

    return SuccessResponse(
        status="success",
        status_code=200,
        message="New OTP sent to your email.",
        data={"email": data.email}
    )


def set_password_after_registration_controller(response: Response, data: SetPasswordAfterRegRequest, db: Session):
    """Final registration step: set password after OTP verification."""
    user = db.query(User).filter(
        User.email == data.email,
        User.status == False,
        User.deleted == 0
    ).first()

    if not user:
        response.status_code = status.HTTP_404_NOT_FOUND
        return ErrorResponse(status="error", status_code=404, message="No pending registration found for this email")

    # Re-verify OTP is still valid
    if not user.otp_code or not user.otp_created_at:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="No OTP found. Please restart registration.")

    now = datetime.datetime.now(datetime.timezone.utc)
    otp_created = user.otp_created_at.replace(tzinfo=datetime.timezone.utc) if user.otp_created_at.tzinfo is None else user.otp_created_at
    if now > (otp_created + datetime.timedelta(minutes=10)):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="OTP has expired. Please restart registration.")

    if not _verify_otp(data.otp, user.otp_code):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Invalid OTP.")

    # Validate passwords match
    if data.new_password != data.confirm_password:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(status="error", status_code=400, message="Passwords don't match")

    password_error = _password_policy_error(data.new_password)
    if password_error:
        return password_error

    # Set password and activate user
    user.password = hash_password(data.new_password)
    user.status = True
    user.otp_code = None
    user.otp_created_at = None

    # Assign default "Customer" role
    from models.roles import Role
    from models.user_roles import UserRole
    default_role = db.query(Role).filter(Role.name == "Customer").first()
    if default_role:
        user_role = UserRole(user_id=user.id, role_id=default_role.id)
        db.add(user_role)

    log_activity(db, user.id, f"{user.first_name} {user.last_name}", "CREATE", entity_type="user", entity_id=user.id, details="Self-registration completed")

    db.commit()

    response.status_code = status.HTTP_201_CREATED
    return SuccessResponse(
        status="success",
        status_code=201,
        message="Account created successfully. You can now log in.",
        data={"user_id": user.id}
    )