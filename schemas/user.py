from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from typing import List, Dict


# Request Schema (for login - accepts email or username)
class UserLoginRequest(BaseModel):
    username: str
    password: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class UserSignupReqeust(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    organization: str
    role_id: str
    department: str
    username: str


class PasswordSchema(BaseModel):
    new_password: str
    confirm_password: str

class ForgotPasswordSchema(BaseModel):
    email: EmailStr


class UserRegisterRequest(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    username: str
    organization: str = ""
    department: str = ""

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class ResendOTPRequest(BaseModel):
    email: EmailStr

class SetPasswordAfterRegRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str
    confirm_password: str

class ChangePasswordSchema(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str

class UserUpdateRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    organization: Optional[str] = None
    department: Optional[str] = None
    role_id: Optional[str] = None

class UserRoleSchema(BaseModel):
    role_id: str
    role_name: str

class UserPermissionSchema(BaseModel):
    permission_name: str
    machine_name: str

class UserProfileResponse(BaseModel):
    id: str
    username: str
    first_name: str
    last_name: str
    email: EmailStr
    organization: str
    department: str
    created_at: datetime
    updated_at: datetime
    roles: List[UserRoleSchema]
    permissions: List[UserPermissionSchema]

class UserProfileResponseWrapper(BaseModel):
    status: str
    status_code: int
    message: str
    data: UserProfileResponse

class UserUpdateResponse(BaseModel):
    status: str
    status_code: int
    message: str
    data: Dict[str, str]

class GetAllPermissionSchema(BaseModel):
    permission_name: str
    machine_name: str

class GetAllRoleSchema(BaseModel):
    role_name: str
    permissions: List[GetAllPermissionSchema]

class GetAllUserSchema(BaseModel):
    first_name: str
    last_name: str
    email: str
    department: str
    roles: List[GetAllRoleSchema]
