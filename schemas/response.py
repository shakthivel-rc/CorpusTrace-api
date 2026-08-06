from typing import Any

from pydantic import BaseModel, Field

class SuccessResponse(BaseModel):
    status: str = "success"
    status_code: int
    message: str
    data: Any = Field(default_factory=dict)

class ErrorResponse(BaseModel):
    status: str = "error"
    status_code: int
    message: str
    data: Any = None