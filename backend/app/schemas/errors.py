"""Global API error envelope (CONTRACT.md §10)."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


def make_error(code: str, message: str, request_id: str = "", **details: Any) -> ErrorEnvelope:
    return ErrorEnvelope(
        error=ErrorDetail(code=code, message=message, details=details, request_id=request_id)
    )
