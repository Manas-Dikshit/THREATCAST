"""Application error hierarchy mapped onto the global API envelope.

Every handler converts these to:
    {"error": {"code", "message", "details", "request_id"}}
Internal stack traces are logged, never returned.
"""

import logging

logger = logging.getLogger("BACKEND")


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidInputError(AppError):
    status_code = 422
    code = "INVALID_INPUT"


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


class UnsupportedFormatError(InvalidInputError):
    code = "UNSUPPORTED_FORMAT"


class JobFailedError(AppError):
    status_code = 500
    code = "JOB_FAILED"


class ModelNotLoadedError(AppError):
    status_code = 503
    code = "MODEL_NOT_LOADED"


class DatabaseError(AppError):
    status_code = 500
    code = "DATABASE_ERROR"


def log_error(exc: Exception, request_id: str) -> None:
    logger.exception("[%s] %s: %s", request_id, type(exc).__name__, exc)


__all__ = [
    "AppError", "InvalidInputError", "NotFoundError", "PayloadTooLargeError",
    "UnsupportedFormatError", "JobFailedError", "ModelNotLoadedError",
    "DatabaseError", "log_error",
]
