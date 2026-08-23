"""FastAPI entrypoint. Phase 1: health endpoint + global error envelope only.

Full API (ingestion, predict, timelines, states) is implemented in a later phase
per docs/API.md.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api.routes import router
from .core.config import get_settings
from .core.logging import configure_logging, logger
from .schemas.errors import make_error


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="THREATCAST Backend",
        version="0.1.0",
        description="Predictive Cyber Defence world-model API. All endpoints live under /api/v1.",
    )
    app.include_router(router)

    def error_response(request_id: str, code: str, message: str, status: int) -> JSONResponse:
        return JSONResponse(status_code=status, content=make_error(code, message, request_id).model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(str(request.url.path), "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR", str(exc.detail), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(str(request.url.path), "INVALID_INPUT", "Request validation failed", 422)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return error_response(str(request.url.path), "INTERNAL_ERROR", "Internal server error", 500)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
