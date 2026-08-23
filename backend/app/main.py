"""FastAPI entrypoint.

Wires: routers, global error envelope, request-id middleware, CORS (env),
model loading at startup (graceful when artifacts are missing).
"""

import logging
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api.routers import ingestion, predictions, states
from .api.routes import router
from .core.config import get_settings
from .core.errors import AppError, log_error
from .core.logging import configure_logging, logger
from .schemas.errors import make_error

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="THREATCAST Backend",
        version="0.1.0",
        description="Predictive Cyber Defence world-model API. All endpoints live under /api/v1.",
    )

    # ---- CORS from environment ----
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # ---- model service on app.state ----
    from .services.model_service import ModelService

    app.state.model_service = ModelService(
        artifacts_dir=settings.ml_artifacts_dir or str(
            (settings.ml_model_path and __import__("pathlib").Path(settings.ml_model_path).parent)
        ),
        device=settings.ml_device,
    )

    @app.on_event("startup")
    def _load_model() -> None:
        app.state.model_service.load()

    # ---- request id + error envelope ----
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = f"req_{uuid.uuid4().hex[:12]}"
        request_id_ctx.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    def _rid(request: Request) -> str:
        return request_id_ctx.get("") or f"req_{uuid.uuid4().hex[:12]}"

    def error_response(request: Request, code: str, message: str,
                       status: int, **details) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content=make_error(code, message, _rid(request), **details).model_dump(),
        )

    STATUS_CODES = {404: "NOT_FOUND", 405: "NOT_ALLOWED", 413: "PAYLOAD_TOO_LARGE",
                    422: "INVALID_INPUT", 503: "MODEL_NOT_LOADED"}

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
        return error_response(request, code, str(exc.detail), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("[%s] validation failed: %s", _rid(request), exc.errors()[:3])
        return error_response(request, "INVALID_INPUT",
                              "Request validation failed", 422,
                              errors=exc.errors()[:5])

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        log_error(exc, _rid(request))
        return error_response(request, exc.code, exc.message,
                              exc.status_code, **exc.details)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        log_error(exc, _rid(request))
        return error_response(request, "INTERNAL_ERROR",
                              "Internal server error", 500)

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.backend_host,
                port=settings.backend_port, reload=True)
