"""FastAPI dependency providers."""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..services.model_service import ModelService

DbSession = Annotated[Session, Depends(get_db)]


def get_model_service(request: Request) -> ModelService | None:
    """The ModelService lives on app.state (wired in create_app)."""
    return getattr(request.app.state, "model_service", None)


ModelSvc = Annotated[ModelService | None, Depends(get_model_service)]
