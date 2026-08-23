"""baseline: all Phase 4 tables

Revision ID: 0001_baseline
Revises:
Create Date: 2026-08-23
"""
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Single source of truth: the SQLAlchemy metadata (backend/app/models/tables.py).
    from app.db.base import Base
    from app.models import tables  # noqa: F401

    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    from app.db.base import Base
    from app.models import tables  # noqa: F401

    Base.metadata.drop_all(bind=op.get_bind())
