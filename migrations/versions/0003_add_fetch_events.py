from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0003_add_fetch_events"
down_revision = "0002_add_match_raw"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fetch_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_fetch_events_run_id", "fetch_events", ["run_id"], unique=False)
    op.create_index("ix_fetch_events_endpoint", "fetch_events", ["endpoint"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_fetch_events_endpoint", table_name="fetch_events")
    op.drop_index("ix_fetch_events_run_id", table_name="fetch_events")
    op.drop_table("fetch_events")

