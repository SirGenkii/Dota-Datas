from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0002_add_match_raw"
down_revision = "0001_init_web_stack"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "match_raw",
        sa.Column("match_id", sa.BigInteger(), primary_key=True, nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_match_raw_fetched_at", "match_raw", ["fetched_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_match_raw_fetched_at", table_name="match_raw")
    op.drop_table("match_raw")

