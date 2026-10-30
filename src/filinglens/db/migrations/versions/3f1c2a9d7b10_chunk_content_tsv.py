"""chunk.content_tsv: generated tsvector + GIN index for full-text search

Revision ID: 3f1c2a9d7b10
Revises: 2550cce7e41b
Create Date: 2026-10-30 18:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3f1c2a9d7b10"
down_revision: str | Sequence[str] | None = "2550cce7e41b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunk",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_chunk_content_tsv", "chunk", ["content_tsv"], unique=False, postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_content_tsv", table_name="chunk", postgresql_using="gin")
    op.drop_column("chunk", "content_tsv")
