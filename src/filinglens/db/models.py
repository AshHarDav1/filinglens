"""Database models. financial_fact (XBRL numbers) arrives in M1; the tsvector column in M2."""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimension of the embedding model (text-embedding-3-small). Changing the model means a migration.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "company"

    id: Mapped[int] = mapped_column(primary_key=True)
    cik: Mapped[int] = mapped_column(BigInteger, unique=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(255))

    filings: Mapped[list["Filing"]] = relationship(back_populates="company")


class Filing(Base):
    __tablename__ = "filing"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("company.id"), index=True)
    accession_no: Mapped[str] = mapped_column(String(32), unique=True)
    form: Mapped[str] = mapped_column(String(16))
    fiscal_year: Mapped[int] = mapped_column(Integer)
    period_end: Mapped[date] = mapped_column(Date)
    filed_at: Mapped[date] = mapped_column(Date)
    url: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    company: Mapped[Company] = relationship(back_populates="filings")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="filing")


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (UniqueConstraint("filing_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filing.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(32))  # filled by section parsing in M1
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    filing: Mapped[Filing] = relationship(back_populates="chunks")


class EmbeddingCache(Base):
    """Embeddings keyed by (sha256 of text, model), so the same text is never embedded twice."""

    __tablename__ = "embedding_cache"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector())  # no fixed dim: any model fits
