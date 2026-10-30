"""Database models."""

from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
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
    __table_args__ = (
        UniqueConstraint("filing_id", "chunk_index"),
        Index("ix_chunk_content_tsv", "content_tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filing.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(32))  # filled by section parsing in M1
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Stemmed words of `content` for full-text (keyword) search; Postgres fills it on every write
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    filing: Mapped[Filing] = relationship(back_populates="chunks")


class FinancialFact(Base):
    """One XBRL fact exactly as a 10-K reported it (SEC companyfacts API).

    Facts are stored as reported, duplicates included: each 10-K repeats the prior years as
    comparison columns, so one period can have several rows. Picking the one right number is a
    query-time decision (`financials.get_metric`), not a load-time one.
    """

    __tablename__ = "financial_fact"
    __table_args__ = (
        # NULLS NOT DISTINCT: instant facts have no period_start and must still dedupe on re-runs
        UniqueConstraint(
            "company_id",
            "concept",
            "unit",
            "period_start",
            "period_end",
            "accession_no",
            postgresql_nulls_not_distinct=True,
        ),
        Index("ix_financial_fact_lookup", "company_id", "concept", "fiscal_year"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("company.id"))
    concept: Mapped[str] = mapped_column(String(255))  # e.g. "CostOfRevenue" (us-gaap)
    unit: Mapped[str] = mapped_column(String(32))  # "USD", "shares", "USD/shares", "pure"
    value: Mapped[Decimal] = mapped_column(Numeric)  # exact decimal, never float
    # None for instant facts (balance sheet: a value *at* period_end)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    # Fiscal year of the period (from period_end, same rule as filing.fiscal_year), not SEC's
    # `fy`, which is the year of the filing that reported the fact. Meaningful for annual and
    # year-end facts, which is all get_metric asks for.
    fiscal_year: Mapped[int] = mapped_column(Integer)
    form: Mapped[str] = mapped_column(String(16))  # "10-K" or "10-K/A"
    accession_no: Mapped[str] = mapped_column(String(32))  # the filing that reported it
    filed_at: Mapped[date] = mapped_column(Date)


class EmbeddingCache(Base):
    """Embeddings keyed by (sha256 of text, model), so the same text is never embedded twice."""

    __tablename__ = "embedding_cache"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector())  # no fixed dim: any model fits
