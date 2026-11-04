"""Vector search within one filing. Hybrid search and section routing build on it (hybrid.py)."""

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from filinglens.db.models import Chunk, Company, Filing


class RetrievedChunk(BaseModel):
    chunk_id: int
    chunk_index: int
    ticker: str
    fiscal_year: int
    form: str
    section: str | None  # 10-K Item, e.g. "8" (None for chunks ingested before M1)
    url: str
    distance: float
    content: str


def filing_chunks(query_vector: list[float], ticker: str, fiscal_year: int, sections=None):
    """SELECT of one filing's chunks with their cosine distance to `query_vector`."""
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    query = (
        select(Chunk, Filing, Company, distance)
        .join(Filing, Chunk.filing_id == Filing.id)
        .join(Company, Filing.company_id == Company.id)
        .where(Company.ticker == ticker.upper(), Filing.fiscal_year == fiscal_year)
    )
    if sections:
        query = query.where(Chunk.section.in_(sections))
    return query, distance


def to_results(rows) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            ticker=company.ticker,
            fiscal_year=filing.fiscal_year,
            form=filing.form,
            section=chunk.section,
            url=filing.url,
            distance=float(dist),
            content=chunk.content,
        )
        for chunk, filing, company, dist, *_ in rows
    ]


def search(
    session: Session,
    query_vector: list[float],
    ticker: str,
    fiscal_year: int,
    k: int = 6,
    sections: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks by cosine distance, restricted to one company's filing for one fiscal year
    (and to `sections`, the 10-K Items, if given).

    The filing filter matters: similarity search always returns *something*, so without it a
    question about one company could be answered from another company's text.
    """
    query, distance = filing_chunks(query_vector, ticker, fiscal_year, sections)
    return to_results(session.execute(query.order_by(distance).limit(k)).all())
