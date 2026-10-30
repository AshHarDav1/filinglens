"""Vector-only retrieval for the M0 proof of concept. Hybrid search (vector + full-text) is M2."""

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


def search(
    session: Session, query_vector: list[float], ticker: str, fiscal_year: int, k: int = 6
) -> list[RetrievedChunk]:
    """Top-k chunks by cosine distance, restricted to one company's filing for one fiscal year.

    The filter matters: similarity search always returns *something*, so without it a question
    about one company could be answered from another company's text.
    """
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    rows = session.execute(
        select(Chunk, Filing, Company, distance)
        .join(Filing, Chunk.filing_id == Filing.id)
        .join(Company, Filing.company_id == Company.id)
        .where(Company.ticker == ticker.upper(), Filing.fiscal_year == fiscal_year)
        .order_by(distance)
        .limit(k)
    ).all()
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
        for chunk, filing, company, dist in rows
    ]
