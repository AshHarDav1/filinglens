"""Hybrid retrieval: vector search + Postgres full-text search, merged with reciprocal rank fusion,
optionally restricted to (or boosted towards) the 10-K Items a router picked.

All searches stay inside one filing (ticker + fiscal year).
"""

import re
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from filinglens.db.models import EMBEDDING_DIM, Chunk
from filinglens.retrieval.vector import RetrievedChunk, filing_chunks, search, to_results

Method = Literal["vector", "keyword", "hybrid"]
SectionMode = Literal["filter", "boost"]

RRF_K = 60  # the standard constant: one first place can't outweigh everything else
CANDIDATES = 100  # how deep each list goes before fusion
# Fewer Item 8 chunks than this means Item 8 only points elsewhere ("see Item 15")
MIN_ITEM8_CHUNKS = 5

# Words that tell the answerer how to answer, not what to look for. FinanceBench questions are
# full of them ("Please state answer in USD millions"), and in a keyword search they match nearly
# every table. Postgres drops ordinary stopwords ("the", "of") by itself.
INSTRUCTION_WORDS = {
    "answer", "answers", "please", "provide", "response", "respond", "question", "state",
    "calculate", "compute", "round", "rounded", "decimal", "decimals", "place", "places",
    "units", "unit", "usd", "million", "millions", "billion", "billions", "percent",
    "percentage", "percents", "using", "use", "based", "basing", "primarily", "information",
    "provided", "shown", "help", "us", "want", "what", "which", "how", "does", "did", "has",
    "have", "is", "are", "was", "were", "the", "a", "an", "of", "in", "for", "to", "by", "and",
    "or", "on", "as", "at", "from", "that", "this", "it", "its", "their", "be", "if", "then",
    "not", "metric", "metrics", "judgments", "judgment", "details", "detail",
}  # fmt: skip


def keyword_query(question: str) -> str | None:
    """The question's content words OR-ed into a to_tsquery string, or None if none are left."""
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 1]
    content = [w for w in dict.fromkeys(words) if w not in INSTRUCTION_WORDS]
    return " | ".join(content) or None


def keyword_search(
    session: Session,
    question: str,
    query_vector: list[float],
    ticker: str,
    fiscal_year: int,
    k: int = 6,
    sections: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks by full-text rank (ts_rank_cd over the content_tsv GIN index)."""
    terms = keyword_query(question)
    if terms is None:
        return []
    tsquery = func.to_tsquery("english", terms)
    rank = func.ts_rank_cd(Chunk.content_tsv, tsquery).label("rank")
    query, _ = filing_chunks(query_vector, ticker, fiscal_year, sections)
    query = query.add_columns(rank).where(Chunk.content_tsv.op("@@")(tsquery))
    return to_results(session.execute(query.order_by(rank.desc()).limit(k)).all())


def rrf(rankings: list[list[RetrievedChunk]], k: int | None = None) -> list[RetrievedChunk]:
    """Reciprocal rank fusion: each list gives a chunk 1 / (RRF_K + rank); sort by the sum."""
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (RRF_K + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    merged = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunks[cid] for cid in merged[:k]]


def statement_sections(
    session: Session, ticker: str, fiscal_year: int, sections: list[str]
) -> list[str]:
    """Where "Item 8" really is in this filing.

    Some filings make Item 8 a one-line pointer and put the statements after Item 15 (Netflix);
    if this filing's Item 8 is that small, Items 15 and 16 stand in for it.
    """
    if "8" not in sections:
        return sections
    query, _ = filing_chunks([0.0] * EMBEDDING_DIM, ticker, fiscal_year, ["8"])
    item8_chunks = session.scalar(select(func.count()).select_from(query.subquery()))
    if item8_chunks >= MIN_ITEM8_CHUNKS:
        return sections
    return sections + [s for s in ("15", "16") if s not in sections]


def _ranked(
    session: Session,
    method: Method,
    question: str,
    query_vector: list[float],
    ticker: str,
    fiscal_year: int,
    depth: int,
    sections: list[str] | None,
) -> list[RetrievedChunk]:
    if method == "vector":
        return search(session, query_vector, ticker, fiscal_year, depth, sections)
    if method == "keyword":
        return keyword_search(session, question, query_vector, ticker, fiscal_year, depth, sections)
    return rrf(
        [
            search(session, query_vector, ticker, fiscal_year, depth, sections),
            keyword_search(session, question, query_vector, ticker, fiscal_year, depth, sections),
        ]
    )


def retrieve(
    session: Session,
    question: str,
    query_vector: list[float],
    ticker: str,
    fiscal_year: int,
    k: int = 6,
    method: Method = "vector",
    sections: list[str] | None = None,
    section_mode: SectionMode = "filter",
) -> list[RetrievedChunk]:
    """The top-k chunks for a question within one filing.

    sections: Items a router picked, or None to search the whole filing.
    section_mode "filter": search only those Items, topped up from the whole filing if they hold
      fewer than k chunks (Netflix's Item 8 is a one-line pointer to Item 15).
    section_mode "boost": fuse the whole-filing ranking with the in-section ranking, so a wrong
      guess costs ranks instead of hiding the answer.
    """
    depth = max(k, CANDIDATES)
    whole = _ranked(session, method, question, query_vector, ticker, fiscal_year, depth, None)
    if not sections:
        return whole[:k]
    sections = statement_sections(session, ticker, fiscal_year, sections)
    inside = _ranked(session, method, question, query_vector, ticker, fiscal_year, depth, sections)
    if section_mode == "boost":
        return rrf([whole, inside], k)
    seen = {c.chunk_id for c in inside}
    return (inside + [c for c in whole if c.chunk_id not in seen])[:k]
