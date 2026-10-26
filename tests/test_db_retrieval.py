"""Integration tests against the dev Postgres (skipped if it is not running)."""

from datetime import date

from langchain_core.embeddings import Embeddings

from filinglens.db.models import EMBEDDING_DIM, Chunk, Company, Filing
from filinglens.ingestion.chunk_embed import embed_with_cache
from filinglens.retrieval.vector import search


def one_hot(i: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[i] = 1.0
    return vector


class CountingEmbedder(Embeddings):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [one_hot(len(t) % EMBEDDING_DIM) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return one_hot(len(text) % EMBEDDING_DIM)


def test_embedding_cache_never_embeds_twice(db_session):
    embedder = CountingEmbedder()
    vectors, new = embed_with_cache(db_session, embedder, "test-model", ["a", "bb", "a"])
    assert new == 2
    assert vectors[0] == vectors[2]

    _, new_again = embed_with_cache(db_session, embedder, "test-model", ["bb", "ccc"])
    assert new_again == 1
    assert embedder.calls == [["a", "bb"], ["ccc"]]


def add_filing(session, cik, ticker, year, vectors) -> None:
    company = Company(cik=cik, ticker=ticker, name=ticker)
    session.add(company)
    session.flush()
    filing = Filing(
        company_id=company.id,
        accession_no=f"test-{ticker}-{year}",
        form="10-K",
        fiscal_year=year,
        period_end=date(year, 12, 31),
        filed_at=date(year + 1, 2, 1),
        url=f"https://example.com/{ticker}",
    )
    session.add(filing)
    session.flush()
    session.add_all(
        Chunk(filing_id=filing.id, chunk_index=i, content=f"{ticker} chunk {i}", embedding=v)
        for i, v in enumerate(vectors)
    )
    session.flush()


def test_search_ranks_by_similarity_within_company_and_year(db_session):
    add_filing(db_session, 9_000_001, "TSTA", 2022, [one_hot(0), one_hot(1), one_hot(2)])
    add_filing(db_session, 9_000_002, "TSTB", 2022, [one_hot(1)])  # perfect match, other company

    results = search(db_session, one_hot(1), "tsta", 2022, k=2)

    assert [r.content for r in results] == ["TSTA chunk 1", "TSTA chunk 0"]
    assert results[0].distance == 0.0
    assert search(db_session, one_hot(1), "TSTA", 2015) == []
