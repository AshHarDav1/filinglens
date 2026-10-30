"""Ingest 10-Ks: EDGAR -> raw cache -> text -> Item sections -> chunks -> embeddings -> Postgres.

    uv run python -m filinglens.ingestion.run --from-subset eval/financebench_subset.jsonl
    uv run python -m filinglens.ingestion.run --filings MSFT:2023,AMD:2022
    uv run python -m filinglens.ingestion.run --from-subset ... --download-only

Idempotent: a filing that already has chunks is skipped (use --rechunk to rebuild it).
"""

import argparse
import json
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from filinglens.config import get_settings
from filinglens.db.models import Chunk, Company, Filing
from filinglens.db.session import get_session
from filinglens.ingestion.chunk_embed import build_chunks, embed_with_cache
from filinglens.ingestion.edgar_client import EdgarClient, FilingRef
from filinglens.ingestion.html_text import html_to_text
from filinglens.ingestion.parse_10k import split_sections
from filinglens.llm import get_embeddings

# Some companies put Item 7/8 in an exhibit (EX-13) instead of the main document; a short main
# document is the tell. Handling exhibits is M1 work, so for now we only warn.
SHORT_DOCUMENT_CHARS = 150_000


def parse_targets(args: argparse.Namespace) -> list[tuple[str, int]]:
    targets: set[tuple[str, int]] = set()
    if args.from_subset:
        for line in args.from_subset.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            targets.add((row["ticker"], int(row["fiscal_year"])))
    if args.filings:
        for item in args.filings.split(","):
            ticker, year = item.strip().split(":")
            targets.add((ticker.upper(), int(year)))
    return sorted(targets)


def upsert_company(session: Session, cik: int, ticker: str, name: str) -> Company:
    company = session.scalar(select(Company).where(Company.cik == cik))
    if company is None:
        company = Company(cik=cik, ticker=ticker, name=name)
        session.add(company)
        session.flush()
    return company


def upsert_filing(session: Session, ref: FilingRef) -> Filing:
    company = upsert_company(session, ref.cik, ref.ticker, ref.company_name)
    filing = session.scalar(select(Filing).where(Filing.accession_no == ref.accession_no))
    if filing is None:
        filing = Filing(
            company_id=company.id,
            accession_no=ref.accession_no,
            form=ref.form,
            fiscal_year=ref.fiscal_year,
            period_end=ref.period_end,
            filed_at=ref.filed_at,
            url=ref.url,
        )
        session.add(filing)
        session.flush()
    return filing


def ingest_one(
    session: Session, client: EdgarClient, ticker: str, year: int, args: argparse.Namespace
) -> None:
    ref = client.find_10k(ticker, year)
    path = client.download_filing(ref)
    text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    label = f"{ticker} FY{year} ({ref.accession_no}, {len(text):,} chars)"
    if len(text) < SHORT_DOCUMENT_CHARS:
        print(f"  WARNING {label}: short document, financials may be in an exhibit")
    if args.download_only:
        print(f"  downloaded {label} -> {path}")
        return

    filing = upsert_filing(session, ref)
    existing = session.scalar(select(func.count()).where(Chunk.filing_id == filing.id))
    if existing and not args.rechunk:
        print(f"  skip {label}: already has {existing} chunks")
        return
    session.execute(delete(Chunk).where(Chunk.filing_id == filing.id))

    sections = split_sections(text)
    items = [s.item for s in sections]
    missing = [item for item in ("1A", "7", "8") if item not in items]
    if missing:
        print(f"  WARNING {label}: no Item {', '.join(missing)} found")
    drafts = build_chunks(sections)

    settings = get_settings()
    vectors, new = embed_with_cache(
        session, get_embeddings(settings), settings.embedding_model, [d.content for d in drafts]
    )
    session.add_all(
        Chunk(
            filing_id=filing.id,
            chunk_index=i,
            section=draft.section,
            content=draft.content,
            embedding=vector,
        )
        for i, (draft, vector) in enumerate(zip(drafts, vectors, strict=True))
    )
    session.commit()
    print(
        f"  ingested {label}: {len(sections)} sections, {len(drafts)} chunks ({new} newly embedded)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest 10-K filings from SEC EDGAR")
    parser.add_argument("--from-subset", type=Path, help="eval subset JSONL (ticker, fiscal_year)")
    parser.add_argument("--filings", help="comma-separated TICKER:FISCAL_YEAR, e.g. MSFT:2023")
    parser.add_argument("--download-only", action="store_true", help="only fetch raw files")
    parser.add_argument("--rechunk", action="store_true", help="rebuild chunks of known filings")
    args = parser.parse_args()

    targets = parse_targets(args)
    if not targets:
        parser.error("give --from-subset and/or --filings")

    print(f"Ingesting {len(targets)} filings")
    with EdgarClient.from_settings(get_settings()) as client, get_session() as session:
        for ticker, year in targets:
            ingest_one(session, client, ticker, year, args)


if __name__ == "__main__":
    main()
