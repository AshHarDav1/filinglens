"""Load XBRL facts (SEC companyfacts API) into financial_fact.

    uv run python -m filinglens.ingestion.load_facts --from-subset eval/financebench_subset.jsonl
    uv run python -m filinglens.ingestion.load_facts --tickers MSFT,AMD

Loads every us-gaap fact reported in a 10-K or 10-K/A, for all years, as reported (duplicates
across filings included; see FinancialFact). Idempotent: re-running inserts nothing twice.
"""

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from filinglens.config import get_settings
from filinglens.db.models import FinancialFact
from filinglens.db.session import get_session
from filinglens.ingestion.edgar_client import EdgarClient, fiscal_year_of
from filinglens.ingestion.run import upsert_company

TAXONOMY = "us-gaap"  # dei (cover-page data) and company-specific extensions are skipped
ANNUAL_FORMS = {"10-K", "10-K/A"}
INSERT_BATCH = 5000


class FactRow(BaseModel):
    concept: str
    unit: str
    value: Decimal
    period_start: date | None
    period_end: date
    fiscal_year: int
    form: str
    accession_no: str
    filed_at: date


def parse_facts(companyfacts: dict) -> list[FactRow]:
    """Flatten a companyfacts JSON into one row per 10-K fact."""
    rows: list[FactRow] = []
    for concept, body in companyfacts.get("facts", {}).get(TAXONOMY, {}).items():
        for unit, facts in body.get("units", {}).items():
            for fact in facts:
                if fact.get("form") not in ANNUAL_FORMS:
                    continue
                period_end = date.fromisoformat(fact["end"])
                rows.append(
                    FactRow(
                        concept=concept,
                        unit=unit,
                        value=Decimal(str(fact["val"])),
                        period_start=date.fromisoformat(fact["start"]) if "start" in fact else None,
                        period_end=period_end,
                        fiscal_year=fiscal_year_of(period_end),
                        form=fact["form"],
                        accession_no=fact["accn"],
                        filed_at=date.fromisoformat(fact["filed"]),
                    )
                )
    return rows


def insert_facts(session: Session, company_id: int, rows: list[FactRow]) -> int:
    """Insert rows, skipping ones already stored. Returns the number of new rows."""
    inserted = 0
    for i in range(0, len(rows), INSERT_BATCH):
        values = [{"company_id": company_id, **r.model_dump()} for r in rows[i : i + INSERT_BATCH]]
        result = session.execute(
            insert(FinancialFact)
            .values(values)
            .on_conflict_do_nothing()
            .returning(FinancialFact.id)
        )
        inserted += len(result.all())
    return inserted


def load_company(session: Session, client: EdgarClient, ticker: str) -> None:
    cik, name = client.lookup_ticker(ticker)
    rows = parse_facts(client.get_companyfacts(cik))
    company = upsert_company(session, cik, ticker.upper(), name)
    new = insert_facts(session, company.id, rows)
    session.commit()
    years = sorted({r.fiscal_year for r in rows})
    span = f"FY{years[0]}-FY{years[-1]}" if years else "no years"
    print(f"  {ticker:<5} {len(rows):>6,} 10-K facts ({span}), {new:,} new")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load XBRL facts from SEC companyfacts")
    parser.add_argument("--from-subset", type=Path, help="eval subset JSONL (uses its tickers)")
    parser.add_argument("--tickers", help="comma-separated tickers, e.g. MSFT,AMD")
    args = parser.parse_args()

    tickers: set[str] = set()
    if args.from_subset:
        for line in args.from_subset.read_text(encoding="utf-8").splitlines():
            tickers.add(json.loads(line)["ticker"].upper())
    if args.tickers:
        tickers.update(t.strip().upper() for t in args.tickers.split(","))
    if not tickers:
        parser.error("give --from-subset and/or --tickers")

    print(f"Loading XBRL facts for {len(tickers)} companies")
    with EdgarClient.from_settings(get_settings()) as client, get_session() as session:
        for ticker in sorted(tickers):
            load_company(session, client, ticker)


if __name__ == "__main__":
    main()
