"""Exact numbers from XBRL facts: metric name -> one value for a company and fiscal year.

This is the "numbers come from XBRL, never from text" rule (ADR-002). The agent's get_financials
tool (M3) is a thin wrapper around get_metric.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from filinglens.db.models import Company, FinancialFact

# Metric -> XBRL concepts, in priority order. Companies tag the same idea differently, and the
# same company renames concepts over time. Order matters when a 10-K reports several of them
# with different values: e.g. Microsoft's FY2016 10-K has CostOfRevenue = 32,780 (total) and
# CostOfGoodsSold = 17,880 (product only). Checked against the 12 filings in the eval subset;
# a metric a company doesn't report (AmEx, a bank, has no cost of revenue) returns None.
METRICS: dict[str, list[str]] = {
    # income statement (duration facts)
    "revenue": [
        "Revenues",
        "RevenuesNetOfInterestExpense",  # banks (AmEx): the total, not just fee revenue
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",  # pre-2018 filings
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss"],
    # cash flow statement (duration facts). Plain "Depreciation" is left out on purpose: it is
    # often property depreciation only, which would make an EBITDA silently wrong.
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    # balance sheet (instant facts, at fiscal year end)
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "accounts_payable": ["AccountsPayableCurrent"],
    "debt_current": [
        "DebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtCurrent",
    ],
    "long_term_debt_noncurrent": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "long_term_debt_total": ["LongTermDebt"],  # current + noncurrent portions
}

ANNUAL_DAYS = (350, 380)  # 52/53-week years included
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"


class Fact(BaseModel):
    ticker: str
    metric: str
    concept: str
    fiscal_year: int
    value: Decimal
    unit: str
    period_start: date | None
    period_end: date
    accession_no: str
    filed_at: date
    # True: from that fiscal year's own 10-K. False: only found as a comparison column in a later
    # 10-K, which may be restated (Microsoft's FY2016 revenue: 85,320 as reported, 91,154 later).
    as_reported: bool
    source_url: str


def _is_annual_or_instant(fact: FinancialFact) -> bool:
    if fact.period_start is None:
        return True
    return ANNUAL_DAYS[0] <= (fact.period_end - fact.period_start).days <= ANNUAL_DAYS[1]


def get_metric(session: Session, ticker: str, fiscal_year: int, metric: str) -> Fact | None:
    """The one value of `metric` for `ticker` in `fiscal_year`, or None if it isn't reported.

    Rules, in order:
    1. Only annual (~365-day) or instant facts whose period falls in `fiscal_year`. Filtering on
       the period, not on SEC's `fy` (the year of the filing that reported the fact).
    2. Prefer the fiscal year's own 10-K: the number as originally reported.
    3. Then the metric's concept priority.
    4. For instant facts, the latest date in the year (the fiscal year end).
    5. Ties: the original filing over amendments; outside the own 10-K, the most recent filing.
    """
    concepts = METRICS.get(metric)
    if concepts is None:
        raise ValueError(f"Unknown metric {metric!r}; known: {', '.join(sorted(METRICS))}")

    rows = session.execute(
        select(FinancialFact, Company.cik)
        .join(Company, FinancialFact.company_id == Company.id)
        .where(
            Company.ticker == ticker.upper(),
            FinancialFact.concept.in_(concepts),
            FinancialFact.fiscal_year == fiscal_year,
            FinancialFact.unit == "USD",
        )
    ).all()
    candidates = [(fact, cik) for fact, cik in rows if _is_annual_or_instant(fact)]
    if not candidates:
        return None

    # A 10-K reports its own year plus comparison years, so the filing whose latest fiscal year
    # is `fiscal_year` is that year's own 10-K.
    accessions = {fact.accession_no for fact, _ in candidates}
    own = set(
        session.scalars(
            select(FinancialFact.accession_no)
            .where(
                FinancialFact.company_id == candidates[0][0].company_id,
                FinancialFact.accession_no.in_(accessions),
            )
            .group_by(FinancialFact.accession_no)
            .having(func.max(FinancialFact.fiscal_year) == fiscal_year)
        )
    )

    def rank(item: tuple[FinancialFact, int]) -> tuple:
        fact = item[0]
        is_own = fact.accession_no in own
        filed = fact.filed_at.toordinal()
        return (
            not is_own,
            concepts.index(fact.concept),
            -fact.period_end.toordinal(),
            filed if is_own else -filed,
        )

    fact, cik = min(candidates, key=rank)
    return Fact(
        ticker=ticker.upper(),
        metric=metric,
        concept=fact.concept,
        fiscal_year=fact.fiscal_year,
        value=fact.value,
        unit=fact.unit,
        period_start=fact.period_start,
        period_end=fact.period_end,
        accession_no=fact.accession_no,
        filed_at=fact.filed_at,
        as_reported=fact.accession_no in own,
        source_url=FILING_INDEX_URL.format(cik=cik, accession=fact.accession_no.replace("-", "")),
    )
