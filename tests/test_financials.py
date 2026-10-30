"""get_metric on small, real-shaped fact sets (values from Microsoft's filings)."""

from datetime import date
from decimal import Decimal

import pytest

from filinglens.db.models import Company, FinancialFact
from filinglens.financials import METRICS, get_metric

FY16_10K = "0001193125-16-662209"  # Microsoft's FY2016 10-K, filed 2016-07-28
FY17_10K = "0001564590-17-014900"
FY18_10K = "0001564590-18-019062"
FILED = {FY16_10K: date(2016, 7, 28), FY17_10K: date(2017, 8, 2), FY18_10K: date(2018, 8, 3)}


def fact(company_id, concept, value, accession, start, end, fiscal_year):
    return FinancialFact(
        company_id=company_id,
        concept=concept,
        unit="USD",
        value=Decimal(value),
        period_start=start,
        period_end=end,
        fiscal_year=fiscal_year,
        form="10-K",
        accession_no=accession,
        filed_at=FILED[accession],
    )


@pytest.fixture
def msft(db_session):
    company = Company(cik=999_000_002, ticker="TMSFT", name="Test Microsoft")
    db_session.add(company)
    db_session.flush()
    fy16 = (date(2015, 7, 1), date(2016, 6, 30), 2016)
    fy15 = (date(2014, 7, 1), date(2015, 6, 30), 2015)
    fy17 = (date(2016, 7, 1), date(2017, 6, 30), 2017)
    fy18 = (date(2017, 7, 1), date(2018, 6, 30), 2018)
    db_session.add_all(
        [
            # cost of revenue: the same FY2016 number in three 10-Ks, under two names
            fact(company.id, "CostOfRevenue", "32780000000", FY16_10K, *fy16),
            fact(company.id, "CostOfRevenue", "33038000000", FY16_10K, *fy15),
            fact(company.id, "CostOfGoodsSold", "17880000000", FY16_10K, *fy16),  # product only
            fact(company.id, "CostOfRevenue", "32780000000", FY17_10K, *fy16),
            fact(company.id, "CostOfRevenue", "34261000000", FY17_10K, *fy17),
            fact(company.id, "CostOfGoodsAndServicesSold", "32780000000", FY18_10K, *fy16),
            fact(company.id, "CostOfGoodsAndServicesSold", "38353000000", FY18_10K, *fy18),
            # revenue: restated in the FY2018 10-K under a new concept
            fact(company.id, "SalesRevenueNet", "85320000000", FY16_10K, *fy16),
            fact(
                company.id,
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "91154000000",
                FY18_10K,
                *fy16,
            ),
            fact(
                company.id,
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "110360000000",
                FY18_10K,
                *fy18,
            ),
            # balance sheet (instant): at the year end and at the prior year end
            fact(
                company.id,
                "LiabilitiesCurrent",
                "59357000000",
                FY16_10K,
                None,
                date(2016, 6, 30),
                2016,
            ),
            fact(
                company.id,
                "LiabilitiesCurrent",
                "49647000000",
                FY16_10K,
                None,
                date(2015, 6, 30),
                2015,
            ),
            # capex only appears in a later 10-K's comparison column
            fact(
                company.id,
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "8343000000",
                FY17_10K,
                *fy16,
            ),
        ]
    )
    db_session.flush()
    return company


def test_cost_of_revenue_is_the_as_reported_total(db_session, msft):
    result = get_metric(db_session, "TMSFT", 2016, "cost_of_revenue")
    assert result.value == Decimal("32780000000")
    assert result.concept == "CostOfRevenue"  # not CostOfGoodsSold (17,880: product only)
    assert result.accession_no == FY16_10K
    assert result.as_reported
    assert result.source_url.endswith("/999000002/000119312516662209/")


def test_own_10k_wins_over_a_later_restatement(db_session, msft):
    result = get_metric(db_session, "TMSFT", 2016, "revenue")
    assert result.value == Decimal("85320000000")  # not the restated 91,154
    assert result.concept == "SalesRevenueNet"


def test_fiscal_year_is_the_period_not_the_filing(db_session, msft):
    # The FY2016 10-K also reports FY2015 (33,038) as a comparison column
    assert get_metric(db_session, "TMSFT", 2015, "cost_of_revenue").value == Decimal("33038000000")
    # FY2017 has no own 10-K here, so the comparison column in a later 10-K is used
    fy17 = get_metric(db_session, "TMSFT", 2017, "cost_of_revenue")
    assert fy17.value == Decimal("34261000000")


def test_instant_fact_is_taken_at_the_year_end(db_session, msft):
    result = get_metric(db_session, "TMSFT", 2016, "current_liabilities")
    assert result.value == Decimal("59357000000")
    assert result.period_start is None


def test_falls_back_to_a_later_filing_and_says_so(db_session, msft):
    result = get_metric(db_session, "TMSFT", 2016, "capex")
    assert result.value == Decimal("8343000000")
    assert result.accession_no == FY17_10K
    # FY17_10K's latest year is 2017, so it isn't FY2016's own 10-K
    assert not result.as_reported


def test_missing_metric_returns_none_and_unknown_metric_raises(db_session, msft):
    assert get_metric(db_session, "TMSFT", 2016, "inventory") is None
    assert get_metric(db_session, "NOPE", 2016, "revenue") is None
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric(db_session, "TMSFT", 2016, "ebitda")


def test_every_metric_has_unique_concepts():
    for metric, concepts in METRICS.items():
        assert concepts, metric
        assert len(concepts) == len(set(concepts)), metric
