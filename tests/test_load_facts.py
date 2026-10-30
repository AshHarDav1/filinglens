from datetime import date
from decimal import Decimal

from sqlalchemy import func, select

from filinglens.db.models import Company, FinancialFact
from filinglens.ingestion.load_facts import insert_facts, parse_facts

# Shaped like a real companyfacts JSON (values from Microsoft's)
COMPANYFACTS = {
    "cik": 789019,
    "entityName": "MICROSOFT CORPORATION",
    "facts": {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "end": "2016-07-14",
                            "val": 7808000000,
                            "accn": "a",
                            "fy": 2016,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2016-07-28",
                        }
                    ]
                }
            }
        },
        "us-gaap": {
            "CostOfRevenue": {
                "label": "Cost of Revenue",
                "units": {
                    "USD": [
                        # FY2016, reported in the FY2016 10-K
                        {
                            "start": "2015-07-01",
                            "end": "2016-06-30",
                            "val": 32780000000,
                            "accn": "0001193125-16-662209",
                            "fy": 2016,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2016-07-28",
                        },
                        # the same period repeated in the FY2017 10-K: SEC's fy says 2017
                        {
                            "start": "2015-07-01",
                            "end": "2016-06-30",
                            "val": 32780000000,
                            "accn": "0001564590-17-014900",
                            "fy": 2017,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2017-08-02",
                            "frame": "CY2016",
                        },
                        # a quarter from a 10-Q: not loaded
                        {
                            "start": "2015-07-01",
                            "end": "2015-09-30",
                            "val": 7207000000,
                            "accn": "0001193125-15-350718",
                            "fy": 2016,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2015-10-22",
                        },
                    ]
                },
            },
            "LiabilitiesCurrent": {
                "units": {
                    "USD": [
                        # instant fact: no start date
                        {
                            "end": "2016-06-30",
                            "val": 59357000000,
                            "accn": "0001193125-16-662209",
                            "fy": 2016,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2016-07-28",
                        }
                    ]
                }
            },
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        {
                            "start": "2015-07-01",
                            "end": "2016-06-30",
                            "val": 2.1,
                            "accn": "0001193125-16-662209",
                            "fy": 2016,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2016-07-28",
                        }
                    ]
                }
            },
        },
    },
}


def test_parse_keeps_only_us_gaap_10k_facts():
    rows = parse_facts(COMPANYFACTS)
    assert {r.concept for r in rows} == {
        "CostOfRevenue",
        "LiabilitiesCurrent",
        "EarningsPerShareDiluted",
    }
    assert all(r.form == "10-K" for r in rows)
    assert len(rows) == 4  # both 10-K copies of FY2016 cost of revenue are kept


def test_fiscal_year_comes_from_the_period_not_from_secs_fy():
    rows = [r for r in parse_facts(COMPANYFACTS) if r.concept == "CostOfRevenue"]
    assert [r.fiscal_year for r in rows] == [2016, 2016]
    assert {r.accession_no for r in rows} == {"0001193125-16-662209", "0001564590-17-014900"}


def test_instant_facts_have_no_start_and_values_stay_exact():
    rows = {r.concept: r for r in parse_facts(COMPANYFACTS)}
    assert rows["LiabilitiesCurrent"].period_start is None
    assert rows["LiabilitiesCurrent"].period_end == date(2016, 6, 30)
    assert rows["EarningsPerShareDiluted"].value == Decimal("2.1")
    assert rows["EarningsPerShareDiluted"].unit == "USD/shares"


def test_insert_is_idempotent_including_instant_facts(db_session):
    company = Company(cik=999_000_001, ticker="TEST", name="Test Co")
    db_session.add(company)
    db_session.flush()
    rows = parse_facts(COMPANYFACTS)

    assert insert_facts(db_session, company.id, rows) == 4
    assert insert_facts(db_session, company.id, rows) == 0  # NULL period_start still dedupes

    count = db_session.scalar(select(func.count()).where(FinancialFact.company_id == company.id))
    assert count == 4
