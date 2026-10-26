import json
from datetime import date

import httpx
import pytest

from filinglens.ingestion.edgar_client import (
    EdgarClient,
    EdgarError,
    fiscal_year_of,
    select_10k,
)

UA = "Test User test@example.com"


def make_client(tmp_path, handler, **kwargs) -> tuple[EdgarClient, list[float]]:
    sleeps: list[float] = []
    client = EdgarClient(
        UA,
        tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        clock=lambda: 0.0,
        **kwargs,
    )
    return client, sleeps


def test_requires_user_agent(tmp_path):
    with pytest.raises(EdgarError, match="SEC_USER_AGENT"):
        EdgarClient("", tmp_path)


def test_sends_user_agent_and_caches_on_disk(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b"hello")

    client, _ = make_client(tmp_path, handler)
    url = "https://www.sec.gov/Archives/edgar/data/1/abc/doc.htm"
    assert client.get_bytes(url) == b"hello"
    assert client.get_bytes(url) == b"hello"

    assert len(calls) == 1  # second call served from disk
    assert calls[0].headers["User-Agent"] == UA
    assert (tmp_path / "www.sec.gov/Archives/edgar/data/1/abc/doc.htm").read_bytes() == b"hello"


def test_retries_with_backoff_then_succeeds(tmp_path):
    responses = iter([httpx.Response(503), httpx.Response(429), httpx.Response(200, content=b"ok")])
    client, sleeps = make_client(tmp_path, lambda r: next(responses))

    assert client.get_bytes("https://data.sec.gov/x.json") == b"ok"
    backoffs = [s for s in sleeps if s >= 1]
    assert backoffs == [1, 2]


def test_gives_up_after_max_retries(tmp_path):
    client, _ = make_client(tmp_path, lambda r: httpx.Response(503), max_retries=3)
    with pytest.raises(EdgarError, match="after 3 attempts"):
        client.get_bytes("https://data.sec.gov/x.json")


def test_does_not_retry_client_errors(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    client, _ = make_client(tmp_path, handler)
    with pytest.raises(EdgarError, match="404"):
        client.get_bytes("https://data.sec.gov/missing.json")
    assert len(calls) == 1


def test_rate_limit_spaces_requests(tmp_path):
    # Frozen clock: each request after the first must wait one interval.
    client, sleeps = make_client(
        tmp_path, lambda r: httpx.Response(200, content=b"x"), max_requests_per_second=4
    )
    for i in range(3):
        client.get_bytes(f"https://data.sec.gov/{i}.json")
    assert sleeps == [0.25, 0.5]


@pytest.mark.parametrize(
    ("period_end", "expected"),
    [
        (date(2022, 12, 31), 2022),
        (date(2023, 6, 30), 2023),  # Microsoft: FY ends in June
        (date(2022, 12, 2), 2022),  # Adobe: FY ends early December
        (date(2022, 1, 1), 2021),  # 52/53-week year ending in the first days of January
    ],
)
def test_fiscal_year_of(period_end, expected):
    assert fiscal_year_of(period_end) == expected


def submissions_page(rows: list[dict]) -> dict:
    keys = ["form", "accessionNumber", "reportDate", "filingDate", "primaryDocument"]
    return {k: [r[k] for r in rows] for k in keys}


def row(form, accession, report_date, filing_date="2023-02-01", doc="doc.htm"):
    return {
        "form": form,
        "accessionNumber": accession,
        "reportDate": report_date,
        "filingDate": filing_date,
        "primaryDocument": doc,
    }


def test_select_10k_skips_amendments_and_other_forms():
    page = submissions_page(
        [
            row("10-K/A", "a-1", "2022-12-31"),
            row("10-Q", "q-1", "2022-09-30"),
            row("4", "f-1", ""),
            row("10-K", "k-2022", "2022-12-31"),
            row("10-K", "k-2021", "2021-12-31"),
        ]
    )
    assert select_10k([page], 2022)["accessionNumber"] == "k-2022"
    assert select_10k([page], 2021)["accessionNumber"] == "k-2021"
    assert select_10k([page], 2015) is None


def test_find_10k_follows_older_submission_pages(tmp_path):
    tickers = {"0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"}}
    recent = submissions_page([row("10-K", "0000950170-23-035122", "2023-06-30", "2023-07-27")])
    older = submissions_page([row("10-K", "0001193125-16-662209", "2016-06-30", "2016-07-28")])
    main = {
        "filings": {"recent": recent, "files": [{"name": "CIK0000789019-submissions-001.json"}]}
    }
    routes = {
        "/files/company_tickers.json": tickers,
        "/submissions/CIK0000789019.json": main,
        "/submissions/CIK0000789019-submissions-001.json": older,
    }

    def handler(request):
        return httpx.Response(200, content=json.dumps(routes[request.url.path]).encode())

    client, _ = make_client(tmp_path, handler)
    ref = client.find_10k("msft", 2016)

    assert ref.ticker == "MSFT"
    assert ref.accession_no == "0001193125-16-662209"
    assert ref.period_end == date(2016, 6, 30)
    assert ref.url == ("https://www.sec.gov/Archives/edgar/data/789019/000119312516662209/doc.htm")
