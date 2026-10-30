"""SEC EDGAR client: one shared, rate-limited HTTP client with retries and an on-disk cache.

SEC rules: every request sends a User-Agent with a name and contact email, and stays under
10 requests/second. Raw downloads are cached under `data_dir/raw/<host>/<path>` so re-running
ingestion never hits SEC twice for the same file.
"""

import json
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from filinglens.config import Settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

RETRY_STATUSES = {429, 500, 502, 503, 504}


class EdgarError(RuntimeError):
    pass


class FilingRef(BaseModel):
    cik: int
    ticker: str
    company_name: str
    form: str
    accession_no: str
    fiscal_year: int
    period_end: date
    filed_at: date
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVES_URL.format(
            cik=self.cik,
            accession=self.accession_no.replace("-", ""),
            document=self.primary_document,
        )


def fiscal_year_of(period_end: date) -> int:
    """Fiscal year a 10-K covers, from its period end date.

    52/53-week fiscal years can end in the first days of January (e.g. 2022-01-01 closes FY2021).
    """
    if period_end.month == 1 and period_end.day <= 7:
        return period_end.year - 1
    return period_end.year


def iter_filings(page: dict) -> Iterator[dict]:
    """Turn one columnar submissions page ({"form": [...], "reportDate": [...]}) into rows."""
    columns = list(page.keys())
    for values in zip(*(page[c] for c in columns), strict=True):
        yield dict(zip(columns, values, strict=True))


def select_10k(pages: Iterable[dict], fiscal_year: int, form: str = "10-K") -> dict | None:
    """Pick the original (not amended) annual report for `fiscal_year`."""
    for page in pages:
        for row in iter_filings(page):
            if row["form"] != form or not row.get("reportDate"):
                continue
            if fiscal_year_of(date.fromisoformat(row["reportDate"])) == fiscal_year:
                return row
    return None


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        max_requests_per_second: float = 8.0,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_agent:
            raise EdgarError("SEC_USER_AGENT is not set (see .env.example)")
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._cache_dir = cache_dir
        self._interval = 1.0 / max_requests_per_second
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "EdgarClient":
        return cls(settings.sec_user_agent or "", settings.data_dir / "raw")

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # --- low level -------------------------------------------------------------------------

    def cache_path(self, url: str) -> Path:
        parsed = urlparse(url)
        return self._cache_dir / parsed.netloc / parsed.path.lstrip("/")

    def get_bytes(self, url: str, use_cache: bool = True) -> bytes:
        path = self.cache_path(url)
        if use_cache and path.exists():
            return path.read_bytes()
        content = self._fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return content

    def get_json(self, url: str, use_cache: bool = True) -> dict:
        return json.loads(self.get_bytes(url, use_cache))

    def _wait_for_turn(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next_request_at - now
            if wait > 0:
                self._sleep(wait)
            self._next_request_at = max(now, self._next_request_at) + self._interval

    def _fetch(self, url: str) -> bytes:
        for attempt in range(self._max_retries):
            self._wait_for_turn()
            try:
                response = self._http.get(url)
            except httpx.TransportError as exc:
                error: str = repr(exc)
            else:
                if response.status_code == 200:
                    return response.content
                if response.status_code not in RETRY_STATUSES:
                    raise EdgarError(f"GET {url} -> {response.status_code}")
                error = f"HTTP {response.status_code}"
            if attempt < self._max_retries - 1:
                self._sleep(2**attempt)
        raise EdgarError(f"GET {url} failed after {self._max_retries} attempts: {error}")

    # --- EDGAR lookups ---------------------------------------------------------------------

    def lookup_ticker(self, ticker: str) -> tuple[int, str]:
        """Ticker -> (CIK, company name)."""
        for entry in self.get_json(TICKERS_URL).values():
            if entry["ticker"].upper() == ticker.upper():
                return int(entry["cik_str"]), entry["title"]
        raise EdgarError(f"Unknown ticker: {ticker}")

    def _submission_pages(self, cik: int) -> Iterator[dict]:
        data = self.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        yield data["filings"]["recent"]
        # Older filings live in extra paginated files, fetched only if needed.
        for extra in data["filings"].get("files", []):
            yield self.get_json(SUBMISSIONS_URL.format(name=extra["name"]))

    def find_10k(self, ticker: str, fiscal_year: int) -> FilingRef:
        cik, name = self.lookup_ticker(ticker)
        row = select_10k(self._submission_pages(cik), fiscal_year)
        if row is None:
            raise EdgarError(f"No 10-K found for {ticker} FY{fiscal_year}")
        return FilingRef(
            cik=cik,
            ticker=ticker.upper(),
            company_name=name,
            form=row["form"],
            accession_no=row["accessionNumber"],
            fiscal_year=fiscal_year,
            period_end=date.fromisoformat(row["reportDate"]),
            filed_at=date.fromisoformat(row["filingDate"]),
            primary_document=row["primaryDocument"],
        )

    def get_companyfacts(self, cik: int) -> dict:
        """All XBRL facts a company ever reported (one JSON, several MB, cached on disk)."""
        return self.get_json(COMPANYFACTS_URL.format(cik=cik))

    def download_filing(self, ref: FilingRef) -> Path:
        """Download the filing's primary document (cached) and return its local path."""
        self.get_bytes(ref.url)
        return self.cache_path(ref.url)
