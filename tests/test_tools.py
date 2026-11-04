"""The agent's tools: calculate safety, get_financials units and errors, search_filings errors,
and the LangChain wrappers (schemas the LLM sees, content vs artifact, errors as messages)."""

from datetime import date
from decimal import Decimal

import pytest
from langchain_core.tools import ToolException

from filinglens.agent import tools
from filinglens.agent.tools import (
    calculate,
    format_musd,
    get_financials,
    make_tools,
    render_financials,
    search_filings,
)
from filinglens.db.models import Company, Filing, FinancialFact
from filinglens.retrieval.vector import RetrievedChunk

# --- calculate ----------------------------------------------------------------------------------


def test_calculate_amd_quick_ratio():
    # AMD FY2022: (cash + short-term investments + receivables) / current liabilities
    result = calculate("(4835 + 1020 + 4126) / 6369")
    assert result.value == pytest.approx(1.5671, abs=1e-4)


def test_calculate_accepts_thousands_separators_and_negatives():
    assert calculate("47,237 - 49,781").value == -2544
    assert calculate("-(3 - 5) * 2.5").value == 5


@pytest.mark.parametrize(
    "expression",
    [
        "2 ** 10",  # power: 9**9**9 would hang the process
        "__import__('os').system('ls')",  # calls and names
        "x + 1",  # names
        "(1).real",  # attributes
        "10 % 3",  # modulo (and "35%" is not a percentage here)
        "[1, 2]",
        "'a' * 3",  # strings
        "True + 1",  # bools are ints in Python; not numbers here
    ],
)
def test_calculate_rejects_anything_but_arithmetic(expression):
    with pytest.raises(ToolException, match=r"Not allowed|Not a valid"):
        calculate(expression)


def test_calculate_errors_are_explained():
    with pytest.raises(ToolException, match="Division by zero"):
        calculate("1 / (2 - 2)")
    with pytest.raises(ToolException, match="too long"):
        calculate("1 + " * 100 + "1")
    with pytest.raises(ToolException, match="Not a valid"):
        calculate("(1 + 2")
    with pytest.raises(ToolException, match="too large"):
        calculate("1e300 * 1e300")


# --- get_financials -----------------------------------------------------------------------------

ACCESSION = "0009999999-22-000001"


@pytest.fixture
def amd(db_session):
    """A test company with one 10-K and AMD's FY2022 balance sheet values."""
    company = Company(cik=999_000_003, ticker="TAMD", name="Test AMD")
    db_session.add(company)
    db_session.flush()
    db_session.add(
        Filing(
            company_id=company.id,
            accession_no=ACCESSION,
            form="10-K",
            fiscal_year=2022,
            period_end=date(2022, 12, 31),
            filed_at=date(2023, 2, 27),
            url="https://www.sec.gov/test",
        )
    )
    for concept, value in [
        ("CashAndCashEquivalentsAtCarryingValue", "4835000000"),
        ("LiabilitiesCurrent", "6369000000"),
        ("DepreciationDepletionAndAmortization", "305826000"),
    ]:
        db_session.add(
            FinancialFact(
                company_id=company.id,
                concept=concept,
                unit="USD",
                value=Decimal(value),
                period_start=None if "Depreciation" not in concept else date(2022, 1, 1),
                period_end=date(2022, 12, 31),
                fiscal_year=2022,
                form="10-K",
                accession_no=ACCESSION,
                filed_at=date(2023, 2, 27),
            )
        )
    db_session.flush()
    return company


def test_get_financials_in_usd_millions(db_session, amd):
    result = get_financials(
        db_session, "tamd", [2022], ["cash", "current_liabilities", "depreciation_amortization"]
    )
    by_metric = {v.metric: v for v in result.values}
    assert by_metric["cash"].value_musd == 4835
    assert by_metric["current_liabilities"].value_musd == 6369
    assert by_metric["depreciation_amortization"].value_musd == pytest.approx(305.826)
    assert by_metric["cash"].as_reported
    assert result.missing == []


def test_get_financials_reports_missing_values(db_session, amd):
    result = get_financials(db_session, "TAMD", [2021, 2022], ["cash", "inventory"])
    assert [v.fiscal_year for v in result.values] == [2022]
    assert result.missing == ["cash FY2021", "inventory FY2021", "inventory FY2022"]
    assert "Not reported: cash FY2021" in render_financials(result)


def test_get_financials_unknown_ticker_lists_the_companies(db_session, amd):
    with pytest.raises(ToolException, match=r"Unknown ticker 'TSLA'.*TAMD"):
        get_financials(db_session, "TSLA", [2022], ["revenue"])


def test_format_musd():
    assert format_musd(4835.0) == "4,835"
    assert format_musd(305.826) == "305.826"
    assert format_musd(-2544.0) == "-2,544"


# --- search_filings (retrieval stubbed: no database or embeddings needed) -----------------------


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


@pytest.fixture
def stub_retrieval(monkeypatch):
    calls = []

    def fake_retrieve(session, question, vector, ticker, fiscal_year, k, sections=None):
        calls.append({"ticker": ticker, "fiscal_year": fiscal_year, "sections": sections})
        return [
            RetrievedChunk(
                chunk_id=219,
                chunk_index=219,
                ticker=ticker,
                fiscal_year=fiscal_year,
                form="10-K",
                section="8",
                url="https://www.sec.gov/ba-10k",
                distance=0.3,
                content="Multiple legal actions have been filed against us ...",
            )
        ]

    monkeypatch.setattr(tools, "available_filings", lambda session: {"BA": [2022]})
    monkeypatch.setattr(tools, "retrieve", fake_retrieve)
    return calls


def test_search_filings_returns_passages(stub_retrieval):
    result = search_filings(None, FakeEmbedder(), "737 MAX lawsuits", "ba", 2022, ["8"])
    assert result.passages[0].chunk_id == 219
    assert stub_retrieval == [{"ticker": "BA", "fiscal_year": 2022, "sections": ["8"]}]


def test_search_filings_wrong_year_says_what_exists(stub_retrieval):
    with pytest.raises(ToolException, match=r"No BA 10-K for fiscal year 2019.*Available: 2022"):
        search_filings(None, FakeEmbedder(), "lawsuits", "BA", 2019)


# --- LangChain wrappers -------------------------------------------------------------------------


def tool_call(name: str, args: dict) -> dict:
    return {"type": "tool_call", "id": "call_1", "name": name, "args": args}


def test_tool_schemas_offer_only_valid_metrics_and_sections():
    by_name = {t.name: t for t in make_tools(None, FakeEmbedder())}
    assert set(by_name) == {"search_filings", "get_financials", "calculate"}
    schema = by_name["get_financials"].tool_call_schema.model_json_schema()
    assert "operating_income" in schema["properties"]["metrics"]["items"]["enum"]
    search_schema = by_name["search_filings"].tool_call_schema.model_json_schema()
    assert "1A" in str(search_schema["properties"]["sections"])


def test_tool_message_has_text_for_the_llm_and_an_artifact_for_our_code():
    calc = {t.name: t for t in make_tools(None, FakeEmbedder())}["calculate"]
    message = calc.invoke(tool_call("calculate", {"expression": "(4835 + 1020 + 4126) / 6369"}))
    assert message.content == "(4835 + 1020 + 4126) / 6369 = 1.56712"
    assert message.artifact.value == pytest.approx(1.56712, abs=1e-5)


def test_tool_errors_come_back_as_messages_not_exceptions():
    calc = {t.name: t for t in make_tools(None, FakeEmbedder())}["calculate"]
    message = calc.invoke(tool_call("calculate", {"expression": "2 ** 10"}))
    assert message.status == "error"
    assert "Not allowed" in message.content
