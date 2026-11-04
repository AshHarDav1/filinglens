"""The agent's three tools: search_filings (text), get_financials (XBRL numbers), calculate (math).

Each tool is a plain function first (easy to test, no LLM needed), then wrapped for LangChain by
make_tools(). The wrappers use response_format="content_and_artifact": the LLM sees `content`, a
compact text rendering; our own code (citations, the grounding check) reads `artifact`, the typed
Pydantic result.

Errors are raised as ToolException with a message written for the LLM (what went wrong, what is
valid), so the agent can fix its call on the next round instead of giving up.
"""

import ast
import operator
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from filinglens.db.models import Company, Filing
from filinglens.financials import METRICS, get_metric
from filinglens.ingestion.parse_10k import ITEM_ORDER
from filinglens.retrieval.hybrid import retrieve

MetricName = Literal[tuple(METRICS)]
SectionId = Literal[("cover", *ITEM_ORDER)]

SEARCH_K = 6  # passages per search, as in M2
MAX_EXPRESSION_CHARS = 200
MAX_ABS_RESULT = 1e15


# --- what exists in the database ----------------------------------------------------------------


def available_filings(session: Session) -> dict[str, list[int]]:
    """Ticker -> fiscal years of the 10-Ks we ingested (text). XBRL facts cover more years."""
    rows = session.execute(
        select(Company.ticker, Filing.fiscal_year)
        .join(Filing, Filing.company_id == Company.id)
        .where(Filing.form == "10-K")
        .order_by(Company.ticker, Filing.fiscal_year)
    ).all()
    filings: dict[str, list[int]] = {}
    for ticker, year in rows:
        filings.setdefault(ticker, []).append(year)
    return filings


def _check_ticker(filings: dict[str, list[int]], ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker not in filings:
        raise ToolException(
            f"Unknown ticker {ticker!r}. FilingLens covers only: {', '.join(sorted(filings))}. "
            "If the question is about another company, its filings are not available."
        )
    return ticker


# --- search_filings -----------------------------------------------------------------------------


class SearchFilingsInput(BaseModel):
    query: str = Field(description="What to look for, in plain words (one topic per search).")
    ticker: str = Field(description="Company ticker, e.g. MSFT.")
    fiscal_year: int = Field(description="Fiscal year of the 10-K, e.g. 2022.")
    sections: list[SectionId] | None = Field(
        default=None,
        description=(
            "Optional 10-K Items to search: 1 Business, 1A Risk Factors, 3 Legal Proceedings, "
            "7 MD&A (drivers, segments, non-GAAP), 7A Market risk, 8 Financial statements and "
            "notes. Omit to search the whole filing."
        ),
    )


class Passage(BaseModel):
    chunk_id: int
    ticker: str
    fiscal_year: int
    form: str
    section: str | None
    url: str
    content: str


class SearchFilingsOutput(BaseModel):
    query: str
    passages: list[Passage]


def search_filings(
    session: Session,
    embedder: Embeddings,
    query: str,
    ticker: str,
    fiscal_year: int,
    sections: list[str] | None = None,
) -> SearchFilingsOutput:
    filings = available_filings(session)
    ticker = _check_ticker(filings, ticker)
    if fiscal_year not in filings[ticker]:
        years = ", ".join(map(str, filings[ticker]))
        raise ToolException(
            f"No {ticker} 10-K for fiscal year {fiscal_year} is ingested. Available: {years}. "
            "For numbers from other years, use get_financials."
        )
    chunks = retrieve(
        session, query, embedder.embed_query(query), ticker, fiscal_year, SEARCH_K,
        sections=sections or None,
    )  # fmt: skip
    return SearchFilingsOutput(
        query=query,
        passages=[
            Passage(
                chunk_id=c.chunk_id,
                ticker=c.ticker,
                fiscal_year=c.fiscal_year,
                form=c.form,
                section=c.section,
                url=c.url,
                content=c.content,
            )
            for c in chunks
        ],
    )


def render_passages(result: SearchFilingsOutput) -> str:
    if not result.passages:
        return f"No passages found for {result.query!r}."
    return "\n\n".join(
        f"[chunk {p.chunk_id}] {p.ticker} {p.form} FY{p.fiscal_year}"
        f"{f' · Item {p.section}' if p.section else ''}\n{p.content}"
        for p in result.passages
    )


# --- get_financials -----------------------------------------------------------------------------


class GetFinancialsInput(BaseModel):
    ticker: str = Field(description="Company ticker, e.g. MSFT.")
    fiscal_years: list[int] = Field(
        min_length=1, max_length=6, description="Fiscal years, e.g. [2021, 2022]."
    )
    metrics: list[MetricName] = Field(
        min_length=1,
        max_length=10,
        description=(
            "Metrics to fetch. Income statement and cash flow values are for the whole fiscal "
            "year; balance sheet values are at the fiscal year end."
        ),
    )


class FinancialValue(BaseModel):
    ticker: str
    fiscal_year: int
    metric: str
    value_musd: float  # USD millions
    concept: str  # the XBRL concept it came from
    period_end: date
    as_reported: bool  # False: taken from a later 10-K (may be restated)
    source_url: str


class GetFinancialsOutput(BaseModel):
    values: list[FinancialValue]
    missing: list[str]  # "inventory FY2019": not reported under any known concept


def get_financials(
    session: Session, ticker: str, fiscal_years: list[int], metrics: list[str]
) -> GetFinancialsOutput:
    ticker = _check_ticker(available_filings(session), ticker)
    values, missing = [], []
    for year in fiscal_years:
        for metric in metrics:
            fact = get_metric(session, ticker, year, metric)
            if fact is None:
                missing.append(f"{metric} FY{year}")
                continue
            values.append(
                FinancialValue(
                    ticker=ticker,
                    fiscal_year=year,
                    metric=metric,
                    value_musd=float(fact.value / Decimal(1_000_000)),
                    concept=fact.concept,
                    period_end=fact.period_end,
                    as_reported=fact.as_reported,
                    source_url=fact.source_url,
                )
            )
    return GetFinancialsOutput(values=values, missing=missing)


def format_musd(value: float) -> str:
    """4835.0 -> '4,835'; 305.826 -> '305.826'."""
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def render_financials(result: GetFinancialsOutput) -> str:
    lines = ["USD millions, from XBRL in the company's 10-K filings:"]
    for v in result.values:
        note = "" if v.as_reported else ", from a later 10-K (may be restated)"
        lines.append(
            f"{v.ticker} FY{v.fiscal_year} {v.metric} = {format_musd(v.value_musd)} "
            f"({v.concept}, period end {v.period_end}{note})"
        )
    if result.missing:
        lines.append("Not reported: " + "; ".join(result.missing))
    return "\n".join(lines)


# --- calculate ----------------------------------------------------------------------------------


class CalculateInput(BaseModel):
    expression: str = Field(
        description=(
            "Arithmetic on numbers only: + - * / and parentheses, e.g. "
            "(4835 + 1020 + 4126) / 6369. Use the values exactly as the tools returned them."
        )
    )


class CalculateOutput(BaseModel):
    expression: str
    value: float


# A whitelist, not a blacklist: anything not listed here is rejected. No ** (2**10**10 hangs the
# process), no names, no calls, no attributes, so nothing can reach Python itself.
_BINARY: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY: dict[type, Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ToolException(
        f"Not allowed in calculate: {ast.unparse(node)!r}. Use only numbers, + - * / and "
        "parentheses (no variables, functions, percent signs or **)."
    )


def calculate(expression: str) -> CalculateOutput:
    cleaned = expression.replace(",", "").strip()  # "4,835" -> "4835"
    if len(cleaned) > MAX_EXPRESSION_CHARS:
        raise ToolException(f"Expression too long (max {MAX_EXPRESSION_CHARS} characters).")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise ToolException(f"Not a valid expression: {expression!r}.") from e
    try:
        value = _evaluate(tree.body)
    except ZeroDivisionError as e:
        raise ToolException("Division by zero.") from e
    if abs(value) > MAX_ABS_RESULT:
        raise ToolException("Result too large; check the units of the inputs.")
    return CalculateOutput(expression=cleaned, value=value)


def render_calculation(result: CalculateOutput) -> str:
    return f"{result.expression} = {result.value:.6g}"


# --- LangChain wrappers -------------------------------------------------------------------------


def make_tools(session: Session, embedder: Embeddings) -> list[BaseTool]:
    """The three tools bound to a DB session and an embedder. Their descriptions are what the LLM
    reads to decide which tool to call, so they are part of the prompt."""

    def _search(query: str, ticker: str, fiscal_year: int, sections: list[str] | None = None):
        result = search_filings(session, embedder, query, ticker, fiscal_year, sections)
        return render_passages(result), result

    def _financials(ticker: str, fiscal_years: list[int], metrics: list[str]):
        result = get_financials(session, ticker, fiscal_years, metrics)
        return render_financials(result), result

    def _calculate(expression: str):
        result = calculate(expression)
        return render_calculation(result), result

    common = {"response_format": "content_and_artifact", "handle_tool_error": True}
    return [
        StructuredTool.from_function(
            _search,
            name="search_filings",
            description=(
                "Search the text of one company's 10-K for one fiscal year. Use it for "
                "explanations, drivers of changes, risks, legal matters, segments and other "
                "context. Do not take reported financial numbers from it when get_financials "
                "has them. Returns up to 6 passages."
            ),
            args_schema=SearchFilingsInput,
            **common,
        ),
        StructuredTool.from_function(
            _financials,
            name="get_financials",
            description=(
                "Exact reported numbers from the company's XBRL financial data, in USD millions. "
                "Use it for every financial statement number (revenue, income, cash flow, "
                "balance sheet items) instead of reading numbers from text. Works for any "
                "fiscal year the company reported."
            ),
            args_schema=GetFinancialsInput,
            **common,
        ),
        StructuredTool.from_function(
            _calculate,
            name="calculate",
            description=(
                "Evaluate an arithmetic expression exactly. Use it for every ratio, margin, "
                "difference or growth rate instead of doing the math yourself."
            ),
            args_schema=CalculateInput,
            **common,
        ),
    ]
