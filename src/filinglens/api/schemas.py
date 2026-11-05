"""Request and response models of the HTTP API. They are the contract: FastAPI validates requests
against them and generates the OpenAPI document (/docs) from them."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

EXAMPLE_QUESTION = (
    "What is the FY2022 unadjusted EBITDA % margin for PepsiCo? Calculate unadjusted EBITDA "
    "using unadjusted operating income and D&A (from cash flow statement)."
)


class AskRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"question": EXAMPLE_QUESTION}]})

    # The length cap is also a cost cap: the question goes into every LLM call of the agent.
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)
    ]


class SourceOut(BaseModel):
    n: int = Field(description="The [n] the answer cites")
    kind: Literal["xbrl", "calculation", "passage"]
    label: str = Field(examples=["XBRL · PEP FY2022 revenue", "BA 10-K FY2022 · Item 8"])
    url: str | None = Field(description="SEC filing URL; null for calculations")


class ToolCallOut(BaseModel):
    name: Literal["search_filings", "get_financials", "calculate"]
    args: dict[str, Any]


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    outcome: Literal["answered", "not_found", "refused"]
    verified: bool = Field(description="Every number and citation traced to a source (M3 check)")
    sources: list[SourceOut] = Field(description="Only the sources the answer cites")
    tool_calls: list[ToolCallOut] = Field(description="The tool calls that ran, in order")
    rounds: int = Field(description="Rounds of tool calls (at most 4)")
    attempts: int = Field(description="Answer attempts: 2 means the check asked for a rewrite")
    latency_ms: int
    trace_id: str | None = Field(default=None, description="Langfuse trace id (from M6)")


class ModelsInfo(BaseModel):
    answer: str
    guard: str
    embeddings: str


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    database: Literal["ok", "unreachable"]
    companies: int | None
    filings: int | None = Field(description="10-K filings with text")
    models: ModelsInfo
    version: str


class ErrorResponse(BaseModel):
    detail: str
