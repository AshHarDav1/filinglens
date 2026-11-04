"""FilingLens HTTP API around the M3 agent.

    uv run uvicorn filinglens.api.main:app --reload       # Swagger UI: http://localhost:8000/docs

Lifetimes: model clients are created once at start-up (about 1.25 s); every request gets its own
database session and its own agent (about 7 ms to build, because the tools hold the session).
"""

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated

import anthropic
import openai
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from filinglens.agent.graph import AgentResult, ask, build_agent
from filinglens.agent.tools import available_filings, make_tools
from filinglens.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    ModelsInfo,
    SourceOut,
    ToolCallOut,
)
from filinglens.config import Settings, get_settings
from filinglens.db.models import Company, Filing
from filinglens.db.session import get_session

VERSION = version("filinglens")
log = logging.getLogger(__name__)

# Errors from the model providers' SDKs (outages, rate limits, timeouts, bad keys)
PROVIDER_ERRORS = (openai.APIError, anthropic.APIError)
ERROR_RESPONSES = {
    502: {"model": ErrorResponse, "description": "OpenAI or Anthropic failed"},
    503: {"model": ErrorResponse, "description": "Database unavailable"},
}


@dataclass(frozen=True)
class Models:
    """Shared by all requests: LangChain chat and embedding clients are thread-safe."""

    llm: BaseChatModel
    guard: BaseChatModel
    embedder: Embeddings


def load_models(settings: Settings) -> Models:
    from filinglens.llm import get_chat_model, get_embeddings, get_router_model

    return Models(get_chat_model(settings), get_router_model(settings), get_embeddings(settings))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models(get_settings())
    yield


def get_db() -> Iterator[Session]:
    """One session per request, closed when the response is done."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def get_agent(request: Request, session: DbSession) -> CompiledStateGraph:
    """A fresh agent per request: its tools hold this request's session (about 7 ms)."""
    models: Models = request.app.state.models
    tools = make_tools(session, models.embedder)
    return build_agent(models.llm, models.guard, tools, available_filings(session))


Agent = Annotated[CompiledStateGraph, Depends(get_agent)]


def to_response(result: AgentResult, request_id: str, started: float) -> AskResponse:
    return AskResponse(
        request_id=request_id,
        question=result.question,
        answer=result.answer,
        outcome=result.outcome,
        verified=result.verified,
        sources=[SourceOut(n=s.n, kind=s.kind, label=s.label, url=s.url) for s in result.sources],
        tool_calls=[ToolCallOut(**c) for c in result.tool_calls],
        rounds=result.rounds,
        attempts=result.attempts,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


router = APIRouter()


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(session: DbSession, settings: AppSettings):
    """Database reachable, and what is loaded. 503 when the database is down."""
    models = ModelsInfo(
        answer=settings.llm_model, guard=settings.router_model, embeddings=settings.embedding_model
    )
    try:
        companies = session.scalar(select(func.count()).select_from(Company))
        filings = session.scalar(
            select(func.count()).select_from(Filing).where(Filing.form == "10-K")
        )
    except SQLAlchemyError:
        down = HealthResponse(
            status="error", database="unreachable", companies=None, filings=None,
            models=models, version=VERSION,
        )  # fmt: skip
        return JSONResponse(status_code=503, content=down.model_dump())
    return HealthResponse(
        status="ok", database="ok", companies=companies, filings=filings,
        models=models, version=VERSION,
    )  # fmt: skip


@router.post("/ask", response_model=AskResponse, responses=ERROR_RESPONSES)
def ask_question(body: AskRequest, agent: Agent) -> AskResponse:
    """Run the agent and return the answer with its sources, in one response.

    A plain `def`: FastAPI runs it in a worker thread, so the synchronous agent (seconds of
    waiting on LLM calls) doesn't block other requests. Refusals and "not found" are 200s with
    their `outcome`.
    """
    request_id, started = uuid.uuid4().hex, time.perf_counter()
    response = to_response(ask(agent, body.question), request_id, started)
    log.info("ask %s %s %d ms", request_id, response.outcome, response.latency_ms)
    return response


def provider_error(request: Request, exc: Exception) -> JSONResponse:
    log.warning("model provider error: %r", exc)
    detail = f"Model provider error ({type(exc).__name__}). Try again in a moment."
    return JSONResponse(status_code=502, content={"detail": detail})


def database_error(request: Request, exc: Exception) -> JSONResponse:
    log.warning("database error: %r", exc)
    return JSONResponse(status_code=503, content={"detail": "Database unavailable."})


def create_app() -> FastAPI:
    app = FastAPI(
        title="FilingLens API",
        version=VERSION,
        description=(
            "Answers questions about US public companies' 10-K filings with exact XBRL numbers "
            "and cited filing text. Not investment advice."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    for error in PROVIDER_ERRORS:
        app.add_exception_handler(error, provider_error)
    app.add_exception_handler(SQLAlchemyError, database_error)
    return app


app = create_app()
