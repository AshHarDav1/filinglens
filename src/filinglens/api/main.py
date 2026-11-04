"""FilingLens HTTP API around the M3 agent.

    uv run uvicorn filinglens.api.main:app --reload       # Swagger UI: http://localhost:8000/docs

Lifetimes: model clients are created once at start-up (about 1.25 s); every request gets its own
database session (and, from /ask on, its own agent, about 7 ms to build).
"""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from filinglens.api.schemas import HealthResponse, ModelsInfo
from filinglens.config import Settings, get_settings
from filinglens.db.models import Company, Filing
from filinglens.db.session import get_session

VERSION = version("filinglens")


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
    return app


app = create_app()
