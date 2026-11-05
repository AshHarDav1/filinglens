"""The HTTP API: no real models and no network. Dependencies are swapped with
app.dependency_overrides; TestClient is used without `with`, so the lifespan (which creates real
model clients) does not run."""

import json

import anthropic
import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy.exc import OperationalError

from filinglens.agent.graph import build_agent
from filinglens.agent.tools import make_tools
from filinglens.api.main import create_app, get_agent, get_db
from tests.fakes import FakeGuard, FakeLLM, call


class BrokenSession:
    def scalar(self, *args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


def client_with(overrides: dict) -> TestClient:
    app = create_app()
    app.dependency_overrides.update(overrides)
    return TestClient(app)


def test_health_reports_the_database(db_session):
    response = client_with({get_db: lambda: db_session}).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["companies"] >= 1
    assert set(body["models"]) == {"answer", "guard", "embeddings"}


def test_health_is_503_when_the_database_is_down():
    response = client_with({get_db: BrokenSession}).get("/health")
    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


def test_openapi_describes_the_api():
    spec = TestClient(create_app()).get("/openapi.json").json()
    assert spec["info"]["title"] == "FilingLens API"
    assert "/health" in spec["paths"]


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_docs_pages_are_served(path):
    assert TestClient(create_app()).get(path).status_code == 200


# --- POST /ask ----------------------------------------------------------------------------------

FILINGS = {"AMD": [2022]}
QUICK_RATIO = [
    AIMessage("", tool_calls=[call("calculate", {"expression": "(4835 + 1020 + 4126) / 6369"})]),
    AIMessage("Done."),
    AIMessage("AMD's FY2022 quick ratio is 1.57 [1]."),
]


def agent_of(llm, verdict: str = "ok"):
    """Overrides get_agent: a fresh agent per request with scripted models and no database."""
    return lambda: build_agent(llm(), FakeGuard(verdict), make_tools(None, None), FILINGS)


def test_ask_returns_the_answer_with_sources_and_trace():
    client = client_with({get_agent: agent_of(lambda: FakeLLM(QUICK_RATIO))})
    response = client.post("/ask", json={"question": "What is AMD's FY2022 quick ratio?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "AMD's FY2022 quick ratio is 1.57 [1]."
    assert body["outcome"] == "answered"
    assert body["verified"] is True
    assert body["sources"] == [{"n": 1, "kind": "calculation", "label": "calculate", "url": None}]
    assert body["tool_calls"] == [
        {"name": "calculate", "args": {"expression": "(4835 + 1020 + 4126) / 6369"}}
    ]
    assert body["rounds"] == 1
    assert len(body["request_id"]) == 32
    assert body["trace_id"] is None


def test_advice_is_a_200_with_outcome_refused():
    client = client_with({get_agent: agent_of(lambda: FakeLLM([]), verdict="advice")})
    body = client.post("/ask", json={"question": "Should I buy AMD?"}).json()
    assert body["outcome"] == "refused"
    assert body["sources"] == []


@pytest.mark.parametrize("question", ["", "  a ", "x" * 2001])
def test_bad_questions_are_422(question):
    client = client_with({get_agent: agent_of(lambda: FakeLLM([]))})
    assert client.post("/ask", json={"question": question}).status_code == 422


class FailingLLM(FakeLLM):
    def __init__(self, error: Exception):
        super().__init__([])
        self.error = error

    def invoke(self, messages):
        raise self.error


REQUEST = httpx.Request("POST", "https://api.example.invalid/v1")


@pytest.mark.parametrize(
    "error", [openai.APIConnectionError(request=REQUEST), anthropic.RateLimitError(
        "overloaded", response=httpx.Response(429, request=REQUEST), body=None)]
)  # fmt: skip
def test_model_provider_errors_are_502(error):
    client = client_with({get_agent: agent_of(lambda: FailingLLM(error))})
    response = client.post("/ask", json={"question": "What is AMD's FY2022 quick ratio?"})
    assert response.status_code == 502
    assert response.json()["detail"].startswith(f"Model provider error ({type(error).__name__})")


def test_database_errors_are_503():
    def broken_agent():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    response = client_with({get_agent: broken_agent}).post("/ask", json={"question": "Hi there"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable."}


# --- POST /ask/stream ---------------------------------------------------------------------------


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_sends_each_step_then_the_result():
    client = client_with({get_agent: agent_of(lambda: FakeLLM(QUICK_RATIO))})
    response = client.post("/ask/stream", json={"question": "What is AMD's FY2022 quick ratio?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    steps = [data for kind, data in events if kind == "step"]
    assert [s["node"] for s in steps] == ["guard", "plan", "tools", "plan", "answer", "check"]
    assert steps[0]["label"] == "Question is in scope"
    assert steps[1]["label"] == "calculate (4835 + 1020 + 4126) / 6369"
    assert steps[2]["results"] == [
        {"name": "calculate", "status": "success", "summary": "= 1.56712"}
    ]
    assert steps[3]["label"] == "Done gathering data"
    assert steps[-1]["verified"] is True
    kind, result = events[-1]
    assert kind == "result"
    assert result["answer"] == "AMD's FY2022 quick ratio is 1.57 [1]."
    assert result["sources"][0]["n"] == 1


def test_stream_refusal():
    client = client_with({get_agent: agent_of(lambda: FakeLLM([]), verdict="advice")})
    events = parse_sse(client.post("/ask/stream", json={"question": "Buy AMD?"}).text)
    assert [(k, d.get("node")) for k, d in events] == [
        ("step", "guard"), ("step", "refuse"), ("result", None)
    ]  # fmt: skip
    assert events[-1][1]["outcome"] == "refused"


def test_stream_error_after_start_is_an_error_event():
    error = openai.APIConnectionError(request=REQUEST)
    client = client_with({get_agent: agent_of(lambda: FailingLLM(error))})
    response = client.post("/ask/stream", json={"question": "What is AMD's FY2022 quick ratio?"})
    assert response.status_code == 200  # already sent when the planner failed
    events = parse_sse(response.text)
    assert [k for k, _ in events] == ["step", "error"]  # the guard finished, then the failure
    assert events[-1][1]["status"] == 502


def test_stream_is_documented_in_openapi():
    spec = TestClient(create_app()).get("/openapi.json").json()
    content = spec["paths"]["/ask/stream"]["post"]["responses"]["200"]["content"]
    assert "text/event-stream" in content
