"""The Python client: SSE parsing, error handling (httpx.MockTransport), and one end-to-end
stream through the real API app with a scripted agent (FastAPI's TestClient is an httpx.Client)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from filinglens.api.client import FilingLensClient, FilingLensError, parse_sse
from filinglens.api.main import create_app, get_agent
from tests.fakes import FakeLLM
from tests.test_api import QUICK_RATIO, agent_of

HEALTH = {
    "status": "ok", "database": "ok", "companies": 10, "filings": 21, "version": "0.1.0",
    "models": {"answer": "gpt-6-luna", "guard": "claude-haiku-5-5", "embeddings": "e"},
}  # fmt: skip


def mock_client(handler) -> FilingLensClient:
    http = httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))
    return FilingLensClient(http=http)


def test_parse_sse_handles_comments_multiline_data_and_a_missing_final_blank_line():
    lines = [
        ": keep-alive",
        "event: step",
        'data: {"node": "guard",',
        'data:  "label": "Question is in scope"}',
        "",
        "event: result",
        'data: {"answer": "1.57 [1]"}',
    ]
    events = list(parse_sse(lines))
    assert [e.type for e in events] == ["step", "result"]
    assert events[0].data == {"node": "guard", "label": "Question is in scope"}
    assert events[1].data == {"answer": "1.57 [1]"}


def test_health_ok_and_down():
    client = mock_client(lambda request: httpx.Response(200, json=HEALTH))
    assert client.health().companies == 10
    down = {**HEALTH, "status": "error", "database": "unreachable", "companies": None}
    client = mock_client(lambda request: httpx.Response(503, json=down))
    assert client.health().database == "unreachable"


def test_errors_raise_with_the_api_detail():
    def handler(request):
        return httpx.Response(502, json={"detail": "Model provider error (APIConnectionError)."})

    with pytest.raises(FilingLensError) as error:
        mock_client(handler).ask("What is AMD's FY2022 quick ratio?")
    assert error.value.status == 502
    assert "APIConnectionError" in error.value.detail
    with pytest.raises(FilingLensError, match="502"):
        list(mock_client(handler).ask_stream("What is AMD's FY2022 quick ratio?"))


def test_stream_through_the_mock_transport():
    body = 'event: step\ndata: {"node": "guard"}\n\nevent: result\ndata: {"node": "x"}\n\n'

    def handler(request):
        assert json.loads(request.content) == {"question": "Q?"}
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    assert [e.type for e in mock_client(handler).ask_stream("Q?")] == ["step", "result"]


def test_end_to_end_stream_through_the_real_app():
    app = create_app()
    app.dependency_overrides[get_agent] = agent_of(lambda: FakeLLM(QUICK_RATIO))
    client = FilingLensClient(http=TestClient(app))
    events = list(client.ask_stream("What is AMD's FY2022 quick ratio?"))
    assert [e.data["node"] for e in events if e.type == "step"] == [
        "guard", "plan", "tools", "plan", "answer", "check"
    ]  # fmt: skip
    result = events[-1].result()
    assert result.answer == "AMD's FY2022 quick ratio is 1.57 [1]."
    assert result.verified
