"""The Streamlit UI with Streamlit's AppTest and a fake API client: no server, no network."""

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from filinglens.api.client import FilingLensError, StreamEvent
from filinglens.api.schemas import HealthResponse, ModelsInfo

APP = str(Path(__file__).parent.parent / "ui" / "app.py")
RESULT = {
    "request_id": "abc", "question": "q", "outcome": "answered", "verified": True,
    "answer": "PepsiCo's FY2022 unadjusted EBITDA margin was 16.5% [4], using operating "
              "income of $11,512 million [2].",
    "sources": [
        {"n": 2, "kind": "xbrl", "label": "XBRL · PEP FY2022 operating_income",
         "url": "https://www.sec.gov/Archives/edgar/data/77476/000007747623000007/"},
        {"n": 4, "kind": "calculation", "label": "calculate", "url": None},
    ],
    "tool_calls": [{"name": "calculate", "args": {"expression": "((11512 + 2763) / 86392) * 100"}}],
    "rounds": 2, "attempts": 1, "latency_ms": 12492, "trace_id": None,
}  # fmt: skip


class FakeClient:
    def __init__(self, events=None, error=None, healthy=True):
        self.events, self.error, self.healthy = events or [], error, healthy
        self.questions: list[str] = []

    def health(self):
        if not self.healthy:
            raise httpx.ConnectError("refused")
        models = ModelsInfo(answer="gpt-6-luna", guard="claude-haiku-5-5", embeddings="e")
        return HealthResponse(status="ok", database="ok", companies=10, filings=21,
                              models=models, version="0.1.0")  # fmt: skip

    def ask_stream(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        yield from self.events


def app_with(client) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=20)
    at.session_state["client"] = client
    return at.run()


def test_first_screen_shows_status_and_examples():
    at = app_with(FakeClient())
    assert not at.exception
    assert at.title[0].value == "FilingLens"
    assert [b.label for b in at.button][:3] == [
        "PepsiCo EBITDA margin", "Boeing legal battles", "Should I buy AMD?"
    ]  # fmt: skip
    assert "10 companies · 21 10-Ks" in at.sidebar.success[0].value


def test_example_question_streams_and_shows_the_answer_with_sources():
    events = [
        StreamEvent(
            type="step", data={"node": "guard", "elapsed_ms": 1243, "label": "Question is in scope"}
        ),
        StreamEvent(type="result", data=RESULT),
    ]
    client = FakeClient(events)
    at = app_with(client)
    at.button[0].click().run()
    assert not at.exception
    assert client.questions[0].startswith("What is the FY2022 unadjusted EBITDA")
    texts = [m.value for m in at.markdown]
    assert any(r"\$11,512 million" in t for t in texts)  # dollars escaped, not LaTeX
    assert any("✅ verified · 2 tool rounds · 12.5 s" in c.value for c in at.caption)
    assert [e.label for e in at.expander] == ["Sources (2)", "Agent steps", "Tool calls"]
    assert any("[SEC filing](https://www.sec.gov/" in t for t in texts)


def test_api_errors_are_shown_not_raised():
    at = app_with(FakeClient(error=FilingLensError(502, "Model provider error (RateLimitError).")))
    at.chat_input[0].set_value("What is AMD's FY2022 quick ratio?").run()
    assert not at.exception
    assert "API error 502" in at.error[0].value


def test_api_down():
    at = app_with(FakeClient(healthy=False, error=httpx.ConnectError("refused")))
    assert "API not reachable" in at.sidebar.error[0].value
    at.chat_input[0].set_value("What is AMD's FY2022 quick ratio?").run()
    assert any("Can't reach the API" in e.value for e in at.main.error)
