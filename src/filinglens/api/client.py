"""Python client for the FilingLens API. The Streamlit UI talks to the backend only through this.

with FilingLensClient() as client:              # API_URL from .env, default localhost:8000
    for event in client.ask_stream("What is AMD's FY2022 quick ratio?"):
        print(event.type, event.data.get("label"))
"""

import json
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from filinglens.api.schemas import AskResponse, HealthResponse
from filinglens.config import get_settings

# An answer takes up to ~30 s, and a stream can be quiet for one long LLM call (~8 s seen).
TIMEOUT = httpx.Timeout(120.0, connect=5.0)


class FilingLensError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class StreamEvent(BaseModel):
    type: Literal["step", "result", "error"]
    data: dict[str, Any]

    def result(self) -> AskResponse:
        return AskResponse.model_validate(self.data)


def parse_sse(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Server-Sent Events from text lines: `event:` and `data:` fields, a blank line ends an
    event, lines starting with `:` are comments (keep-alive pings)."""
    event, data = None, []
    for line in lines:
        if not line:
            if event and data:
                yield StreamEvent(type=event, data=json.loads("\n".join(data)))
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if event and data:
        yield StreamEvent(type=event, data=json.loads("\n".join(data)))


def _raise_for_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise FilingLensError(response.status_code, str(detail))


class FilingLensClient:
    def __init__(self, base_url: str | None = None, http: httpx.Client | None = None):
        """http: an existing client (tests pass FastAPI's TestClient, which is one)."""
        self._http = http or httpx.Client(
            base_url=base_url or get_settings().api_url, timeout=TIMEOUT
        )

    def health(self) -> HealthResponse:
        """The API's /health. A 503 (database down) is returned too, with status "error"."""
        response = self._http.get("/health")
        if response.status_code == 503:
            return HealthResponse.model_validate(response.json())
        _raise_for_error(response)
        return HealthResponse.model_validate(response.json())

    def ask(self, question: str) -> AskResponse:
        response = self._http.post("/ask", json={"question": question})
        _raise_for_error(response)
        return AskResponse.model_validate(response.json())

    def ask_stream(self, question: str) -> Iterator[StreamEvent]:
        with self._http.stream("POST", "/ask/stream", json={"question": question}) as response:
            if not response.is_success:
                response.read()
                _raise_for_error(response)
            yield from parse_sse(response.iter_lines())

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "FilingLensClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
