# M4 results: HTTP API and demo UI

Date: 2026-11-04. Measured against the running API (`uvicorn filinglens.api.main:app --port 8010`) with the
real models (gpt-6-luna answering, claude-haiku-5-5 guarding). Single runs, so timings are examples, not
averages.

## What was built

- `api/main.py`: `create_app()`, `lifespan` (model clients once), per-request DB session and agent,
  `GET /health`, `POST /ask`, `POST /ask/stream`, error mapping (422 / 502 / 503 / `error` event).
- `api/schemas.py`: `AskRequest`, `AskResponse`, `SourceOut`, `ToolCallOut`, `HealthResponse`.
- `api/events.py`: SSE encoding and one readable `label` per agent step.
- `api/client.py`: `FilingLensClient` (`health`, `ask`, `ask_stream`) with an SSE parser; `API_URL` setting.
- `ui/app.py`: Streamlit chat over `/ask/stream`: live steps, answer, verified badge, sources with SEC links,
  tool calls, API status, three example questions.
- Tests: 152 (26 new across API, client and UI), no network and no real models.

## Measurements

| What | Result |
|---|---|
| `/health` | 200: database ok, 10 companies, 21 10-Ks, models gpt-6-luna / claude-haiku-5-5 / text-embedding-3-small |
| `/ask`, PepsiCo EBITDA margin | 13,220 ms at the client, 13,201 ms in the agent: **19 ms** for the API layer (session, agent build, JSON, network) |
| `/ask`, "Should I buy AMD?" | refused in 1,182 ms |
| `/ask/stream`, PepsiCo | **first event at 1,071 ms** (guard), result at 10,566 ms, 9 events |
| `/ask/stream`, Boeing legal battles | first event at 923 ms, two searches (Items 3/8/1A, then Item 8), result at 15,188 ms |
| Server → client delay per event | 13-17 ms: events are not buffered |
| Streamlit UI against the real API | Boeing question: 6 live steps, cited answer with 2 SEC links, "verified", 12.0 s |

The PepsiCo stream, as the client saw it (ms since the request):

| ms | node | label |
|---|---|---|
| 1,071 | guard | Question is in scope |
| 3,194 | plan | XBRL PEP FY2022: operating_income, depreciation_amortization, revenue |
| 3,218 | tools | 3 XBRL values |
| 4,837 | plan | calculate ((11512 + 2763) / 86392) * 100 |
| 4,839 | tools | = 16.5235 |
| 6,681 | plan | Done gathering data |
| 10,564 | answer | Answer written |
| 10,565 | check | Checked: every number traces to a source |
| 10,566 | result | "PepsiCo's FY2022 unadjusted EBITDA margin was **16.5%** [4]." |

## Notes and limits

- Answer quality is the M3 agent's (16/20); the API calls the same `ask()` path, so no new eval run.
- Port 8000 was taken on the dev machine, so the API ran on 8010 (`API_URL=http://localhost:8010` for the UI).
- Not built yet: authentication and rate limits (each question costs money), Langfuse (`trace_id` is null),
  keep-alive pings for long streams behind proxies, token streaming of the final answer, Docker images for the
  API and UI (M6).
