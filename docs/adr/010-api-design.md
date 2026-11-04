# ADR-010: API design: sync agent in a threadpool, a fresh agent per request, SSE step events

Status: accepted (M4, 2026-11-04)

## Context

The M3 agent is synchronous (sync SQLAlchemy sessions, `llm.invoke()`), takes 1-30 s per question, and its
tools hold a database session (`make_tools(session, embedder)`). Measured before building: creating the model
clients costs 1,251 ms, building the agent from existing clients 6-7 ms, and 99.8% of an answer's time is LLM
calls (PepsiCo: 12.5 s total, tools 25 ms). M4 has to serve it over HTTP to a demo UI and other clients.

## Decision

- **Plain `def` endpoints**: FastAPI runs them in its threadpool (40 threads), so the sync agent never blocks
  the event loop.
- **Model clients once, agent per request**: the `lifespan` creates the chat and embedding clients; each request
  gets its own DB session (`Depends(get_db)`) and its own agent built on that session (`Depends(get_agent)`).
- **`POST /ask`** returns the whole `AskResponse` (answer, outcome, verified, cited sources, tool calls, rounds,
  attempts, latency, request id; `trace_id` reserved for Langfuse).
- **`POST /ask/stream`** streams Server-Sent Events with a plain `StreamingResponse`: one `step` event per
  finished graph node (`agent.stream(stream_mode=["updates", "values"])`, with a human-readable `label`), then a
  `result` event with the `/ask` JSON. Steps, not tokens: the answer passes the grounding check, which may ask
  for a rewrite, so draft text is not streamed.
- **Errors**: 422 for bad input (question 3-2,000 characters), 502 for OpenAI/Anthropic SDK errors, 503 for
  database errors; after a stream has started, an `error` event with the same status. Refusals and "not found"
  are 200s with their `outcome`.
- **The UI is a thin client**: Streamlit uses only `filinglens.api.client`, never the agent or the database.

## Consequences

- Measured overhead of the API layer: 19 ms per `/ask`; the first stream event arrives after ~1 s (the guard).
- About 40 concurrent questions per process, enough for the demo; more needs more processes or the async path.
- Tests run the real graph, schemas and SSE encoding with a scripted model (`dependency_overrides`), the client
  against `httpx.MockTransport`, and the UI with Streamlit's `AppTest`.
- Alternatives not picked: a fully async agent (`ainvoke` + async sessions; the upgrade path if concurrency
  matters); websockets (two-way, more than a progress feed needs); token streaming of the answer (shows text
  the check may reject); LangGraph Platform / LangServe (would hide the API design this project is meant to
  show); a job queue with polling (useful past ~60 s per request, not at 1-30 s).
