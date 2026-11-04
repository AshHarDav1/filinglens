# ADR-009: Agent design: typed tools, a cheap guard, fixed exits and a code grounding check

Status: accepted (M3, 2026-11-04)

## Context

M2's single-pass RAG answered 13/20. Its misses were mostly number questions (quick ratio, debt change,
EBITDA and operating margins) whose values already sat in XBRL (`get_metric`, ADR-002) but could not be
reached from a pipeline that only retrieves text. ADR-003 fixed the shape (LangGraph, explicit steps, a round
limit); this ADR records how the steps work.

## Decision

- **Tools are plain functions with Pydantic inputs, wrapped for LangChain** (`agent/tools.py`). They return
  `content_and_artifact`: compact text for the LLM, the typed result for our code. Metric and section names are
  enums in the schema, so the LLM cannot ask for a metric that doesn't exist. Errors are `ToolException`s
  written for the LLM ("Unknown ticker 'TSLA'. FilingLens covers only: ...").
- **calculate parses with `ast` and allows only numbers, + - * / and parentheses** (no `eval`, no `**`).
- **A guard runs first on the small model** (claude-haiku-5-5, structured output: ok / advice / off_topic).
  Unknown companies pass as ok; the tools then lead to "not found".
- **The planner (gpt-6-luna) has at most 4 rounds of tool calls**; one round can hold parallel calls. Calls
  requested after the limit are dropped and the answer is written from what was gathered.
- **The answer step sees only numbered sources** built from the tool artifacts (XBRL values, calculations,
  passages), not the raw conversation, and must cite them as [n].
- **A grounding check in code** (`agent/check.py`) requires every number in the answer to appear in a source
  or the question (allowing rounding and unit changes) and every [n] to exist. One rewrite with the problems
  listed; if it still fails, the answer is returned with `verified=False`.
- **Refusals and "not found" are fixed texts**, not generated.

## Consequences

- 16/20 vs 13/20 (M2 RAG) and 11/20 (closed-book) on the same 20 questions and judge (`docs/m3-results.md`).
- About 3x the tokens and 3.7x the latency of M2 (mean 13.9 s, max 29.1 s): 4.9 LLM calls per question.
- The check verifies where numbers came from, not whether the right numbers were used: two wrong answers in
  the run passed it (a wrong comparison year, a wrong XBRL concept).
- Alternatives not picked: a free-form ReAct loop (unbounded, ADR-003); LangChain's prebuilt agent (no place
  for the guard or the check); an LLM-based citation checker (a second model call for something code can
  verify exactly); one combined prompt for screening, planning and answering (harder to test and to protect
  from injected text).
