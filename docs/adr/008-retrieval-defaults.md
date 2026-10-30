# ADR-008: Keep vector search as the default; hybrid search and routing stay optional

Status: accepted (M2, 2026-10-30)

## Context

M2 built Postgres full-text search (a generated `tsvector` column with a GIN index), hybrid search with
reciprocal rank fusion, and section routers (keyword rules and a small LLM). A prototype showed that searching
only the Items that hold the evidence would lift hit@6 from 23/51 to 36/51, if the right Items were known.

Variants were chosen on 31 dev questions and confirmed on the 20 held-out ones (`docs/m2-results.md`):

- hybrid + LLM router won on dev (19/31 vs 15/31) but lost on test (7/20 vs 8/20);
- vector + LLM router, after fixing filings whose Item 8 is a pointer, scored 10/20 on test (not a clean
  held-out number, the fix came after seeing test results) and 24/51 vs 23/51 overall;
- end to end, the LLM-judged score was 13/20 with and without the router.

## Decision

Default retrieval is vector search within the filing. Keyword search, hybrid fusion and both routers ship as
tested options (`retrieve(method=..., sections=..., section_mode=...)`), selectable in the eval runners. Section
choice moves to the M3 agent's plan step, which can retry without a filter when the first search finds nothing.

## Consequences

- No extra LLM call or latency on the default path.
- The full-text index is in place for exact-term lookups the agent may need (names, defined terms).
- If a later change (a better router, a reranker, cleaner questions from the agent) makes these options win on
  held-out questions, switching is a parameter change.
- Alternative not picked: adopting the dev winner, which would have shipped a variant that lost on held-out data.
