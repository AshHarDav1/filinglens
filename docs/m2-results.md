# M2 results: hybrid search and section routing

Date: 2026-10-30. Every number comes from `eval/results/2026-10-30-*.json`.

## What was built

- `chunk.content_tsv`: a generated `tsvector` column with a GIN index (Postgres full-text search).
- `retrieval/hybrid.py`: `keyword_search` (question words OR-ed after dropping instruction words such as
  "please", "answer", "millions"), reciprocal rank fusion (`rrf`), and `retrieve()` with an optional
  section filter or boost. If a filing's Item 8 is a pointer (fewer than 5 chunks), Items 15 and 16 stand in.
- `retrieval/routing.py`: section routers, keyword rules and a small LLM call (claude-haiku-5-5, router
  prompt v1).
- Citations: RAG excerpts and saved sources carry company, form, fiscal year, Item and filing URL; RAG prompt v2.

## How it was measured

Retrieval hit rate on the 51-question set, split into **dev** (31 questions, used to choose) and **test**
(the 20-question set, used once to confirm). The "oracle" router is told the Items that hold the evidence:
a ceiling, not a real router.

| Variant (dev, 31 questions) | hit@6 | hit@20 |
|---|---|---|
| vector (M1) | 15/31 | 24/31 |
| keyword only | 12/31 | 18/31 |
| hybrid (vector + keyword, RRF) | 16/31 | 23/31 |
| vector + rules router, filter | 15/31 | 24/31 |
| vector + LLM router, filter | 17/31 | 26/31 |
| hybrid + LLM router, filter | 19/31 | 26/31 |
| oracle sections (ceiling) | 24/31 | 30/31 |

Confirmation on test (20 questions):

| Variant | hit@6 | hit@20 |
|---|---|---|
| vector (M1) | 8/20 | 15/20 |
| hybrid + LLM router (the dev winner) | 7/20 | 11/20 |
| after the Item 8 pointer fix: vector + LLM router | 10/20 | 15/20 |
| after the fix: hybrid + LLM router | 9/20 | 13/20 |
| oracle sections (ceiling) | 12/20 | 17/20 |

- The dev winner did not hold on test: Netflix's statements sit after Item 15, so routing to "Item 8"
  hid them (rank 1 → 37). The fix (Item 15/16 stand in for a pointer Item 8) was made after looking at
  test results, so the post-fix test numbers are not a clean held-out check.
- The LLM router is not deterministic: identical dev runs moved by 1-2 questions.
- On all 51 questions, vector + LLM router scores hit@6 24/51 vs 23/51 for vector alone
  (`m2all-llm-filter-v2`): within noise.

## Full pipeline (LLM answer + judge, 20 questions)

| Run | Correct | "Not found" | Mean latency |
|---|---|---|---|
| closed-book (no retrieval) | 11/20 | 0/20 | 9.45 s |
| M0 plain RAG (`rag-poc`) | 6/20 | 9/20 | 2.92 s |
| M1 chunks, prompt v2, no router (`rag-m2-norouter`) | **13/20** | 6/20 | 3.81 s |
| M1 chunks, prompt v2, LLM router (`rag-m2-router`) | 13/20 | 6/20 | 4.19 s |

- RAG now beats the closed-book baseline (13/20 vs 11/20) for the first time.
- The gain from 6/20 comes from M1 (table rows the LLM can read, chunks that don't mix Items) and the
  labelled excerpts of prompt v2, not from routing: with and without the router the score is the same.
  Retrieval hit@6 had not moved in M1 (8/20 → 8/20); the same chunks became readable.
- The judge itself flips verdicts between runs (6 of 20 for the M0 baseline), so differences of 1-2
  questions are noise.

## Decision

- Default retrieval stays vector search within the filing (no router, no keyword search). Hybrid search and
  the routers are in the code, tested, and selectable (`--method`, `--router`), but none gave a gain that
  survived the held-out check.
- Section choice moves to the M3 agent's plan step, where the agent can also retry without a filter.
- Number questions go to XBRL in M3 (`get_metric`): most remaining wrong answers involve figures or
  calculations (quick ratio, debt change, operating and EBITDA margins).
