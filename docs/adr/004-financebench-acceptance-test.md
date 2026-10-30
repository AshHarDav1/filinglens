# ADR-004: FinanceBench as the acceptance test

Status: accepted (M0, 2026-10-22); widened in M1 (2026-10-30)

## Context

Quality claims need questions with known answers and known source passages, about real filings.

## Decision

Use the public FinanceBench sample (PatronusAI, CC-BY-NC-4.0): questions about real 10-Ks, each with a
reference answer and the evidence passage. Companies and filing years for v1 are picked from the questions,
so the data we ingest matches the test.

Two question sets, both from the same 10 companies:

- `eval/financebench_10k.jsonl`: all 51 FinanceBench 10-K questions (21 filings). Used for the retrieval
  hit rate, which needs no LLM and is cheap.
- `eval/financebench_subset.jsonl`: 20 of them. Used for LLM-judged runs, to keep their cost low.

## Consequences

- Retrieval is measured on its own (hit@k: is the evidence in the top k chunks?), separately from answer
  accuracy (LLM judge against the reference answer).
- Runs are only compared on the same question set. The closed-book baseline exists for the 20; the next
  full run will add closed-book and system results on the 51.
- 20 questions are too few to see small changes (one question = 5 points); 51 still leave about
  plus or minus 7 questions of uncertainty (95%) at 23/51.
