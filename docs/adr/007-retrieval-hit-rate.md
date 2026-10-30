# ADR-007: Retrieval hit rate as the per-change metric

Status: accepted (M0, 2026-10-29); matcher revised in M1 (2026-10-30)

## Context

The M0 RAG run scored 6/20, below the closed-book baseline (11/20). Answer accuracy alone can't say whether
the search or the LLM failed, and an LLM-judged run is slow, costs money and is noisy (the judge flipped
6 of 20 verdicts between two identical baseline runs).

## Decision

Measure retrieval on its own with hit@k (`eval/retrieval_hit.py`): for each question, is the FinanceBench
evidence passage inside the top k retrieved chunks (k = 6, 10, 20, 40)? No LLM is involved; it runs in
seconds and costs only the question embeddings. Run it after every retrieval change; confirm with the
LLM-judged eval at milestones.

Matching normalizes both texts to lowercase letters and digits. Since M1 a hit is any 150-character window
of the evidence (step 50) found in a chunk; matching only the first 150 characters missed passages whose
start falls on a section border. Before trusting a run, every evidence passage must be findable somewhere
in its filing.

## Consequences

- In M0 this showed 12 of 14 wrong RAG answers had no evidence in their 6 chunks: the search was the problem.
- hit@k is a lower bound: a chunk stating the same fact elsewhere in the filing doesn't count.
- Matcher v2 changed M0's score from 6/20 to 8/20; results are only compared under the same matcher.
