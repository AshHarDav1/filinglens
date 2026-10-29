# M0 results: plain RAG proof of concept, go/no-go

**Decision: GO.** Plain RAG scored lower than the closed-book baseline (6/20 vs 11/20), but the
measurements show the loss comes from retrieval, not from the LLM or the approach. M1 and M2 target
retrieval directly.

Date: 2026-10-29. All numbers come from the saved runs in `eval/results/` listed at the end.

## What was tested

20 FinanceBench questions about 12 10-K filings from 10 companies (`eval/financebench_subset.jsonl`).

| | Baseline (closed book) | Plain RAG |
|---|---|---|
| Answerer | gpt-6-luna, default temperature | same |
| Context | none: the model answers from memory | top 6 chunks by cosine distance, filtered to the question's company and fiscal year |
| Chunks | | 2,000 characters, 200 overlap, `text-embedding-3-small`, no section or table awareness |
| Judge | claude-haiku-5-5 against the FinanceBench reference answer | same |

## Results

| | Baseline | Plain RAG |
|---|---|---|
| Correct (Haiku judge) | **11/20** | **6/20** |
| Answers that are "Not found in the filings" | 0/20 | 9/20 |
| Mean latency per answer | 9.45 s | 2.92 s |
| Answerer tokens, all 20 questions (input / output) | 1,841 / 14,982 | 53,957 / 4,367 |

- Per question, RAG fixed 3 baseline errors (NFLX current liabilities, Boeing 787 production rate,
  AmEx effective tax rate) and lost 8 questions the baseline got right.
- The baseline is noisy: with a Luna judge it also scored 11/20, but 6/20 questions changed verdict
  between the two runs. Differences of 1 to 2 questions are within noise at this size.
- The baseline cites nothing, so even its correct answers don't meet the product requirement that
  every claim links to a filing.

## Why RAG lost: retrieval hit rate

`eval/retrieval_hit.py` checks, without any LLM, whether the FinanceBench evidence passage is among
the top k retrieved chunks.

| k | hit@k |
|---|---|
| 6 (used by the RAG run) | 6/20 |
| 10 | 9/20 |
| 20 | 14/20 |
| 40 | 15/20 |

Comparing evidence rank with the RAG verdicts:

- Evidence in the top 6: RAG correct on **4/6**. When retrieval works, the answers are accurate.
- Evidence not in the top 6: RAG correct on **2/14**.
- **12 of the 14 wrong RAG answers** had no evidence in their 6 chunks. Only 2 are generation errors
  (NFLX EBITDA margin, which needs a calculation, and Verizon debt).
- 7 of the 8 questions the baseline won were retrieval misses.

The misses fall into two groups that need different fixes:

1. **Ranking (9 questions, evidence at rank 7 to 40).** The right chunk is retrieved but ordered too
   low. Fix: hybrid search (vector + full-text), section filters, reranking.
2. **Representation (5 questions, evidence beyond rank 40).** All five are financial statement
   tables (MSFT income statement and balance sheet, MMM segment sales table, AmEx income statement and
   summary table). The HTML-to-text step flattens tables to one value per line, so labels and numbers
   are separated, and embeddings carry little meaning for numbers. Example: the MSFT FY2016 income
   statement chunk ranks 73rd of 203 for "What is the FY2016 COGS for Microsoft?"; it also mixes the
   end of Item 7A with the start of Item 8, and the filing says "cost of revenue", never "COGS".
   Fix: numbers from XBRL (ADR-002), table-aware parsing, section-aware chunking.

The hit rate is a lower bound: it only credits the passage FinanceBench cites. The AmEx tax-rate
question was answered correctly from a prose chunk at rank 1 while the cited table ranked 79th.

## Why GO and not NO-GO

- The failure is localized and measurable: retrieval, with hit@k as the yardstick.
- When retrieval succeeds, the answers are exact and cited, which the baseline can't offer.
- There is clear headroom: the evidence is in the top 20 for 14/20 questions, against 6/20 in the top 6.
- Every planned fix maps to a measured failure:

| Failure | Fix | Milestone |
|---|---|---|
| Numbers in flattened tables | `financial_fact` from XBRL, `get_financials` tool | M1, M3 |
| Chunks mixing sections | section-aware chunking with a section column and contextual headers | M1 |
| Weak ordering, keyword mismatch | hybrid search with section filters | M2 |
| Question wording vs filing wording (COGS vs cost of revenue) | query rewriting in the agent's plan step | M3 |
| Right chunk at rank 7 to 20 | retrieve wide, rerank to a small k | M5 |

Each change is re-measured with `retrieval_hit.py` (cheap) and then the full eval (judged).

## Caveats

- 20 questions is small; one question is 5 percentage points.
- The LLM judge isn't spot-checked yet beyond a few answers.
- Cost per question isn't recorded in dollars yet; the runs record tokens only.

## Runs

- `eval/results/2026-10-27-baseline.json`: baseline, Luna judge, 11/20
- `eval/results/2026-10-28-baseline-haiku-judge.json`: baseline, Haiku judge, 11/20
- `eval/results/2026-10-28-rag-poc.json`: plain RAG, k=6, 6/20
- `eval/results/2026-10-29-hit-poc.json`: retrieval hit rate at k = 6, 10, 20, 40
