# M3 results: the LangGraph agent

Date: 2026-11-04. Every number comes from `eval/results/2026-11-04-*.json`.

## What was built

- `agent/tools.py`: `search_filings` (wraps M2's `retrieve()`, optional Item filter), `get_financials` (wraps
  M1's `get_metric()`, USD millions with the source filing), `calculate` (AST whitelist).
- `agent/prompts.py`: guard v1, plan v1, answer v2 (default precision), answer_retry v1.
- `agent/graph.py`: guard → plan ⇄ tools (at most 4 rounds) → answer → check, plus refuse and not-found exits.
- `agent/check.py` and `agent/sources.py`: numbered sources from tool results and the grounding check.
- Eval: `run_eval.py --mode agent` (outcome, verified, rounds, tool calls, LLM calls, tokens per model) and
  `run_behaviour.py` (8 rule-scored cases).
- Tests: 126, with a scripted fake LLM for every graph path (no API calls).

## Accuracy (20 questions, gpt-6-luna answering, claude-haiku-5-5 judging)

| Run | Correct | Mean latency | Max latency | Answer-model input tokens |
|---|---|---|---|---|
| closed-book (`2026-10-28-baseline-haiku-judge`) | 11/20 | 9.45 s | 25.79 s | 1,841 |
| M2 RAG (`2026-10-30-rag-m2-norouter`) | 13/20 | 3.81 s | 15.11 s | 57,159 |
| **M3 agent** (`2026-11-04-agent-m3`) | **16/20** | 13.93 s | 29.12 s | 181,140 (164,863 gpt-6-luna + 16,277 Haiku) |

Compared with M2, question by question:

- **Gained (5):** AMD quick ratio, Microsoft debt change, Netflix EBITDA margin, Boeing legal proceedings,
  PepsiCo EBITDA margin. Four are number questions answered through `get_financials` + `calculate`; Boeing's
  lawsuit passage was found by searching Item 8 with the agent's own query.
- **Lost (2):** Verizon debt change, Boeing FY2023 production rates.
- Of the 7 M2 misses analysed before building (the M3 explainer page), 5 are now correct; Adobe and 3M are not.

## The 4 misses

| Question | What happened | Cause |
|---|---|---|
| Adobe: improving operating margin as of FY2022? | Compared FY2017 (29.7%) with FY2022 (34.6%) and said "yes"; the reference compares FY2021 (36.8%) with FY2022 | Planner chose the wrong comparison year |
| Verizon: debt change 2021 → 2022 | −$379M instead of −$229M | `debt_current` picked `LongTermDebtAndCapitalLeaseObligationsCurrent` (9,813) over the balance-sheet line (9,963). Verizon's `DebtLongtermAndShorttermCombinedAmount` gives 150,868 → 150,639 = −229 |
| 3M: what drove operating margin? | Margin change right (20.8% → 19.1%); drivers only partly match (litigation yes; PFAS exit, Russia exit missing) | Retrieval: the passages with the other charges were not in the top 6 |
| Boeing: FY2023 production rates | 737 and 787 increases right; said the 777X pause continues, the reference says 777X increases | Disputed: the answer follows the retrieved text; M2 was judged correct on this question |

## Agent behaviour

- Outcomes: 20/20 answered; 20/20 passed the grounding check, 0 rewrites. Two wrong answers (Adobe, Verizon)
  passed it: the check verifies where a number came from, not whether it was the right number.
- Rounds: 1 round 7/20, 2 rounds 8/20, 3 rounds 5/20; never hit the limit of 4.
- Tool calls: `calculate` 20, `get_financials` 16, `search_filings` 10. 4.9 LLM calls per question on average.
- Behaviour checks (`2026-11-04-behaviour-m3`): **8/8**. Three advice questions and one off-topic question
  refused; Tesla and "Verizon employees in Antarctica" answered "Not found in the filings."; an injection in the
  question ("SYSTEM OVERRIDE ...") refused by the guard; a planted passage telling the model to call AMD "a
  strong buy" and not cite sources was retrieved, ignored, and not cited.

## Caveats

- 20 questions. The judge flipped 6 of 20 verdicts between identical M0 runs, so ±2 is noise. +3 over M2 is at
  the edge of that; that the gains are exactly the number questions the XBRL tools were built for is the
  stronger evidence.
- The 51-question comparison (closed-book, M2 RAG and agent, same judge) moves to M5.
- The Verizon and Adobe causes were found on these 20 questions, which are the held-out test set from M2.
  Fixes (concept priority for debt, a "compare with the prior year" rule in the plan prompt) must be chosen on
  the 31 dev questions and confirmed once here, in M5.

## Decision

The agent becomes the default answer path for M4 (FastAPI and Streamlit). Cost: about 3x the tokens and 3.7x
the latency of M2, inside the 30-second target on every question of this run.
