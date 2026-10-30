# ADR-002: Numbers come from XBRL, not from text or tables

Status: accepted (M0, 2026-10-22); implemented in M1

## Context

Analysts ask for exact figures ("FY2016 COGS"). In the M0 proof of concept, number questions failed at the
retrieval step: financial-statement tables embed badly (Microsoft's FY2016 income statement ranked 73rd of
203 chunks for a COGS question), and an LLM reading numbers from flattened tables can pick the wrong row or
column. Every number in a 10-K's financial statements is also filed with the SEC as a tagged XBRL fact.

## Decision

Exact numbers come from SEC XBRL facts (`companyfacts` API) stored in `financial_fact`, looked up with
`financials.get_metric()` (the agent's `get_financials` tool in M3). Text retrieval is for explanations,
risks and context. Arithmetic goes through a `calculate` tool, not the LLM.

## Consequences

- Number answers are exact and cite the filing they came from.
- A metric map (metric name to XBRL concepts) has to be maintained; see ADR-005 for how values are chosen.
- Only tagged numbers are covered. Figures that exist only in prose or in untagged tables (segment details,
  some non-GAAP measures) still need text retrieval.
