# v2 ideas

Ideas that are out of scope for v1. Add one line each, with a short reason.

- Split Items at sub-headings (statements, "Note 13 — Income Taxes"): 22 of Microsoft's 75 Item 8 chunks still
  span two notes. Sub-heading formats differ by company, so only if a measurement shows it matters.
- Contextual chunk headers for cross-filing search: they lowered hit@6 when search is filtered to one filing
  (ADR-006) but could help once a question spans several companies or years.
- Synthetic retrieval questions generated from our own chunks: hundreds of questions with known source
  chunks; optimistic because they reuse the chunk's wording, so for retrieval checks only.
- More eval data beyond FinanceBench's public 150: DocFinQA (questions over full filings), FinQA and TAT-QA
  (table + text reasoning). Their formats don't map directly to EDGAR filings.
- Restated vs as-reported numbers as a user choice: `get_metric` returns as-reported; analysts sometimes
  want the latest restated figure.
- 10-Q and 8-K filings (15 + 9 FinanceBench questions), and the company-specific XBRL extension concepts
  that `load_facts` skips.
