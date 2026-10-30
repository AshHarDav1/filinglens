# ADR-006: Chunk per 10-K Item; no embedded contextual header

Status: accepted (M1, 2026-10-30)

## Context

M0 cut each filing into blind 2,000-character windows. Some chunks mixed two Items (Microsoft FY2016
chunk 115: the end of Item 7A market risk plus the start of the Item 8 income statement), and HTML tables
came out as one value per line. A common fix is a "contextual header" (company, year, section) prepended to
each chunk before embedding.

We measured five variants on the same 20 questions with the same matcher
(`eval/results/2026-10-30-m1-ablation.json`):

| Variant | hit@6 | hit@20 |
|---|---|---|
| M0 chunks | 8/20 | 15/20 |
| + table rows | 6/20 | 14/20 |
| + table rows + per-Item chunks | 8/20 | 15/20 |
| + embedded long header | 6/20 | 12/20 |
| + embedded short header | 7/20 | 14/20 |

## Decision

- Convert tables to one line per row (`Total cost of revenue | 32,780 | 33,038 | 27,078`).
- Split each 10-K into Items (`parse_10k.split_sections`: skip the table of contents, accept a heading only
  if it comes later in the outline), chunk each Item on its own, store the Item in `chunk.section`.
  Keep the cover page as a section; skip near-empty Items.
- Do not embed a contextual header. Show the context to the LLM in the prompt instead
  (`[1] MSFT 10-K FY2016 · Item 8 (chunk 12)`).

## Consequences

- No retrieval gain at hit@6 (8/20 vs 8/20); the value is structural: no chunk crosses two Items, every
  chunk has a section for M2's filters, and the LLM reads intact table rows.
- Headers didn't help because search is already filtered to one filing: company and year are identical on
  every candidate chunk. A one-chunk experiment had suggested the opposite (rank 73 to 2) because only that
  chunk had a header; the full measurement overturned it.
- Chunks inside an Item can still span two sub-sections (e.g. the end of Note 12 and the start of Note 13);
  splitting at sub-headings is parked in `docs/v2-ideas.md` until a measurement asks for it.
