# M1 results: XBRL facts and structured chunks

Date: 2026-10-30. Numbers come from `eval/results/2026-10-30-m1-ablation.json` and `2026-10-30-hit-m1.json`.

## Numbers: XBRL facts

- `financial_fact` holds 105,092 10-K facts (us-gaap, all years) for the 10 subset companies, stored as reported,
  duplicates included. `financials.get_metric()` picks one value per company, metric and fiscal year:
  the year's own 10-K first (as originally reported), then the metric's concept priority.
- Checked against the filings: MSFT FY2016 cost of revenue 32,780M; NFLX FY2017 current liabilities 5,466.312M;
  AMZN revenue 177,866M vs 135,987M (+30.8%); PEP FY2022 operating income 11,512M and D&A 2,763M;
  AXP FY2022 income tax 2,071M on pretax income 9,585M (21.6%).
- Traps handled: SEC's `fy` is the filing's year, not the period's; every value repeats in later 10-Ks;
  concepts get renamed (MSFT `CostOfRevenue` → `CostOfGoodsAndServicesSold`); later 10-Ks restate
  (MSFT FY2016 revenue 85,320M as reported, 91,154M restated); a bank (AXP) has no cost of revenue (returns None).

## Text: sections and table rows

- `html_to_text` keeps table rows on one line. The M0 "one value per line" came from line breaks between `<td>`s.
- `split_sections` finds Items 1A, 7 and 8 in all 12 filings (skips the table of contents and running page headers).
- Chunks are built per Item, so none crosses two Items; every chunk has `section`.

## Retrieval: ablation (20 questions, same matcher)

| Variant | hit@6 | hit@10 | hit@20 | hit@40 |
|---|---|---|---|---|
| M0 chunks | 8/20 | 10/20 | 15/20 | 16/20 |
| + table rows | 6/20 | 9/20 | 14/20 | 16/20 |
| + rows + sections (**kept**) | 8/20 | 8/20 | 15/20 | 16/20 |
| + embedded long header | 6/20 | 7/20 | 12/20 | 17/20 |
| + embedded short header | 7/20 | 8/20 | 14/20 | 18/20 |

- The matcher changed in M1 (any 150-char window of the evidence instead of only the first), so M0 reads 8/20
  here, not the 6/20 reported in `docs/m0-results.md`.
- Embedded contextual headers pulled far misses closer (MSFT COGS rank 73 → 23) but pushed near hits down
  (ADBE 9 → 48, VZ 14 → 45). Search is filtered to one filing, so company and year are identical on every
  candidate chunk. A one-chunk test had suggested 73 → 2; it held every other chunk at M0, which made it misleading.
- Decision: keep sections + rows (no hit@6 change; structural benefits: section filters for M2, readable rows for
  the LLM); show the header context in the prompt, not in the embedding.
- Still beyond rank 40: MSFT 00552 (debt) and MSFT 04700 (COGS), both now answerable from XBRL;
  MMM 01865 (segment table) and AXP 00723 (operating margin). Retrieval gains must come from M2 (hybrid search,
  section filters) and M3 (query rewriting).

## Wider eval: 51 questions

20 questions were too few to see small changes (one question = 5 points). All 51 FinanceBench 10-K questions
for the same 10 companies are now in `eval/financebench_10k.jsonl` (21 filings; the 20-question set stays for
LLM-judged runs). Ingesting the extra filings exposed one gap: question 00476 is answered on the cover page
(securities registered under Section 12(b)), which `build_chunks` used to skip; the cover is now kept.

`2026-10-30-hit-m1-51q.json`: hit@6 23/51, hit@10 26/51, hit@20 39/51, hit@40 42/51. This is the baseline for M2.
