# ADR-005: Store XBRL facts as reported, choose the value at query time

Status: accepted (M1, 2026-10-30)

## Context

SEC's `companyfacts` data is not one number per metric and year:

- every 10-K repeats the previous one or two years as comparison columns, so one period has several rows
  (Microsoft's FY2016 cost of revenue appears in the FY2016, FY2017 and FY2018 10-Ks);
- the `fy` field is the fiscal year of the filing that reported the fact, not of the period it describes;
- concepts get renamed (Microsoft: `CostOfRevenue` until FY2017, `CostOfGoodsAndServicesSold` after);
- later filings restate (Microsoft's FY2016 revenue: 85,320M as reported, 91,154M restated in FY2018);
- a 10-K can tag several related concepts (Microsoft FY2016: `CostOfRevenue` 32,780M total,
  `CostOfGoodsSold` 17,880M products only).

## Decision

`financial_fact` keeps every us-gaap 10-K fact exactly as reported, duplicates included, with a unique key
(company, concept, unit, period start, period end, accession) using `NULLS NOT DISTINCT` so instant
(balance-sheet) facts dedupe on re-runs. `get_metric()` chooses the value:

1. annual (350-380 days) or instant facts whose period falls in the fiscal year (our `fiscal_year`,
   computed from the period end, not SEC's `fy`);
2. the year's own 10-K first, i.e. the number as originally reported (what FinanceBench expects);
3. then the metric's concept priority (`METRICS` in `financials.py`);
4. instant facts at the fiscal year end; ties: original filing first.

The result says whether it came from the year's own 10-K (`as_reported`) and links the filing.

## Consequences

- The loader stays a dumb, idempotent copy; the choosing rules live in one tested function.
- Changing a rule needs no reload.
- The metric map is curated and checked against the 21 eval filings; unknown metrics raise, unreported
  ones return None (a bank has no cost of revenue), never a wrong substitute.
- Alternative not picked: deduplicating at load time, which hides the rule in the loader and needs a reload
  to change.
