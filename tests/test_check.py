"""The grounding check on real answers from the agent's first runs (AMD, 3M, PepsiCo, Microsoft)."""

import pytest

from filinglens.agent.check import check_answer, numbers_in
from filinglens.agent.sources import Source


def xbrl(n: int, text: str) -> Source:
    return Source(n=n, kind="xbrl", label="XBRL", url="https://www.sec.gov/x", text=text)


def calc(n: int, text: str) -> Source:
    return Source(n=n, kind="calculation", label="calculate", url=None, text=text)


def passage(n: int, text: str) -> Source:
    return Source(n=n, kind="passage", label="10-K", url="https://www.sec.gov/x", text=text)


AMD = [
    xbrl(1, "AMD FY2022 cash = 4,835 USD millions (CashAndCashEquivalentsAtCarryingValue)"),
    xbrl(2, "AMD FY2022 short_term_investments = 1,020 USD millions (ShortTermInvestments)"),
    xbrl(3, "AMD FY2022 receivables = 4,126 USD millions (AccountsReceivableNetCurrent)"),
    xbrl(4, "AMD FY2022 current_liabilities = 6,369 USD millions (LiabilitiesCurrent)"),
    calc(5, "(4835 + 1020 + 4126) / 6369 = 1.56712"),
]
QUESTION = "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?"


def test_real_amd_answer_passes():
    answer = (
        "Yes—AMD's FY22 quick ratio was **1.56712x**, indicating a reasonably healthy liquidity "
        "profile on this measure [5]. It compares cash, short-term investments and receivables "
        "with current liabilities [1][2][3][4]."
    )
    assert check_answer(answer, AMD, QUESTION).ok


@pytest.mark.parametrize(
    ("shown", "ok"),
    [("1.57", True), ("1.6", True), ("1.567", True), ("1.5", False), ("1.62", False)],
)
def test_rounding_is_allowed_but_not_other_numbers(shown, ok):
    result = check_answer(f"The quick ratio is {shown} [5].", AMD, QUESTION)
    assert result.ok is ok
    if not ok:
        assert result.unsupported_numbers == [shown]


@pytest.mark.parametrize(
    ("shown", "ok"),
    [("$4.8 billion", True), ("$4,835 million", True), ("$4,835,000,000", True),
     ("$4.9 billion", False)],
)  # fmt: skip
def test_unit_changes_are_allowed(shown, ok):
    assert check_answer(f"Cash was {shown} [1].", AMD, QUESTION).ok is ok


def test_ratio_shown_as_percent():
    # PepsiCo FY2022 unadjusted EBITDA margin
    sources = [calc(1, "(11512 + 2763) / 86392 = 0.165235")]
    assert check_answer("The EBITDA margin was 16.5% [1].", sources, "EBITDA margin?").ok
    assert not check_answer("The EBITDA margin was 17.5% [1].", sources, "EBITDA margin?").ok


def test_sign_is_ignored_for_decreases():
    # Microsoft's debt, FY2022 -> FY2023
    sources = [calc(1, "47237 - 49781 = -2544")]
    assert check_answer("Debt decreased by $2,544 million [1].", sources, "Debt change?").ok


def test_passage_numbers_count_as_sources():
    # 3M FY2022: the margin bridge in MD&A
    sources = [
        calc(1, "(6539 / 34229) * 100 - (7369 / 35355) * 100 = -1.73923"),
        passage(2, "Raw material impacts decreased margins by 2.4 percentage points, partially "
                   "offset by organic growth and productivity of 1.0 percentage points."),
    ]  # fmt: skip
    answer = (
        "3M's operating margin declined 1.74 percentage points in FY2022 [1]. Raw materials cost "
        "2.4 points, partly offset by a 1.0-point benefit from organic growth [2]."
    )
    assert check_answer(answer, sources, "What drove operating margin change for 3M?").ok


def test_numbers_from_the_question_are_allowed():
    question = "DPO is defined as: 365 * (average accounts payable) / (COGS + change in inventory)"
    sources = [calc(1, "365 * 30000 / 120000 = 91.25")]
    assert check_answer("Using 365 days, DPO is 91.25 days [1].", sources, question).ok


def test_citations_must_exist():
    result = check_answer("The quick ratio is 1.57 [5][12].", AMD, QUESTION)
    assert result.bad_citations == [12]
    assert "Citation [12] does not exist; the sources are [1] to [5]." in result.problems(5)


def test_numbers_without_any_citation_fail():
    result = check_answer("The quick ratio is 1.57.", AMD, QUESTION)
    assert result.missing_citations
    assert not result.ok


def test_years_items_forms_counts_and_citations_are_not_data():
    answer = "In FY2022 and 2021, Item 7 and Item 1A of the 10-K describe 3 segments [1][2]."
    assert check_answer(answer, AMD, "q").ok


def test_number_extraction():
    assert [n.text for n in numbers_in("4,835 and 1.56712x, -2,544, 36.8%, FY22, Q4")] == [
        "4,835",
        "1.56712",
        "2,544",
        "36.8",
    ]
