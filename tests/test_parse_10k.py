from itertools import pairwise
from pathlib import Path

import pytest

from filinglens.ingestion.html_text import html_to_text
from filinglens.ingestion.parse_10k import split_sections

FILLER = "Lorem ipsum dolor sit amet. " * 30  # ~840 chars of body text

# A tiny 10-K with the three traps: a table of contents, running page headers, a cross-reference
FAKE_10K = f"""FORM 10-K
ACME CORP
Table of Contents
Item 1. Business
Item 1A. Risk Factors
Item 7. Management's Discussion and Analysis
Item 8. Financial Statements
Item 15. Exhibits

PART I
Item 1. Business
{FILLER}
Item 1A. Risk Factors
{FILLER} As discussed in Item 7 of this report, margins may fall.
{FILLER}
PART II
Item 7. Management's Discussion and Analysis
{FILLER}
Item 8. Financial Statements and Supplementary Data
INCOME STATEMENTS
{FILLER}
PART II
Item 8
{FILLER}
PART IV
Item 15. Exhibits
{FILLER}
"""


def test_sections_skip_toc_running_headers_and_cross_references():
    sections = split_sections(FAKE_10K)
    assert [s.item for s in sections] == ["cover", "1", "1A", "7", "8", "15"]

    cover = sections[0]
    assert "Table of Contents" in cover.text  # the TOC stays in the cover, not in Item 1

    item8 = next(s for s in sections if s.item == "8")
    assert item8.text.startswith("Item 8. Financial Statements")
    assert item8.text.count("Item 8") == 2  # the running header stayed inside Item 8
    assert item8.title == "Financial Statements and Supplementary Data"


def test_sections_tile_the_text_without_gaps_or_overlaps():
    sections = split_sections(FAKE_10K)
    assert sections[0].start == 0
    for previous, nxt in pairwise(sections):
        assert previous.end == nxt.start
    assert sections[-1].end == len(FAKE_10K)
    assert "".join(s.text for s in sections) == FAKE_10K


def test_item_10_is_not_item_1():
    text = f"Item 1. Business\n{FILLER}\nItem 10. Directors\n{FILLER}"
    assert [s.item for s in split_sections(text)] == ["1", "10"]


def test_text_without_headings_is_kept_whole():
    sections = split_sections("Just some text with no items.")
    assert [(s.item, s.text) for s in sections] == [("cover", "Just some text with no items.")]


RAW = Path("data/raw/www.sec.gov/Archives/edgar/data")
REAL_FILINGS = sorted(RAW.glob("*/*/*.htm")) if RAW.exists() else []


@pytest.mark.skipif(not REAL_FILINGS, reason="raw filings not downloaded")
@pytest.mark.parametrize("path", REAL_FILINGS, ids=lambda p: p.name)
def test_real_filings_have_the_main_items_in_order(path):
    sections = split_sections(html_to_text(path.read_text(encoding="utf-8", errors="replace")))
    items = [s.item for s in sections]
    for item in ("1A", "7", "8"):
        assert item in items, f"{path.name}: missing Item {item}"
    assert items.index("1A") < items.index("7") < items.index("8")
