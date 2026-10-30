"""Split a 10-K's plain text into its Item sections (Item 1, 1A, ..., 7, 7A, 8, ..., 16).

Every 10-K follows the same outline set by the SEC, so Items are a reliable border for chunking:
a chunk never mixes two Items, and each chunk knows its section. Three things make a plain regex
for "Item 8" not enough, all seen in the eval filings:

- the table of contents lists every heading first, a few lines apart;
- running page headers repeat the current Item on every page ("PART II / Item 8": 46 times in
  Microsoft's FY2016 10-K);
- some filings keep content outside its usual Item (Netflix's statements follow Item 15,
  PepsiCo's sit inside Item 7). That's why sections are labelled, never dropped.
"""

import re

from pydantic import BaseModel

ITEM_ORDER = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
              "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"]  # fmt: skip

# Standard 10-K item names (Form 10-K / Regulation S-K). Used for chunk headers, because the
# heading line found in a filing may be a bare running header ("Item 8") without the name.
ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity and Related Stockholder Matters",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# A heading LINE: "Item 7A.", "ITEM 8 - Financial ...", "Item 1A". Longer numbers come first in
# the alternation so "Item 10" isn't read as "Item 1" followed by "0".
_ITEMS_BY_LENGTH = sorted(ITEM_ORDER, key=len, reverse=True)
HEADING = re.compile(
    r"^[ \t]*item[ \t]*("
    + "|".join(_ITEMS_BY_LENGTH)
    + r")[ \t]*[.:\-\u2014\u2013]?(?=\s|$)",  # . : - em/en dash
    re.IGNORECASE | re.MULTILINE,
)
# Table-of-contents lines sit close together; a real Item 1 is followed by pages of text
TOC_GAP_CHARS = 400


class Section(BaseModel):
    item: str  # "1A", "7", "8"; "cover" for the text before Item 1
    title: str
    start: int  # character offsets in the full text
    end: int
    text: str


def find_section_starts(text: str) -> list[tuple[int, str]]:
    """(offset, item) of each accepted section heading, in document order."""
    headings = [(m.start(), m.group(1).upper()) for m in HEADING.finditer(text)]
    if not headings:
        return []

    # Rule 2: the body starts at the first "Item 1" that isn't a table-of-contents line
    body_start = 0
    for i, (offset, item) in enumerate(headings):
        is_last = i + 1 == len(headings)
        if item == "1" and (is_last or headings[i + 1][0] - offset >= TOC_GAP_CHARS):
            body_start = i
            break

    # Rule 3: walk forward; a heading starts a section only if it comes later in the outline.
    # A repeated "Item 8" while already in Item 8 is a running page header.
    starts: list[tuple[int, str]] = []
    current = -1
    for offset, item in headings[body_start:]:
        position = ITEM_ORDER.index(item)
        if position > current:
            starts.append((offset, item))
            current = position
    return starts


def split_sections(text: str) -> list[Section]:
    """Split 10-K plain text into Item sections in document order.

    Text before the first section (cover page, table of contents) becomes item "cover". If no
    headings are found, the whole text is one "cover" section, so nothing is ever lost.
    """
    starts = find_section_starts(text)
    first = starts[0][0] if starts else len(text)
    sections = [Section(item="cover", title="Cover page", start=0, end=first, text=text[:first])]
    for i, (offset, item) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        sections.append(
            Section(
                item=item, title=ITEM_TITLES[item], start=offset, end=end, text=text[offset:end]
            )
        )
    return [s for s in sections if s.text.strip()]
