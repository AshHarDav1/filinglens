"""Section routing: guess which 10-K Items hold the answer to a question.

In the M2 prototype, searching only the right Item lifted hit@6 from 23/51 to 36/51 (a ceiling:
the search was told the Item). These routers guess it from the question.
"""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from filinglens.agent import prompts
from filinglens.ingestion.parse_10k import ITEM_ORDER

VALID_SECTIONS = set(ITEM_ORDER) | {"cover"}

# (pattern, Items). Checked in order; every match adds its Items.
RULES: list[tuple[str, list[str]]] = [
    (r"\brisk", ["1A", "7A"]),
    (r"\blegal|lawsuit|litigation|court|settlement|investigation", ["1A", "3", "8"]),
    (r"\bsegment|geograph|region|organic", ["7", "8"]),
    (r"\bproduct|customer|competitor|employee|business model|operate", ["1", "7"]),
    (r"\bregistered|exchange|ticker|shares outstanding", ["cover", "5"]),
    (
        r"revenue|sales|margin|income|profit|ebitda|cash|debt|liabilit|asset|ratio|expense|cost|"
        r"cogs|capex|capital expenditure|payable|receivable|inventor|dividend|tax|eps|earnings|"
        r"depreciation|amortization|liquidity|balance sheet|statement",
        ["7", "8"],
    ),
]


def rule_sections(question: str) -> list[str] | None:
    """Items picked by keyword rules, or None (search everything) if no rule matches."""
    text = question.lower()
    picked: list[str] = []
    for pattern, items in RULES:
        if re.search(pattern, text):
            picked += [i for i in items if i not in picked]
    return picked or None


class SectionChoice(BaseModel):
    items: list[str] = Field(description="1 to 3 Item numbers, most likely first, e.g. ['8', '7']")


def llm_sections(question: str, router: BaseChatModel) -> list[str] | None:
    """Items picked by a small LLM, or None if it returns nothing valid."""
    choice = router.with_structured_output(SectionChoice).invoke(
        [SystemMessage(prompts.ROUTER_SYSTEM), HumanMessage(question)]
    )
    items = [i.strip().upper() for i in choice.items] if choice else []
    items = [("cover" if i == "COVER" else i) for i in items]
    picked = [i for i in dict.fromkeys(items) if i in VALID_SECTIONS]
    return picked or None
