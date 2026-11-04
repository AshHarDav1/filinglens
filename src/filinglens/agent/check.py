"""Grounding check on a finished answer: plain code, no LLM.

1. Every number in the answer must appear in a source (an XBRL value, a calculate result, a
   passage) or in the question, allowing rounding (1.57 for 1.56712) and unit changes (4.8
   billion for 4,835 million; 16.5% for 0.165).
2. Every citation [n] must point to an existing source, and an answer with numbers must cite.

It checks where a number came from, not whether it was used correctly: dividing the wrong two
numbers passes. The LLM judge still grades correctness.
"""

import re

from pydantic import BaseModel

from filinglens.agent.sources import Source

# 4,835 / 4,835.5 / 1.56712 / 36.8 — not part of a word (FY2022, Q4), a citation ([3]) or a
# form name (10-K).
NUMBER = re.compile(r"(?<![\w.\[])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d\]]|-[KQ])")
# "Item 7", "Item 1A": section names, not data. Markdown list markers ("1. ") likewise.
IGNORED = re.compile(r"\bItems?\s+\d+[A-C]?\b|^\s*\d+\.\s", re.MULTILINE)
CITATION = re.compile(r"\[(\d+)\]")

# A source value times one of these can appear in the answer: same unit, thousands, millions
# <-> billions or units, and ratio <-> percent.
SCALES = (1, 1_000, 0.001, 1_000_000, 0.000_001, 100, 0.01)


class Number(BaseModel):
    text: str
    value: float
    decimals: int


class CheckResult(BaseModel):
    unsupported_numbers: list[str]
    bad_citations: list[int]
    missing_citations: bool

    @property
    def ok(self) -> bool:
        return not (self.unsupported_numbers or self.bad_citations or self.missing_citations)

    def problems(self, source_count: int) -> list[str]:
        problems = [
            f"The number {n} does not appear in any source (not even rounded or in other units)."
            for n in self.unsupported_numbers
        ]
        problems += [
            f"Citation [{n}] does not exist; the sources are [1] to [{source_count}]."
            for n in self.bad_citations
        ]
        if self.missing_citations:
            problems.append("The answer contains numbers but cites no sources.")
        return problems


def numbers_in(text: str) -> list[Number]:
    found = []
    for match in NUMBER.finditer(IGNORED.sub(" ", text)):
        token = match.group(1)
        decimals = len(token.split(".")[1]) if "." in token else 0
        found.append(Number(text=token, value=float(token.replace(",", "")), decimals=decimals))
    return found


def is_data(number: Number) -> bool:
    """Small counts ("3 segments") and years (2022) are not checked."""
    if number.decimals or "," in number.text:
        return True
    return not (number.value < 10 or 1900 <= number.value <= 2100)


def supported(number: Number, pool: list[float]) -> bool:
    """True if `number` is some pool value, scaled, then rounded to the answer's precision.

    Signs are ignored: "decreased by 2,544" is the calculation result -2544.
    """
    tolerance = 0.5 * 10**-number.decimals + 1e-9
    target = abs(number.value)
    return any(abs(target - abs(value) * scale) <= tolerance for value in pool for scale in SCALES)


def check_answer(answer: str, sources: list[Source], question: str) -> CheckResult:
    pool = [n.value for s in sources for n in numbers_in(s.text)]
    pool += [n.value for n in numbers_in(question)]
    claimed = [n for n in numbers_in(CITATION.sub(" ", answer)) if is_data(n)]
    unsupported = list(dict.fromkeys(n.text for n in claimed if not supported(n, pool)))

    citations = [int(n) for n in CITATION.findall(answer)]
    valid = {s.n for s in sources}
    bad = sorted({n for n in citations if n not in valid})
    return CheckResult(
        unsupported_numbers=unsupported,
        bad_citations=bad,
        missing_citations=bool(claimed) and not citations,
    )
