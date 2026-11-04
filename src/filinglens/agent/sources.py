"""Tool results -> numbered sources: what the answer cites as [n] and the check verifies against."""

import re
from typing import Literal

from langchain_core.messages import AnyMessage, ToolMessage
from pydantic import BaseModel

from filinglens.agent.tools import (
    CalculateOutput,
    GetFinancialsOutput,
    SearchFilingsOutput,
    format_musd,
)


class Source(BaseModel):
    """One citable piece of evidence; the answer cites it as [n]."""

    n: int
    kind: Literal["xbrl", "calculation", "passage"]
    label: str  # "AMD 10-K FY2022 · Item 7", "XBRL · AMD FY2022 cash", "calculate"
    url: str | None
    text: str  # what the answer step reads


def collect_sources(messages: list[AnyMessage]) -> list[Source]:
    """Every successful tool result as numbered sources, deduplicated, in the order found.

    Reads the artifacts (typed results), not the text the LLM saw. Failed calls have no artifact.
    """
    sources: list[Source] = []
    seen: set = set()

    def add(key, kind, label, url, text):
        if key not in seen:
            seen.add(key)
            sources.append(Source(n=len(sources) + 1, kind=kind, label=label, url=url, text=text))

    for message in messages:
        if not isinstance(message, ToolMessage) or message.artifact is None:
            continue
        result = message.artifact
        if isinstance(result, GetFinancialsOutput):
            for v in result.values:
                note = "" if v.as_reported else ", from a later 10-K (may be restated)"
                add(
                    ("xbrl", v.ticker, v.fiscal_year, v.metric),
                    "xbrl",
                    f"XBRL · {v.ticker} FY{v.fiscal_year} {v.metric}",
                    v.source_url,
                    f"{v.ticker} FY{v.fiscal_year} {v.metric} = {format_musd(v.value_musd)} "
                    f"USD millions ({v.concept}, period end {v.period_end}{note})",
                )
        elif isinstance(result, CalculateOutput):
            add(
                ("calc", result.expression),
                "calculation",
                "calculate",
                None,
                f"{result.expression} = {result.value:.6g}",
            )
        elif isinstance(result, SearchFilingsOutput):
            for p in result.passages:
                label = f"{p.ticker} {p.form} FY{p.fiscal_year}"
                label += f" · Item {p.section}" if p.section else ""
                add(("chunk", p.chunk_id), "passage", label, p.url, p.content)
    return sources


def render_sources(sources: list[Source]) -> str:
    def header(s: Source) -> str:
        if s.kind == "passage":
            return f"[{s.n}] Filing passage · {s.label}\n"
        return f"[{s.n}] {'XBRL' if s.kind == 'xbrl' else 'Calculation'}: "

    return "\n\n".join(header(s) + s.text for s in sources)


def cited(answer: str, sources: list[Source]) -> list[Source]:
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [s for s in sources if s.n in numbers]
