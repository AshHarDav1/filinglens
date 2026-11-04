"""The agent graph with a scripted fake LLM: every path (answer, refuse, not found, round limit)
runs end to end with no API calls and no database."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from filinglens.agent import prompts, tools
from filinglens.agent.graph import ask, build_agent
from filinglens.agent.sources import Source, cited, collect_sources
from filinglens.agent.tools import (
    CalculateOutput,
    FinancialValue,
    GetFinancialsOutput,
    make_tools,
)
from tests.fakes import FakeGuard, FakeLLM, call

FILINGS = {"AMD": [2022], "BA": [2022]}
AMD_URL = "https://www.sec.gov/Archives/edgar/data/2488/000000248823000047/"


AMD_VALUES = {"cash": 4835, "short_term_investments": 1020, "receivables": 4126,
              "current_liabilities": 6369}  # fmt: skip


@pytest.fixture
def amd_financials(monkeypatch):
    """get_financials answers from AMD's real FY2022 values; ticker checks use FILINGS."""

    def fake(session, ticker, fiscal_years, metrics):
        return GetFinancialsOutput(
            values=[
                FinancialValue(
                    ticker="AMD",
                    fiscal_year=2022,
                    metric=m,
                    value_musd=AMD_VALUES[m],
                    concept="X",
                    period_end=date(2022, 12, 31),
                    as_reported=True,
                    source_url=AMD_URL,
                )
                for m in metrics
            ],
            missing=[],
        )

    monkeypatch.setattr(tools, "get_financials", fake)
    monkeypatch.setattr(tools, "available_filings", lambda session: FILINGS)


def agent_with(llm: FakeLLM, verdict: str = "ok"):
    return build_agent(llm, FakeGuard(verdict), make_tools(None, None), FILINGS)


def test_amd_quick_ratio_two_rounds_then_cited_answer(amd_financials):
    llm = FakeLLM(
        [
            AIMessage("", tool_calls=[call("get_financials", {
                "ticker": "AMD", "fiscal_years": [2022], "metrics": list(AMD_VALUES)})]),
            AIMessage("", tool_calls=[call("calculate", {
                "expression": "(4835 + 1020 + 4126) / 6369"}, id="c2")]),
            AIMessage("Have the four values and the ratio."),
            AIMessage("AMD's FY2022 quick ratio is 1.57 [5], from cash of 4,835 [1] ..."),
        ]
    )  # fmt: skip
    result = ask(agent_with(llm), "What is AMD's FY2022 quick ratio?")

    assert result.outcome == "answered"
    assert result.rounds == 2
    assert [c["name"] for c in result.tool_calls] == ["get_financials", "calculate"]
    assert [s.n for s in result.sources] == [1, 5]  # only the cited sources come back
    assert result.sources[0].url == AMD_URL
    # The answer step saw numbered sources, not the raw conversation
    answer_prompt = llm.prompts[-1][1].content
    assert "[1] XBRL: AMD FY2022 cash = 4,835 USD millions" in answer_prompt
    assert "[5] Calculation: (4835 + 1020 + 4126) / 6369 = 1.56712" in answer_prompt
    # The planner saw the system prompt with the filings and the tool results
    assert "AMD: 2022" in llm.prompts[0][0].content
    assert isinstance(llm.prompts[1][-1], ToolMessage)


def test_advice_is_refused_without_calling_the_llm():
    llm = FakeLLM([])  # any LLM call fails the test
    result = ask(agent_with(llm, verdict="advice"), "Should I buy AMD stock?")
    assert result.outcome == "refused"
    assert result.answer == prompts.REFUSE_ADVICE
    assert result.tool_calls == []


def test_off_topic_gets_its_own_refusal():
    result = ask(agent_with(FakeLLM([]), verdict="off_topic"), "Write a poem about the sea.")
    assert result.answer == prompts.REFUSE_OFF_TOPIC


def test_unknown_company_ends_in_not_found(amd_financials):
    llm = FakeLLM(
        [
            AIMessage("", tool_calls=[call("search_filings", {
                "query": "revenue", "ticker": "TSLA", "fiscal_year": 2022})]),
            AIMessage("TSLA is not covered."),
        ]
    )  # fmt: skip
    result = ask(agent_with(llm), "What was Tesla's FY2022 revenue?")
    assert result.outcome == "not_found"
    assert result.answer == prompts.NOT_FOUND
    # The error the planner saw on round 2 names the covered companies
    assert "FilingLens covers only: AMD, BA" in llm.prompts[1][-1].content


def test_round_limit_stops_the_loop():
    endless = [
        AIMessage("", tool_calls=[call("calculate", {"expression": f"{i} + 1"}, id=f"c{i}")])
        for i in range(10)
    ]
    llm = FakeLLM([*endless[:5], AIMessage("Answer [1]")])
    result = ask(build_agent(llm, FakeGuard("ok"), make_tools(None, None), FILINGS, 4), "Loop?")
    assert len(result.tool_calls) == 4  # the 5th request was dropped, not run
    assert result.rounds == 4
    assert result.outcome == "answered"
    assert len(llm.prompts) == 6  # 5 plan calls + 1 answer call: the 5th request was dropped
    answer_prompt = llm.prompts[-1][1].content
    assert "3 + 1 = 4" in answer_prompt
    assert "4 + 1" not in answer_prompt


def test_parallel_calls_in_one_round(amd_financials):
    llm = FakeLLM(
        [
            AIMessage("", tool_calls=[
                call("calculate", {"expression": "1 / 4"}, id="a"),
                call("calculate", {"expression": "2 ** 3"}, id="b"),  # rejected
            ]),
            AIMessage("Done."),
            AIMessage("0.25 [1]"),
        ]
    )  # fmt: skip
    result = ask(agent_with(llm), "Two calculations")
    assert result.rounds == 1
    tool_messages = [m for m in llm.prompts[1] if isinstance(m, ToolMessage)]
    assert [m.status for m in tool_messages] == ["success", "error"]
    assert [s.text for s in result.sources] == ["1 / 4 = 0.25"]


def test_answer_saying_not_found_is_marked_not_found():
    llm = FakeLLM(
        [
            AIMessage("", tool_calls=[call("calculate", {"expression": "1 + 1"})]),
            AIMessage("Done."),
            AIMessage("Not found in the filings."),
        ]
    )
    result = ask(agent_with(llm), "Something the sources don't cover")
    assert result.outcome == "not_found"


def test_collect_sources_deduplicates_and_skips_failed_calls():
    calc = CalculateOutput(expression="1 + 1", value=2)
    messages = [
        ToolMessage("1 + 1 = 2", tool_call_id="a", artifact=calc),
        ToolMessage("1 + 1 = 2", tool_call_id="b", artifact=calc),
        ToolMessage("Error", tool_call_id="c", status="error"),
    ]
    assert [s.text for s in collect_sources(messages)] == ["1 + 1 = 2"]


def test_cited_keeps_only_cited_sources():
    sources = [
        Source(n=i, kind="calculation", label="calculate", url=None, text="") for i in (1, 2, 3)
    ]
    assert [s.n for s in cited("x [3] and [1], not 2", sources)] == [1, 3]


# --- the grounding check and the one rewrite ----------------------------------------------------

AMD_ROUNDS = [
    AIMessage("", tool_calls=[call("get_financials", {
        "ticker": "AMD", "fiscal_years": [2022], "metrics": list(AMD_VALUES)})]),
    AIMessage("", tool_calls=[call("calculate", {
        "expression": "(4835 + 1020 + 4126) / 6369"}, id="c2")]),
    AIMessage("Done."),
]  # fmt: skip


def test_unsupported_number_gets_one_rewrite(amd_financials):
    llm = FakeLLM([*AMD_ROUNDS, AIMessage("The quick ratio is 1.62 [5]."),
                   AIMessage("The quick ratio is 1.57 [5].")])  # fmt: skip
    result = ask(agent_with(llm), "What is AMD's FY2022 quick ratio?")
    assert result.verified
    assert result.attempts == 2
    assert result.answer == "The quick ratio is 1.57 [5]."
    rewrite_prompt = llm.prompts[-1]
    assert rewrite_prompt[2].content == "The quick ratio is 1.62 [5]."  # its previous answer
    assert "The number 1.62 does not appear in any source" in rewrite_prompt[3].content


def test_still_wrong_after_the_rewrite_is_returned_unverified(amd_financials):
    llm = FakeLLM([*AMD_ROUNDS, AIMessage("It is 1.62 [5]."), AIMessage("It is 1.62 [9].")])
    result = ask(agent_with(llm), "What is AMD's FY2022 quick ratio?")
    assert not result.verified
    assert result.attempts == 2  # no third attempt
    assert result.answer == "It is 1.62 [9]."
    assert len(result.problems) == 2  # the number and the citation [9]
