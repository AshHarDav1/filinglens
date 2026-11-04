"""The FilingLens agent: a LangGraph state machine around the three tools.

    START -> guard -> plan <-> tools (at most MAX_ROUNDS rounds) -> answer -> END
               |        \\-> not_found (no tool returned data) -> END
               \\-> refuse (advice / off topic) -> END

The LLM decides which tools to call and writes the answer; code decides the flow, the limits and
the exits (ADR-003). Refusals and "not found" are fixed texts, not generated.

Try it: uv run python -m filinglens.agent.graph "What is AMD's FY2022 quick ratio?"
"""

import re
import sys
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from filinglens.agent import prompts
from filinglens.agent.tools import (
    CalculateOutput,
    GetFinancialsOutput,
    SearchFilingsOutput,
    format_musd,
)

MAX_ROUNDS = 4  # rounds of tool calls; one round can hold several parallel calls

Verdict = Literal["ok", "advice", "off_topic"]
Outcome = Literal["answered", "not_found", "refused"]


class GuardVerdict(BaseModel):
    verdict: Verdict


class Source(BaseModel):
    """One citable piece of evidence; the answer cites it as [n]."""

    n: int
    kind: Literal["xbrl", "calculation", "passage"]
    label: str  # "AMD 10-K FY2022 · Item 7", "XBRL · AMD FY2022 cash", "calculate"
    url: str | None
    text: str  # what the answer step reads


class AgentState(TypedDict, total=False):
    question: str
    # add_messages is a reducer: a node returns only its new messages and LangGraph appends them
    # (plain keys like `rounds` are simply overwritten).
    messages: Annotated[list[AnyMessage], add_messages]
    verdict: Verdict
    rounds: int
    answer: str
    sources: list[Source]
    outcome: Outcome


class AgentResult(BaseModel):
    question: str
    answer: str
    outcome: Outcome
    verdict: Verdict
    rounds: int
    tool_calls: list[dict]  # [{"name": ..., "args": ...}] in call order
    sources: list[Source]  # only the ones the answer cites


# --- sources: tool results -> numbered evidence -------------------------------------------------


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


def format_filings(filings: dict[str, list[int]]) -> str:
    return "\n".join(f"{ticker}: {', '.join(map(str, years))}" for ticker, years in filings.items())


# --- the graph ----------------------------------------------------------------------------------


def build_agent(
    llm: BaseChatModel,
    guard_llm: BaseChatModel,
    tools: list[BaseTool],
    filings: dict[str, list[int]],
    max_rounds: int = MAX_ROUNDS,
) -> CompiledStateGraph:
    """llm plans and answers (gpt-6-luna in our runs); guard_llm screens questions (Haiku)."""
    planner = llm.bind_tools(tools)
    screener = guard_llm.with_structured_output(GuardVerdict)
    plan_system = prompts.PLAN_SYSTEM.format(filings=format_filings(filings), max_rounds=max_rounds)

    def guard(state: AgentState) -> dict:
        result = screener.invoke(
            [SystemMessage(prompts.GUARD_SYSTEM), HumanMessage(state["question"])]
        )
        return {"verdict": result.verdict}

    def plan(state: AgentState) -> dict:
        # The system prompt is added per call, not stored: state keeps only the conversation.
        reply = planner.invoke([SystemMessage(plan_system), *state["messages"]])
        # Rounds already run: every earlier tool-call message went through the tools node.
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        return {"messages": [reply], "rounds": rounds}

    def answer(state: AgentState) -> dict:
        sources = collect_sources(state["messages"])
        reply = llm.invoke(
            [
                SystemMessage(prompts.ANSWER_SYSTEM),
                HumanMessage(
                    prompts.ANSWER_USER.format(
                        sources=render_sources(sources), question=state["question"]
                    )
                ),
            ]
        )
        text = reply.text.strip()
        outcome = "not_found" if text.startswith(prompts.NOT_FOUND) else "answered"
        return {"answer": text, "sources": sources, "outcome": outcome}

    def not_found(state: AgentState) -> dict:
        return {"answer": prompts.NOT_FOUND, "sources": [], "outcome": "not_found"}

    def refuse(state: AgentState) -> dict:
        text = prompts.REFUSE_ADVICE if state["verdict"] == "advice" else prompts.REFUSE_OFF_TOPIC
        return {"answer": text, "sources": [], "outcome": "refused"}

    def after_guard(state: AgentState) -> str:
        return "plan" if state["verdict"] == "ok" else "refuse"

    def after_plan(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["rounds"] < max_rounds:
            return "tools"
        # Done, or out of rounds (pending calls are dropped): answer from what we have, if any.
        return "answer" if collect_sources(state["messages"]) else "not_found"

    graph = StateGraph(AgentState)
    graph.add_node("guard", guard)
    graph.add_node("plan", plan)
    graph.add_node("tools", ToolNode(tools))  # runs a round's calls in parallel; errors -> messages
    graph.add_node("answer", answer)
    graph.add_node("not_found", not_found)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", after_guard, ["plan", "refuse"])
    graph.add_conditional_edges("plan", after_plan, ["tools", "answer", "not_found"])
    graph.add_edge("tools", "plan")
    for node in ("answer", "not_found", "refuse"):
        graph.add_edge(node, END)
    return graph.compile()


def ask(agent: CompiledStateGraph, question: str) -> AgentResult:
    state = agent.invoke({"question": question, "messages": [HumanMessage(question)], "rounds": 0})
    ran = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
    calls = [
        {"name": c["name"], "args": c["args"]}
        for m in state["messages"]
        if isinstance(m, AIMessage)
        for c in m.tool_calls
        if c["id"] in ran  # calls dropped at the round limit never ran
    ]
    return AgentResult(
        question=question,
        answer=state["answer"],
        outcome=state["outcome"],
        verdict=state["verdict"],
        rounds=state.get("rounds", 0),
        tool_calls=calls,
        sources=cited(state["answer"], state.get("sources", [])),
    )


def default_agent(session) -> CompiledStateGraph:
    """The agent wired to the models in .env, our database and the embedding model."""
    from filinglens.agent.tools import available_filings, make_tools
    from filinglens.llm import get_chat_model, get_embeddings, get_router_model

    tools = make_tools(session, get_embeddings())
    return build_agent(get_chat_model(), get_router_model(), tools, available_filings(session))


if __name__ == "__main__":
    from filinglens.db.session import get_session

    with get_session() as session:
        result = ask(default_agent(session), " ".join(sys.argv[1:]))
    for call in result.tool_calls:
        print(f"tool  {call['name']}({call['args']})")
    print(f"\n{result.outcome} · {result.rounds} rounds\n\n{result.answer}\n")
    for s in result.sources:
        print(f"[{s.n}] {s.label}  {s.url or ''}")
