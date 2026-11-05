"""The FilingLens agent: a LangGraph state machine around the three tools.

    START -> guard -> plan <-> tools (at most MAX_ROUNDS rounds) -> answer -> check -> END
               |        \\-> not_found (no tool returned data) -> END        |
               \\-> refuse (advice / off topic) -> END       answer <- once -/ (problems found)

The LLM decides which tools to call and writes the answer; code decides the flow, the limits and
the exits (ADR-003). Refusals and "not found" are fixed texts, not generated.

Try it: uv run python -m filinglens.agent.graph "What is AMD's FY2022 quick ratio?"
"""

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
from filinglens.agent.check import check_answer
from filinglens.agent.sources import Source, cited, collect_sources, render_sources

MAX_ROUNDS = 4  # rounds of tool calls; one round can hold several parallel calls
MAX_ANSWER_ATTEMPTS = 2  # the first answer plus one rewrite if the check finds problems

Verdict = Literal["ok", "advice", "off_topic"]
Outcome = Literal["answered", "not_found", "refused"]


class GuardVerdict(BaseModel):
    verdict: Verdict


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
    attempts: int  # answer attempts so far
    problems: list[str]  # what the check found in the latest answer
    verified: bool


class AgentResult(BaseModel):
    question: str
    answer: str
    outcome: Outcome
    verdict: Verdict
    rounds: int
    tool_calls: list[dict]  # [{"name": ..., "args": ...}] in call order
    sources: list[Source]  # only the ones the answer cites
    verified: bool  # passed the grounding check (refusals and "not found" pass trivially)
    attempts: int
    problems: list[str]  # left after the last attempt, if not verified


def format_filings(filings: dict[str, list[int]]) -> str:
    return "\n".join(f"{ticker}: {', '.join(map(str, years))}" for ticker, years in filings.items())


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
        prompt = [
            SystemMessage(prompts.ANSWER_SYSTEM),
            HumanMessage(
                prompts.ANSWER_USER.format(
                    sources=render_sources(sources), question=state["question"]
                )
            ),
        ]
        if state.get("problems"):  # a rewrite: show the previous answer and what was wrong
            prompt += [
                AIMessage(state["answer"]),
                HumanMessage(
                    prompts.ANSWER_RETRY.format(
                        problems="\n".join(f"- {p}" for p in state["problems"])
                    )
                ),
            ]
        text = llm.invoke(prompt).text.strip()
        outcome = "not_found" if text.startswith(prompts.NOT_FOUND) else "answered"
        return {
            "answer": text,
            "sources": sources,
            "outcome": outcome,
            "attempts": state.get("attempts", 0) + 1,
        }

    def check(state: AgentState) -> dict:
        if state["outcome"] != "answered":
            return {"verified": True, "problems": []}
        result = check_answer(state["answer"], state["sources"], state["question"])
        return {"verified": result.ok, "problems": result.problems(len(state["sources"]))}

    def not_found(state: AgentState) -> dict:
        return {
            "answer": prompts.NOT_FOUND,
            "sources": [],
            "outcome": "not_found",
            "verified": True,
        }

    def refuse(state: AgentState) -> dict:
        text = prompts.REFUSE_ADVICE if state["verdict"] == "advice" else prompts.REFUSE_OFF_TOPIC
        return {"answer": text, "sources": [], "outcome": "refused", "verified": True}

    def after_guard(state: AgentState) -> str:
        return "plan" if state["verdict"] == "ok" else "refuse"

    def after_plan(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["rounds"] < max_rounds:
            return "tools"
        # Done, or out of rounds (pending calls are dropped): answer from what we have, if any.
        return "answer" if collect_sources(state["messages"]) else "not_found"

    def after_check(state: AgentState) -> str:
        if state["verified"] or state["attempts"] >= MAX_ANSWER_ATTEMPTS:
            return END  # still failing after the rewrite: returned with verified=False
        return "answer"

    graph = StateGraph(AgentState)
    graph.add_node("guard", guard)
    graph.add_node("plan", plan)
    graph.add_node("tools", ToolNode(tools))  # runs a round's calls in parallel; errors -> messages
    graph.add_node("answer", answer)
    graph.add_node("check", check)
    graph.add_node("not_found", not_found)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", after_guard, ["plan", "refuse"])
    graph.add_conditional_edges("plan", after_plan, ["tools", "answer", "not_found"])
    graph.add_edge("tools", "plan")
    graph.add_edge("answer", "check")
    graph.add_conditional_edges("check", after_check, ["answer", END])
    graph.add_edge("not_found", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def initial_state(question: str) -> AgentState:
    return {"question": question, "messages": [HumanMessage(question)], "rounds": 0}


def ask(agent: CompiledStateGraph, question: str, config: dict | None = None) -> AgentResult:
    """config: LangGraph run config, e.g. {"callbacks": [...]} to count tokens in the eval."""
    return to_result(agent.invoke(initial_state(question), config), question)


def to_result(state: AgentState, question: str) -> AgentResult:
    """The final graph state as an AgentResult (shared by ask() and the API's stream)."""
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
        verified=state["verified"],
        attempts=state.get("attempts", 0),
        problems=state.get("problems", []),
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
    checked = "verified" if result.verified else "NOT verified: " + "; ".join(result.problems)
    print(f"\n{result.outcome} · {result.rounds} rounds · {result.attempts} attempt(s) · {checked}")
    print(f"\n{result.answer}\n")
    for s in result.sources:
        print(f"[{s.n}] {s.label}  {s.url or ''}")
