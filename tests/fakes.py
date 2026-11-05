"""Scripted stand-ins for the chat models, shared by the graph and API tests."""

from langchain_core.messages import AIMessage

from filinglens.agent.graph import GuardVerdict


class FakeLLM:
    """Plays back scripted replies in order and records every prompt it was sent.

    Stands in for both the planner (bind_tools) and the answerer (plain invoke); the same
    object is used for both, as with the real model.
    """

    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.prompts: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        return self.replies.pop(0)


class FakeGuard:
    def __init__(self, verdict: str):
        self.verdict = verdict

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return GuardVerdict(verdict=self.verdict)


def call(name: str, args: dict, id: str = "c1") -> dict:
    return {"name": name, "args": args, "id": id, "type": "tool_call"}
