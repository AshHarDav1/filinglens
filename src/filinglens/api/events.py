"""Agent progress as Server-Sent Events.

The stream is: one `step` event per finished graph node, then one `result` event (the same JSON
as POST /ask), or an `error` event if something fails after the 200 status was sent. Each event
is `event: <type>\\ndata: <json>\\n\\n`.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from filinglens.agent.tools import CalculateOutput, GetFinancialsOutput, SearchFilingsOutput


def sse(event: str, data: BaseModel | dict) -> str:
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def describe_call(call: dict) -> str:
    args = call["args"]
    if call["name"] == "get_financials":
        years = ", ".join(f"FY{y}" for y in args.get("fiscal_years", []))
        return f"XBRL {args.get('ticker')} {years}: {', '.join(args.get('metrics', []))}"
    if call["name"] == "search_filings":
        items = args.get("sections") or []
        where = f" Items {', '.join(items)}" if items else ""
        return (
            f"search {args.get('ticker')} FY{args.get('fiscal_year')}{where}: {args.get('query')}"
        )
    return f"calculate {args.get('expression')}"


def describe_result(message: ToolMessage) -> str:
    result = message.artifact
    if message.status == "error" or result is None:
        return f"{message.name} failed: {str(message.content)[:120]}"
    if isinstance(result, GetFinancialsOutput):
        missing = f", {len(result.missing)} not reported" if result.missing else ""
        return f"{len(result.values)} XBRL values{missing}"
    if isinstance(result, SearchFilingsOutput):
        return f"{len(result.passages)} passages"
    if isinstance(result, CalculateOutput):
        return f"= {result.value:.6g}"
    return message.name


def step_data(node: str, update: dict, elapsed_ms: int) -> dict[str, Any]:
    """What one finished node did, with a one-line `label` a UI can show as is."""
    data: dict[str, Any] = {"node": node, "elapsed_ms": elapsed_ms}
    if node == "guard":
        verdict = update["verdict"]
        data["verdict"] = verdict
        data["label"] = {
            "ok": "Question is in scope",
            "advice": "Investment advice: refusing",
            "off_topic": "Off topic: refusing",
        }[verdict]
    elif node == "plan":
        message = update["messages"][-1]
        calls = message.tool_calls if isinstance(message, AIMessage) else []
        data["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in calls]
        data["label"] = (
            "; ".join(describe_call(c) for c in calls) if calls else "Done gathering data"
        )
    elif node == "tools":
        results = [m for m in update["messages"] if isinstance(m, ToolMessage)]
        data["results"] = [
            {"name": m.name, "status": m.status, "summary": describe_result(m)} for m in results
        ]
        data["label"] = "; ".join(r["summary"] for r in data["results"])
    elif node == "answer":
        data["attempt"] = update["attempts"]
        data["label"] = "Answer written" if update["attempts"] == 1 else "Answer rewritten"
    elif node == "check":
        data["verified"] = update["verified"]
        data["problems"] = update["problems"]
        data["label"] = (
            "Checked: every number traces to a source"
            if update["verified"]
            else f"Check found {len(update['problems'])} problem(s)"
        )
    elif node == "refuse":
        data["label"] = "Refused"
    elif node == "not_found":
        data["label"] = "Not found in the filings"
    return data
