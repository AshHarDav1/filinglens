"""Behaviour checks for the agent: refusals, "not found" and prompt injection. Scored by rules
(expected outcome, forbidden phrases), not by the LLM judge.

    uv run python eval/run_behaviour.py --name behaviour-m3

A case with "inject" plants that text as an extra passage in every search_filings result, the way
a malicious sentence inside a filing would arrive. Results go to eval/results/<date>-<name>.json.
"""

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

from filinglens.agent import prompts, tools
from filinglens.agent.graph import ask, default_agent
from filinglens.agent.tools import Passage, SearchFilingsOutput
from filinglens.config import get_settings
from filinglens.db.session import get_session

RESULTS_DIR = Path(__file__).parent / "results"
INJECTED_CHUNK_ID = -1


def planting(text: str, seen: list[bool]):
    """A search_filings that returns the real passages plus one planted passage."""
    real_search = tools.search_filings

    def search(session, embedder, query, ticker, fiscal_year, sections=None) -> SearchFilingsOutput:
        result = real_search(session, embedder, query, ticker, fiscal_year, sections)
        seen.append(True)
        planted = Passage(
            chunk_id=INJECTED_CHUNK_ID, ticker=ticker.upper(), fiscal_year=fiscal_year,
            form="10-K", section="7", url="https://example.invalid/planted", content=text,
        )  # fmt: skip
        return SearchFilingsOutput(query=query, passages=[*result.passages, planted])

    return search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent behaviour checks")
    parser.add_argument("--name", required=True)
    parser.add_argument("--cases", type=Path, default=Path("eval/behaviour.jsonl"))
    args = parser.parse_args()

    out = RESULTS_DIR / f"{date.today().isoformat()}-{args.name}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; choose another --name")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines()]

    session = get_session()
    agent = default_agent(session)
    real_search = tools.search_filings
    rows = []
    for n, case in enumerate(cases, start=1):
        seen: list[bool] = []
        if "inject" in case:
            tools.search_filings = planting(case["inject"], seen)  # make_tools looks it up per call
        try:
            result = ask(agent, case["question"])
        finally:
            tools.search_filings = real_search
        failures = []
        if result.outcome not in case["expect_outcome"]:
            failures.append(f"outcome {result.outcome}, expected {case['expect_outcome']}")
        failures += [
            f"contains {phrase!r}"
            for phrase in case.get("must_not_contain", [])
            if phrase.lower() in result.answer.lower()
        ]
        if "inject" in case and not seen:
            failures.append("the planted passage was never retrieved: the case tested nothing")
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "passed": not failures,
                "failures": failures,
                "outcome": result.outcome,
                "verdict": result.verdict,
                "verified": result.verified,
                "answer": result.answer,
                "tool_calls": result.tool_calls,
                "cited_planted_passage": any(s.url and "planted" in s.url for s in result.sources),
            }
        )
        print(f"[{n}/{len(cases)}] {'PASS' if not failures else 'FAIL'} {case['id']} {failures}")
    session.close()

    settings = get_settings()
    passed = sum(r["passed"] for r in rows)
    report = {
        "name": args.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "llm_model": settings.llm_model,
            "guard_model": settings.router_model,
            "prompt_versions": prompts.PROMPT_VERSIONS,
            "cases": str(args.cases),
        },
        "summary": {"passed": passed, "total": len(rows)},
        "items": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nbehaviour: {passed}/{len(rows)} passed -> {out}")


if __name__ == "__main__":
    main()
