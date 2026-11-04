"""Run the eval subset through the system and score it with an LLM judge.

    uv run python eval/run_eval.py --mode baseline --name baseline       # closed-book LLM
    uv run python eval/run_eval.py --mode rag --name rag-poc             # plain RAG (M0 PoC)
    uv run python eval/run_eval.py --mode rag --name smoke --limit 2     # quick cheap check
    uv run python eval/run_eval.py --mode agent --name agent-m3          # LangGraph agent (M3)

Results go to eval/results/<date>-<name>.json. Existing results are never overwritten.
"""

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from filinglens.agent import prompts
from filinglens.agent.graph import MAX_ROUNDS, ask, default_agent
from filinglens.config import get_settings
from filinglens.db.session import get_session
from filinglens.ingestion.chunk_embed import CHUNK_OVERLAP, CHUNK_SIZE
from filinglens.llm import get_chat_model, get_embeddings, get_judge_model, get_router_model
from filinglens.retrieval.hybrid import retrieve
from filinglens.retrieval.routing import llm_sections, rule_sections
from filinglens.retrieval.vector import RetrievedChunk

RESULTS_DIR = Path(__file__).parent / "results"


class JudgeVerdict(BaseModel):
    reason: str = Field(description="One or two sentences comparing candidate and reference")
    correct: bool


def usage(message) -> dict:
    meta = getattr(message, "usage_metadata", None) or {}
    return {"input": meta.get("input_tokens", 0), "output": meta.get("output_tokens", 0)}


def format_excerpts(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] {c.ticker} {c.form} FY{c.fiscal_year}"
        f"{f' · Item {c.section}' if c.section else ''} (chunk {c.chunk_index})\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )


def answer_question(item: dict, mode: str, llm, embedder, session, k: int, args, router) -> dict:
    retrieved: list[RetrievedChunk] = []
    sections = None
    if mode == "baseline":
        messages = [
            SystemMessage(prompts.BASELINE_SYSTEM),
            HumanMessage(item["question"]),
        ]
    else:
        query_vector = embedder.embed_query(item["question"])
        if args.router == "rules":
            sections = rule_sections(item["question"])
        elif args.router == "llm":
            sections = llm_sections(item["question"], router)
        retrieved = retrieve(
            session, item["question"], query_vector, item["ticker"], item["fiscal_year"], k,
            method=args.method, sections=sections, section_mode=args.section_mode,
        )  # fmt: skip
        if not retrieved:
            print(f"  WARNING no chunks for {item['ticker']} FY{item['fiscal_year']} (ingested?)")
        messages = [
            SystemMessage(prompts.RAG_SYSTEM),
            HumanMessage(
                prompts.RAG_USER.format(
                    excerpts=format_excerpts(retrieved), question=item["question"]
                )
            ),
        ]
    started = time.perf_counter()
    response = llm.invoke(messages)
    return {
        "answer": response.text,
        "latency_s": round(time.perf_counter() - started, 2),
        "tokens": usage(response),
        "sections": sections,
        "sources": [
            {
                "ref": f"[{i}]",
                "label": f"{c.ticker} {c.form} FY{c.fiscal_year}"
                + (f" · Item {c.section}" if c.section else ""),
                "url": c.url,
                "chunk_id": c.chunk_id,
                "distance": round(c.distance, 4),
            }
            for i, c in enumerate(retrieved, start=1)
        ],
    }


class UsageCounter(UsageMetadataCallbackHandler):
    """Tokens per model (from LangChain) plus the number of LLM calls, across one agent run."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self.calls += 1
        super().on_llm_end(response, **kwargs)


def answer_agent(item: dict, agent) -> dict:
    counter = UsageCounter()
    started = time.perf_counter()
    try:
        result = ask(agent, item["question"], {"callbacks": [counter]})
    except Exception as e:  # one failing question must not lose the whole run
        result, error = None, f"{type(e).__name__}: {e}"
    latency = round(time.perf_counter() - started, 2)
    by_model = {
        model: {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}
        for model, u in counter.usage_metadata.items()
    }
    row = {
        "latency_s": latency,
        "tokens": {
            "input": sum(u["input"] for u in by_model.values()),
            "output": sum(u["output"] for u in by_model.values()),
        },
        "tokens_by_model": by_model,
        "llm_calls": counter.calls,
    }
    if result is None:
        return {**row, "answer": f"ERROR {error}", "error": error, "outcome": "error",
                "verified": False, "attempts": 0, "problems": [], "rounds": 0,
                "tool_calls": [], "sources": []}  # fmt: skip
    return {
        **row,
        "answer": result.answer,
        "outcome": result.outcome,
        "verified": result.verified,
        "attempts": result.attempts,
        "problems": result.problems,
        "rounds": result.rounds,
        "tool_calls": result.tool_calls,
        "sources": [
            {"ref": f"[{s.n}]", "kind": s.kind, "label": s.label, "url": s.url}
            for s in result.sources
        ],
    }


def agent_summary(rows: list[dict]) -> dict:
    by_model: dict[str, dict[str, int]] = {}
    for r in rows:
        for model, u in r["tokens_by_model"].items():
            total = by_model.setdefault(model, {"input": 0, "output": 0})
            total["input"] += u["input"]
            total["output"] += u["output"]
    answered = [r for r in rows if r["outcome"] == "answered"]
    return {
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "verified": f"{sum(r['verified'] for r in answered)}/{len(answered)} answered",
        "rewrites": sum(r["attempts"] > 1 for r in rows),
        "rounds": dict(sorted(Counter(r["rounds"] for r in rows).items())),
        "tool_calls": dict(Counter(c["name"] for r in rows for c in r["tool_calls"])),
        "llm_calls_mean": round(statistics.mean(r["llm_calls"] for r in rows), 2),
        "tokens_by_model": by_model,
        "errors": sum(r["outcome"] == "error" for r in rows),
    }


def judge(judge_llm, item: dict, candidate: str) -> tuple[JudgeVerdict, dict]:
    result = judge_llm.invoke(
        [
            SystemMessage(prompts.JUDGE_SYSTEM),
            HumanMessage(
                prompts.JUDGE_USER.format(
                    question=item["question"], reference=item["answer"], candidate=candidate
                )
            ),
        ]
    )
    if result["parsed"] is None:
        verdict = JudgeVerdict(
            reason=f"judge output unparseable: {result['parsing_error']}", correct=False
        )
    else:
        verdict = result["parsed"]
    return verdict, usage(result["raw"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FinanceBench subset eval")
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["baseline", "rag", "agent"], required=True)
    parser.add_argument("--subset", type=Path, default=Path("eval/financebench_subset.jsonl"))
    parser.add_argument("--k", type=int, default=6, help="chunks retrieved per question (rag)")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--method", choices=["vector", "keyword", "hybrid"], default="vector")
    parser.add_argument("--router", choices=["none", "rules", "llm"], default="none")
    parser.add_argument("--section-mode", choices=["filter", "boost"], default="filter")
    args = parser.parse_args()

    out = RESULTS_DIR / f"{date.today().isoformat()}-{args.name}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; choose another --name")

    items = [json.loads(line) for line in args.subset.read_text(encoding="utf-8").splitlines()]
    items = items[: args.limit] if args.limit else items

    settings = get_settings()
    llm = get_chat_model(settings)
    judge_llm = get_judge_model(settings).with_structured_output(JudgeVerdict, include_raw=True)
    embedder = get_embeddings(settings) if args.mode == "rag" else None
    session = get_session() if args.mode in ("rag", "agent") else None
    agent = default_agent(session) if args.mode == "agent" else None
    router = get_router_model(settings) if args.mode == "rag" and args.router == "llm" else None

    rows = []
    for n, item in enumerate(items, start=1):
        if args.mode == "agent":
            result = answer_agent(item, agent)
        else:
            result = answer_question(item, args.mode, llm, embedder, session, args.k, args, router)
        verdict, judge_tokens = judge(judge_llm, item, result["answer"])
        rows.append(
            {
                "id": item["id"],
                "ticker": item["ticker"],
                "fiscal_year": item["fiscal_year"],
                "question_type": item["question_type"],
                "question": item["question"],
                "reference": item["answer"],
                **result,
                "correct": verdict.correct,
                "judge_reason": verdict.reason,
                "judge_tokens": judge_tokens,
            }
        )
        mark = "OK " if verdict.correct else "ERR"
        extra = ""
        if args.mode == "agent":
            calls = ",".join(c["name"].split("_")[0] for c in result["tool_calls"])
            checked = "verified" if result["verified"] else "UNVERIFIED"
            extra = f" {result['outcome']} r{result['rounds']} {checked} [{calls}]"
        print(
            f"[{n}/{len(items)}] {mark} {item['ticker']} {item['id'][-5:]} "
            f"({result['latency_s']}s){extra}"
        )

    if session is not None:
        session.close()

    latencies = [r["latency_s"] for r in rows]
    correct = sum(r["correct"] for r in rows)
    summary = {
        "correct": correct,
        "total": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "latency_s": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "max": max(latencies),
        },
        "answer_tokens": {
            "input": sum(r["tokens"]["input"] for r in rows),
            "output": sum(r["tokens"]["output"] for r in rows),
        },
        "judge_tokens": {
            "input": sum(r["judge_tokens"]["input"] for r in rows),
            "output": sum(r["judge_tokens"]["output"] for r in rows),
        },
    }
    if args.mode == "agent":
        summary["agent"] = agent_summary(rows)
    report = {
        "name": args.name,
        "mode": args.mode,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "judge_provider": settings.judge_provider,
            "judge_model": settings.judge_model,
            "judge_temperature": settings.judge_temperature,
            "embedding_model": (
                settings.embedding_model if args.mode in ("rag", "agent") else None
            ),
            "guard_model": settings.router_model if args.mode == "agent" else None,
            "max_rounds": MAX_ROUNDS if args.mode == "agent" else None,
            "k": args.k if args.mode == "rag" else None,
            "retrieval_method": args.method if args.mode == "rag" else None,
            "router": args.router if args.mode == "rag" else None,
            "router_model": settings.router_model if args.router == "llm" else None,
            "section_mode": args.section_mode if args.router != "none" else None,
            "chunk_size_chars": CHUNK_SIZE if args.mode == "rag" else None,
            "chunk_overlap_chars": CHUNK_OVERLAP if args.mode == "rag" else None,
            "prompt_versions": prompts.PROMPT_VERSIONS,
            "subset": str(args.subset),
        },
        "summary": summary,
        "items": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{args.mode}: {correct}/{len(rows)} correct -> {out}")


if __name__ == "__main__":
    main()
