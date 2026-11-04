"""Retrieval hit rate: for each question, is the FinanceBench evidence passage in the top-k chunks?

No LLM is involved (except the optional LLM section router), only the embedding of each
question, so this is fast and almost free.

    uv run python eval/retrieval_hit.py --name hit-x --subset eval/financebench_10k.jsonl
    uv run python eval/retrieval_hit.py --name hit-x --subset eval/financebench_10k.jsonl \
        --method hybrid --router llm --section-mode boost --split dev

--split dev/test: the 51-question set minus / restricted to the 20-question set. Choose between
variants on dev (31 questions), confirm the winner once on test (20).

Results go to eval/results/<date>-<name>.json. Existing results are never overwritten.
"""

import argparse
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

from filinglens.config import get_settings
from filinglens.db.session import get_session
from filinglens.ingestion.chunk_embed import CHUNK_OVERLAP, CHUNK_SIZE
from filinglens.llm import get_embeddings, get_router_model
from filinglens.retrieval.hybrid import retrieve
from filinglens.retrieval.routing import llm_sections, rule_sections
from filinglens.retrieval.vector import RetrievedChunk

RESULTS_DIR = Path(__file__).parent / "results"
TEST_SET = Path(__file__).parent / "financebench_subset.jsonl"
KS = (6, 10, 20, 40)
# A window long enough to identify a passage, short enough to fit inside one chunk's overlap
PREFIX_CHARS = 150
WINDOW_STEP = 50


def normalize(text: str) -> str:
    """Lowercase words and numbers only: FinanceBench's PDF text and our HTML text differ in
    whitespace, quotes (“ vs ") and table dashes (— is dropped in the PDF)."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def evidence_windows(evidence: str) -> list[str]:
    """PREFIX_CHARS-long windows of the evidence, every WINDOW_STEP chars.

    A long passage spans several chunks, and its first characters can fall on a section border
    (page-header lines like "Table of Contents PART II Item 8"), so any window may prove the hit.
    """
    text = normalize(evidence)
    if len(text) <= PREFIX_CHARS:
        return [text]
    return [text[i : i + PREFIX_CHARS] for i in range(0, len(text) - PREFIX_CHARS + 1, WINDOW_STEP)]


def evidence_hit(evidence: str, chunks: list[RetrievedChunk]) -> bool:
    """True if any of `chunks` contains a window of the `evidence` passage.

    This is a lower bound: a chunk that states the same fact elsewhere in the filing doesn't count.
    """
    windows = evidence_windows(evidence)
    return any(w in normalize(chunk.content) for chunk in chunks for w in windows)


def first_hit_rank(evidence: list[str], hits: list[RetrievedChunk]) -> int | None:
    """1-based rank of the first chunk that contains any evidence passage, or None."""
    for rank, chunk in enumerate(hits, start=1):
        if any(evidence_hit(passage, [chunk]) for passage in evidence):
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval hit rate on the eval subset")
    parser.add_argument("--name", required=True)
    parser.add_argument("--subset", type=Path, default=Path("eval/financebench_subset.jsonl"))
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--method", choices=["vector", "keyword", "hybrid"], default="vector")
    parser.add_argument(
        "--router",
        choices=["none", "rules", "llm", "oracle"],
        default="none",
        help="oracle: the Items that hold the evidence (a ceiling, not a router)",
    )
    parser.add_argument("--section-mode", choices=["filter", "boost"], default="filter")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    args = parser.parse_args()

    out = RESULTS_DIR / f"{date.today().isoformat()}-{args.name}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; choose another --name")

    items = [json.loads(line) for line in args.subset.read_text(encoding="utf-8").splitlines()]
    if args.split != "all":
        test_ids = {json.loads(line)["id"] for line in TEST_SET.read_text().splitlines()}
        items = [i for i in items if (i["id"] in test_ids) == (args.split == "test")]
    items = items[: args.limit] if args.limit else items

    settings = get_settings()
    embedder = get_embeddings(settings)
    vectors = embedder.embed_documents([item["question"] for item in items])
    router = get_router_model(settings) if args.router == "llm" else None

    results = []
    with get_session() as session:
        for i, (item, vector) in enumerate(zip(items, vectors, strict=True), start=1):
            ticker, year = item["ticker"], item["fiscal_year"]
            if args.router == "rules":
                sections = rule_sections(item["question"])
            elif args.router == "llm":
                sections = llm_sections(item["question"], router)
            elif args.router == "oracle":
                everything = retrieve(session, item["question"], vector, ticker, year, k=10_000)
                sections = sorted(
                    {c.section for c in everything if first_hit_rank(item["evidence"], [c])}
                )
            else:
                sections = None
            hits = retrieve(
                session, item["question"], vector, ticker, year, k=max(KS),
                method=args.method, sections=sections, section_mode=args.section_mode,
            )  # fmt: skip
            rank = first_hit_rank(item["evidence"], hits)
            shown = f"rank {rank}" if rank else f"not in top {max(KS)}"
            print(f"[{i}/{len(items)}] {ticker:<5} {item['id'][-5:]}  {shown}  sections={sections}")
            results.append({"id": item["id"], "ticker": ticker, "rank": rank, "sections": sections})

    n = len(results)
    hit_rate = {k: sum(1 for r in results if r["rank"] and r["rank"] <= k) for k in KS}
    print()
    for k, hit in hit_rate.items():
        print(f"hit@{k:<2}: {hit}/{n}")

    report = {
        "name": args.name,
        "run_at": datetime.now(UTC).isoformat(),
        "config": {
            "embedding_model": settings.embedding_model,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "ks": list(KS),
            "subset": str(args.subset),
            "split": args.split,
            "method": args.method,
            "router": args.router,
            "router_model": settings.router_model if args.router == "llm" else None,
            "section_mode": args.section_mode if args.router != "none" else None,
        },
        "summary": {"n": n, **{f"hit@{k}": f"{hit}/{n}" for k, hit in hit_rate.items()}},
        "items": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
