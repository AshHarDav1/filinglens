"""Build the FinanceBench eval subset (the "exam file") from the downloaded dataset.

FinanceBench is used ONLY for evaluation. Nothing from it is loaded into the database.

    curl -L -o data/financebench/financebench_merged.jsonl \
      https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/financebench_merged.jsonl
    uv run python eval/build_subset.py                 # 20-question quick set (LLM-judged runs)
    uv run python eval/build_subset.py --all-10k       # every 10-K question, same 10 companies

Dataset: PatronusAI/financebench (CC-BY-NC-4.0).
"""

import argparse
import json
from pathlib import Path

# M0 subset: 10 companies x 2 questions, all from 10-K filings. Chosen 2026-10-22.
SUBSET_IDS = [
    "01198", "00222",  # AMD
    "00552", "04700",  # Microsoft
    "08135", "06655",  # Amazon
    "00438", "00591",  # Adobe
    "04458", "03282",  # Netflix
    "00215", "00566",  # Verizon
    "01226", "01865",  # 3M
    "01091", "00494",  # Boeing
    "04481", "01009",  # PepsiCo
    "00723", "01351",  # American Express
]  # fmt: skip

TICKERS = {
    "AMD": "AMD",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Adobe": "ADBE",
    "Netflix": "NFLX",
    "Verizon": "VZ",
    "3M": "MMM",
    "Boeing": "BA",
    "PepsiCo": "PEP",
    "American Express": "AXP",
}


def to_item(row: dict) -> dict:
    return {
        "id": row["financebench_id"],
        "company": row["company"],
        "ticker": TICKERS[row["company"]],
        "fiscal_year": int(row["doc_period"]),
        "doc_name": row["doc_name"],
        "question_type": row["question_type"],
        "question": row["question"],
        "answer": row["answer"],
        "justification": row["justification"],
        "evidence": [e["evidence_text"] for e in row["evidence"]],
    }


def build(source: Path, all_10k: bool = False) -> list[dict]:
    rows = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["financebench_id"].removeprefix("financebench_id_")] = row

    if all_10k:
        # Added 2026-10-30: 20 questions were too few to see small retrieval changes
        ids = sorted(
            short_id
            for short_id, row in rows.items()
            if row["doc_type"] == "10k" and row["company"] in TICKERS
        )
    else:
        ids = SUBSET_IDS
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"IDs not found in {source}: {missing}")

    for short_id in ids:
        if rows[short_id]["doc_type"] != "10k":
            raise SystemExit(f"{short_id} is not a 10-K question")
    return [to_item(rows[short_id]) for short_id in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=Path("data/financebench/financebench_merged.jsonl")
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="default: eval/financebench_subset.jsonl, or "
        "eval/financebench_10k.jsonl with --all-10k",
    )
    parser.add_argument(
        "--all-10k",
        action="store_true",
        help="every 10-K question for the 10 companies, not just the 20",
    )
    args = parser.parse_args()
    out = args.out or Path(
        "eval/financebench_10k.jsonl" if args.all_10k else "eval/financebench_subset.jsonl"
    )

    subset = build(args.source, all_10k=args.all_10k)
    with out.open("w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    filings = sorted({(q["ticker"], q["fiscal_year"]) for q in subset})
    print(f"Wrote {len(subset)} questions, {len(filings)} filings -> {out}")


if __name__ == "__main__":
    main()
