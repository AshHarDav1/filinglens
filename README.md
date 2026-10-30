# FilingLens

An AI analyst for SEC filings. Ask about a US public company and get **exact numbers** from the SEC's XBRL
data plus **management's explanations** from the 10-K text, with every claim linked to its source filing.

> Portfolio project for learning AI engineering by building a production-style system. Not investment advice.

**Status:** M0 (proof of concept) and M1 (data) are done. M2 (hybrid search) is next. See
[Roadmap](#roadmap).

## How it works

```
Question
   │
   ├─ numbers ──► get_metric(): SQL over SEC XBRL facts ─────────────┐   exact, as reported
   │                                                                   ├─► LLM answer with citations
   └─ context ──► vector search over 10-K chunks (one filing, top k) ─┘   (agent with tools in M3)

Ingestion:  SEC EDGAR ─► 10-K HTML ─► text (table rows intact) ─► Item sections ─► chunks ─► embeddings
            SEC companyfacts ─► financial_fact (every 10-K fact, as reported)
Storage:    PostgreSQL + pgvector: company · filing · chunk (+ section, embedding) · financial_fact
```

Design rules:

- **Numbers come from XBRL, never from the LLM reading tables** ([ADR-002](docs/adr/002-numbers-from-xbrl.md)).
  Microsoft's FY2016 income statement ranked 73rd of 203 chunks for a COGS question; the XBRL lookup returns
  $32,780M from the FY2016 10-K directly.
- **Facts are stored as reported; the right value is chosen at query time**
  ([ADR-005](docs/adr/005-xbrl-facts-as-reported.md)): the year's own 10-K first, so restatements and the
  comparison columns of later filings don't leak in.
- **Chunks follow the 10-K outline**: one Item per chunk, the Item stored for filtering
  ([ADR-006](docs/adr/006-chunk-per-item.md)).
- **Every quality change is measured** on FinanceBench questions and saved under `eval/results/`
  ([ADR-004](docs/adr/004-financebench-acceptance-test.md), [ADR-007](docs/adr/007-retrieval-hit-rate.md)).

## Results so far

All numbers come from saved runs in `eval/results/`. Denominators are the question counts.

| Measurement | Result | Run |
|---|---|---|
| Closed-book LLM (gpt-6-luna, no retrieval), judged by claude-haiku-5-5 | 11/20 | `2026-10-28-baseline-haiku-judge` |
| Plain RAG (M0: vector search, top 6) | 6/20 | `2026-10-28-rag-poc` |
| Retrieval hit@6 / hit@20 (evidence in top k), M1 chunks | 23/51 / 39/51 | `2026-10-30-hit-m1-51q` |

What the numbers say:

- **M0:** plain RAG scored below the closed-book model, and 12 of its 14 wrong answers had no evidence in the
  retrieved chunks: the search, not the LLM, was the problem ([docs/m0-results.md](docs/m0-results.md)).
- **M1:** numbers now come from XBRL and match the filings (Microsoft FY2016 COGS 32,780M, Netflix FY2017
  current liabilities 5,466M, AmEx FY2022 effective tax rate 21.6%). Restructuring the text chunks did
  not move hit@6 (8/20 before and after on the original 20 questions), and an embedded contextual header made
  it worse (6/20). Retrieval gains have to come from ranking: hybrid search in M2
  ([docs/m1-results.md](docs/m1-results.md)).

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env            # add OPENAI_API_KEY, ANTHROPIC_API_KEY and SEC_USER_AGENT ("Name email@example.com")
uv sync
docker compose up -d db         # Postgres 16 + pgvector
uv run alembic upgrade head

# data: 21 FinanceBench 10-Ks and the XBRL facts of their 10 companies
uv run python eval/build_subset.py --all-10k
uv run python -m filinglens.ingestion.run --from-subset eval/financebench_10k.jsonl
uv run python -m filinglens.ingestion.load_facts --from-subset eval/financebench_10k.jsonl

# evaluation
uv run python eval/retrieval_hit.py --name my-run --subset eval/financebench_10k.jsonl   # no LLM
uv run python eval/run_eval.py --mode rag --name my-rag-run                              # LLM + judge, 20 questions

uv run pytest
```

SEC EDGAR needs no API key, but every request must identify you in `SEC_USER_AGENT`; the client stays under
SEC's 10 requests per second and caches raw downloads in `data/raw/`.

## Repository layout

```
src/filinglens/
  config.py, llm.py           settings (.env) and the swappable chat/embedding models
  financials.py               get_metric(): one XBRL value per company, metric and fiscal year
  db/                         SQLAlchemy models + Alembic migrations
  ingestion/
    edgar_client.py           rate-limited, cached SEC EDGAR client
    html_text.py              10-K HTML to text, one line per table row
    parse_10k.py              split a 10-K into its Item sections
    chunk_embed.py            per-Item chunking, embeddings cached by content hash
    run.py, load_facts.py     ingestion CLIs: filings and XBRL facts
  retrieval/vector.py         vector search within one filing
  agent/prompts.py            versioned prompts
eval/                         question sets, eval runners, saved results
docs/                         milestone results, ADRs, v2 ideas
tests/
```

## Roadmap

- [x] **M0 PoC:** EDGAR client, FinanceBench subset, closed-book baseline, plain RAG, go/no-go
- [x] **M1 Data:** XBRL facts, Item sections, table rows, per-Item chunks, wider eval (51 questions)
- [ ] **M2 RAG:** hybrid search (vector + Postgres full-text) with ticker, year and section filters, citations
- [ ] **M3 Agent:** LangGraph with search, financials and calculator tools; refuse and not-found paths
- [ ] **M4 Interfaces:** FastAPI with streaming, Streamlit UI
- [ ] **M5 Quality:** full eval on 51 questions against the closed-book baseline, one measured improvement
- [ ] **M6 Ship:** MCP server, Langfuse tracing, CI, deploy

## Data and licences

- Filings and XBRL facts: [SEC EDGAR](https://www.sec.gov/edgar), public data.
- Questions: [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) by Patronus AI,
  CC-BY-NC-4.0, used for evaluation only.
