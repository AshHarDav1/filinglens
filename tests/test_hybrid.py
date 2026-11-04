"""Hybrid retrieval and section routing."""

from datetime import date

from filinglens.db.models import EMBEDDING_DIM, Chunk, Company, Filing
from filinglens.retrieval.hybrid import keyword_query, keyword_search, retrieve, rrf
from filinglens.retrieval.routing import SectionChoice, llm_sections, rule_sections
from filinglens.retrieval.vector import RetrievedChunk


def one_hot(i: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[i] = 1.0
    return vector


def chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, chunk_index=chunk_id, ticker="T", fiscal_year=2016, form="10-K",
        section="8", url="u", distance=0.5, content=f"chunk {chunk_id}",
    )  # fmt: skip


def test_keyword_query_drops_instruction_words():
    q = "What is the FY2016 COGS for Microsoft? Please state answer in USD millions."
    assert keyword_query(q) == "fy2016 | cogs | microsoft"
    assert keyword_query("Please answer, in millions.") is None


def test_rrf_rewards_chunks_high_on_both_lists():
    by_meaning = [chunk(1), chunk(2), chunk(3)]
    by_words = [chunk(3), chunk(4), chunk(1)]
    merged = [c.chunk_id for c in rrf([by_meaning, by_words])]
    assert merged[:2] == [1, 3]  # on both lists beats first-on-one-list-only
    assert set(merged) == {1, 2, 3, 4}
    assert len(rrf([by_meaning, by_words], k=2)) == 2


def test_rule_sections():
    assert rule_sections("What is the FY2016 COGS for Microsoft?") == ["7", "8"]
    assert rule_sections("Has Boeing reported any ongoing legal battles?") == ["1A", "3", "8"]
    assert rule_sections("Who is the CEO?") is None


class FakeRouter:
    def __init__(self, items):
        self.items = items

    def with_structured_output(self, schema):
        assert schema is SectionChoice
        return self

    def invoke(self, messages):
        return SectionChoice(items=self.items)


def test_llm_sections_keeps_only_valid_items():
    assert llm_sections("q", FakeRouter(["8", "item 99", "7", "8", "Cover"])) == ["8", "7", "cover"]
    assert llm_sections("q", FakeRouter(["nonsense"])) is None


def _filing(db_session, ticker: str, sections: dict[str, list[str]]):
    company = Company(cik=hash(ticker) % 10**9, ticker=ticker, name=f"{ticker} Inc")
    db_session.add(company)
    db_session.flush()
    filing = Filing(
        company_id=company.id, accession_no=f"acc-{ticker}", form="10-K", fiscal_year=2016,
        period_end=date(2016, 6, 30), filed_at=date(2016, 7, 28), url=f"https://x/{ticker}",
    )  # fmt: skip
    db_session.add(filing)
    db_session.flush()
    i = 0
    for section, texts in sections.items():
        for text in texts:
            db_session.add(
                Chunk(
                    filing_id=filing.id,
                    chunk_index=i,
                    section=section,
                    content=text,
                    embedding=one_hot(i),
                )
            )
            i += 1
    db_session.flush()


def test_keyword_search_ranks_by_matching_words(db_session):
    item8 = ["Total cost of revenue | 32,780 | 33,038", "Income statements: cost of revenue"]
    _filing(db_session, "TKW", {"7": ["Revenue grew in every segment."], "8": item8})
    hits = keyword_search(db_session, "What was the cost of revenue?", one_hot(0), "TKW", 2016, k=5)
    # words are OR-ed, so "revenue" alone also matches Item 7; matching more words ranks higher
    assert [h.section for h in hits] == ["8", "8", "7"]


def test_filter_searches_the_routed_items_and_tops_up(db_session):
    _filing(db_session, "TFL", {"1A": ["risk a", "risk b"], "8": ["statement"]})
    query = one_hot(0)  # closest to the first Item 1A chunk
    hits = retrieve(db_session, "q", query, "TFL", 2016, k=2, sections=["8"])
    assert hits[0].section == "8"  # the routed Item comes first
    assert len(hits) == 2  # topped up from the whole filing


def test_pointer_item8_falls_back_to_items_15_and_16(db_session):
    # Netflix-style filing: Item 8 is a one-line pointer, the statements follow Item 15
    _filing(db_session, "TNF", {"7": ["mdna"] * 3, "8": ["See Item 15."],
                                "15": ["Consolidated balance sheets"]})  # fmt: skip
    hits = retrieve(db_session, "q", one_hot(0), "TNF", 2016, k=2, sections=["8"])
    assert {h.section for h in hits} == {"8", "15"}
