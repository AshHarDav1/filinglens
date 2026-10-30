"""Chunking and cached embedding.

`build_chunks` (M1) chunks each 10-K Item separately with `chunk_text`, so no chunk crosses two
Items, and labels every chunk with its Item.

No contextual header is embedded: the M1 ablation (eval/results/2026-10-30-m1-ablation.json) found
that a "company · FY · Item" header lowered hit@6 (8/20 -> 6/20). Search is already filtered to one
filing, so company and year are identical on every candidate chunk and only make them look alike.
The prompt shows that context to the LLM instead (eval/run_eval.py format_excerpts).
"""

import hashlib
import re

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from filinglens.db.models import EmbeddingCache
from filinglens.ingestion.parse_10k import Section

CHUNK_SIZE = 2000  # characters, roughly 500 tokens
CHUNK_OVERLAP = 200
# Smaller sections are boilerplate ("Item 9B. None.", "see Item 15") and would only add noise
MIN_SECTION_CHARS = 200

_WHITESPACE = re.compile(r"\s+")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into windows of at most `size` chars, overlapping by about `overlap` chars.

    Each window prefers to end at a line break (or else a space) in its second half, so chunks
    rarely cut a sentence or table row in two.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # Start the next window `overlap` chars back, moved forward to a word boundary.
        next_start = max(end - overlap, start + 1)
        boundary = _WHITESPACE.search(text, next_start, end)
        start = boundary.end() if boundary else next_start
    return chunks


class ChunkDraft(BaseModel):
    section: str  # the Item, e.g. "8"
    content: str


def build_chunks(sections: list[Section]) -> list[ChunkDraft]:
    """Chunk every Item on its own, so no chunk crosses two Items.

    Near-empty Items are skipped. The cover page is kept: besides the table of contents it holds
    facts that questions ask about (e.g. the securities registered under Section 12(b)).
    """
    return [
        ChunkDraft(section=section.item, content=piece)
        for section in sections
        if len(section.text.strip()) >= MIN_SECTION_CHARS
        for piece in chunk_text(section.text)
    ]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_with_cache(
    session: Session,
    embedder: Embeddings,
    model: str,
    texts: list[str],
    batch_size: int = 100,
) -> tuple[list[list[float]], int]:
    """Embed `texts`, reusing cached vectors. Returns (vectors, number of newly embedded texts)."""
    hashes = [content_hash(t) for t in texts]
    cached = {
        row.content_hash: row.embedding
        for row in session.scalars(
            select(EmbeddingCache).where(
                EmbeddingCache.model == model, EmbeddingCache.content_hash.in_(set(hashes))
            )
        )
    }
    missing = {h: t for h, t in zip(hashes, texts, strict=True) if h not in cached}
    items = list(missing.items())
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        vectors = embedder.embed_documents([t for _, t in batch])
        for (h, _), vector in zip(batch, vectors, strict=True):
            session.add(EmbeddingCache(content_hash=h, model=model, embedding=vector))
            cached[h] = vector
    session.flush()
    return [list(cached[h]) for h in hashes], len(items)
