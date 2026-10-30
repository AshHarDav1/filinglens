# ADR-001: pgvector in Postgres instead of a separate vector database

Status: accepted (M0, 2026-10-22)

## Context

FilingLens stores companies, filings, text chunks with embeddings, and (from M1) XBRL financial facts.
Retrieval always filters by company and fiscal year before ranking by similarity. The corpus is small:
about 6,000 chunks for 21 filings.

## Decision

Store embeddings in Postgres with the `pgvector` extension (`vector(1536)` column, cosine distance `<=>`),
next to the relational data. No separate vector database.

## Consequences

- One store, one backup, one connection. Filters and similarity search are one SQL query, with joins.
- Full-text search (M2) lives in the same database, so hybrid search is one query too.
- At this size an exact scan is fast; an HNSW index can be added later without changing the schema.
- Alternatives not picked: Qdrant, Pinecone, Weaviate. Worth it at millions of vectors or with heavy
  filtering at scale, not for a few thousand chunks.
