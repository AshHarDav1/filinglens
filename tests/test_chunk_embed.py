import itertools

import pytest

from filinglens.ingestion.chunk_embed import chunk_text


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_overlap_and_cover_text():
    words = [f"w{i}" for i in range(2000)]
    text = " ".join(words)
    chunks = chunk_text(text, size=200, overlap=40)

    assert all(len(c) <= 200 for c in chunks)
    # every word appears in some chunk, in order of first appearance
    assert {w for c in chunks for w in c.split()} == set(words)
    # consecutive chunks overlap
    for a, b in itertools.pairwise(chunks):
        assert a.split()[-1] in b.split()


def test_prefers_line_breaks():
    text = ("a" * 60 + "\n") * 10
    chunks = chunk_text(text, size=150, overlap=0)
    assert all(set(c) <= {"a", "\n"} for c in chunks)
    assert all(len(line) == 60 for c in chunks for line in c.split("\n"))


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)
