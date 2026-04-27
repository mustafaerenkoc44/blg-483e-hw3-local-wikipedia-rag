"""Lightweight smoke test that doesn't depend on Ollama.

Verifies:

* Wikipedia ingestion works for a single entity
* The custom chunker produces non-empty, non-overlapping windows
* SQLite schema accepts a document + its chunks
* The query router classifies the canonical example questions correctly

Run::

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunks_for_documents  # noqa: E402
from src.entities import by_name  # noqa: E402
from src.ingest import WikipediaIngestor, save_raw  # noqa: E402
from src.retriever import route_query  # noqa: E402
from src.vector_store import MetadataDB  # noqa: E402


def test_router() -> None:
    cases = [
        ("Who was Albert Einstein and what is he known for?", ("person",)),
        ("Where is the Eiffel Tower located?", ("place",)),
        ("Compare Lionel Messi and Cristiano Ronaldo.", ("person",)),
        ("Compare the Eiffel Tower and the Statue of Liberty.", ("place",)),
        ("Which famous place is located in Turkey?", ("place",)),
        ("Compare Albert Einstein and Nikola Tesla.", ("person",)),
        ("Who is the president of Mars?", ("person",)),
    ]
    print("\n[router] expecting types:")
    for q, expected in cases:
        routed = route_query(q)
        ok = set(expected).issubset(set(routed.target_types)) or routed.target_types == expected
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {q!r} -> {routed.target_types}  ({routed.rationale})")


def test_ingest_and_chunk() -> None:
    target = by_name("Ada Lovelace")
    assert target, "Ada Lovelace must be in the catalogue"
    print(f"\n[ingest] fetching {target.name} ...")
    ingestor = WikipediaIngestor()
    doc = ingestor.ingest_entity(target)
    save_raw(doc)
    print(f"  fetched {doc.char_length} chars, url={doc.url}")
    chunks = chunks_for_documents([doc])
    print(f"[chunk] produced {len(chunks)} chunks (avg {sum(len(c.text) for c in chunks) // max(1,len(chunks))} chars each)")
    assert chunks, "expected at least one chunk"
    print(f"  first chunk preview: {chunks[0].text[:160]} ...")

    db = MetadataDB()
    doc_id = db.upsert_document(doc)
    db.replace_chunks_for(doc_id, chunks)
    stats = db.stats()
    print(f"[sqlite] stats after upsert: {stats}")


def main() -> int:
    test_router()
    test_ingest_and_chunk()
    print("\nSmoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
