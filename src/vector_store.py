"""Vector storage and metadata persistence.

Design choice — **Option B: a single Chroma collection with metadata**.

Reasoning (also documented in ``recommendation.md``):

* A single collection lets us answer mixed queries ("compare Einstein
  and the Eiffel Tower") with one vector search and a metadata filter,
  rather than fanning out into two stores and merging the results.
* Filtering is cheap in Chroma — ``where={"type": "person"}`` runs as a
  metadata predicate and never recomputes embeddings.
* SQLite holds the *canonical* document text and chunk provenance so we
  can rebuild the vector store from scratch without re-fetching
  Wikipedia.

The two persistence layers therefore play different roles:

============  ===================================================
SQLite        canonical text, source URL, chunk → document mapping
Chroma        embeddings + minimal metadata for fast retrieval
============  ===================================================
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import chromadb

from .chunker import Chunk
from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    SQLITE_PATH,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    summary TEXT,
    text TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    char_length INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    section TEXT,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type);
"""


class MetadataDB:
    """SQLite wrapper for documents and chunks."""

    def __init__(self, path: Path = SQLITE_PATH) -> None:
        ensure_dirs()
        self.path = path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def upsert_document(self, doc) -> int:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                """
                INSERT INTO documents (entity_name, type, title, url, summary, text, fetched_at, char_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_name) DO UPDATE SET
                    type=excluded.type,
                    title=excluded.title,
                    url=excluded.url,
                    summary=excluded.summary,
                    text=excluded.text,
                    fetched_at=excluded.fetched_at,
                    char_length=excluded.char_length
                """,
                (
                    doc.entity_name,
                    doc.type,
                    doc.title,
                    doc.url,
                    doc.summary,
                    doc.text,
                    doc.fetched_at,
                    len(doc.text),
                ),
            )
            row = conn.execute(
                "SELECT id FROM documents WHERE entity_name = ?",
                (doc.entity_name,),
            ).fetchone()
            return int(row["id"])

    def replace_chunks_for(self, document_id: int, chunks: Iterable[Chunk]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, document_id, chunk_index, section, text, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        document_id,
                        c.chunk_index,
                        c.section,
                        c.text,
                        c.char_start,
                        c.char_end,
                    )
                    for c in chunks
                ],
            )

    def stats(self) -> dict:
        with closing(self._connect()) as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            people = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE type = 'person'"
            ).fetchone()[0]
            places = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE type = 'place'"
            ).fetchone()[0]
            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "people": people,
                "places": places,
            }

    def list_entities(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT entity_name, type, title, url, char_length, fetched_at FROM documents ORDER BY type, entity_name"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT c.chunk_id, c.text, c.section, c.chunk_index,
                       d.entity_name, d.type, d.title, d.url
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        return dict(row) if row else None


class VectorStore:
    """Persistent Chroma collection (single collection, metadata-filtered)."""

    def __init__(
        self,
        directory: Path = CHROMA_DIR,
        collection: str = CHROMA_COLLECTION,
    ) -> None:
        ensure_dirs()
        self._client = chromadb.PersistentClient(path=str(directory))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection.name)
        except Exception:
            logger.exception("Failed to delete collection — recreating anyway")
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        # Chroma rejects re-adding the same id, so upsert chunk-by-chunk.
        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(v) for v in embeddings],
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs = {
            "query_embeddings": [list(embedding)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        if not result.get("ids") or not result["ids"][0]:
            return []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        return [
            {
                "id": _id,
                "text": doc,
                "metadata": meta,
                "distance": float(dist),
                "score": 1.0 - float(dist),
            }
            for _id, doc, meta, dist in zip(ids, docs, metas, dists)
        ]


def metadata_for_chunk(chunk: Chunk) -> dict:
    """Minimal metadata stored alongside each Chroma vector."""
    return {
        "entity_name": chunk.entity_name,
        "type": chunk.type,
        "title": chunk.title,
        "url": chunk.url,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
    }


def export_metadata_summary(db: MetadataDB) -> str:
    """Pretty-printed corpus stats for the CLI/UI."""
    return json.dumps(db.stats(), indent=2)


def serialize_chunk(chunk: Chunk) -> dict:
    return asdict(chunk)
