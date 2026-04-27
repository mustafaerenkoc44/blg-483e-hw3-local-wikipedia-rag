"""Top-level orchestration tying ingest → chunk → embed → retrieve → generate.

Two main objects:

* :class:`IngestionPipeline` — fetches Wikipedia, chunks the text,
  embeds chunks in batches and writes both SQLite and Chroma.
* :class:`RAGEngine` — answers a question end-to-end. Designed to be
  used by *both* the CLI and the Streamlit UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

from .chunker import chunks_for_documents
from .config import TOP_K
from .embeddings import Embedder
from .entities import ALL_ENTITIES, Entity
from .generator import GenerationResult, LLMGenerator
from .ingest import WikiDocument, WikipediaIngestor, save_raw
from .retriever import Retriever, RoutedQuery
from .vector_store import MetadataDB, VectorStore, metadata_for_chunk

logger = logging.getLogger(__name__)


_GENERIC_QUERY_TERMS = {
    "about",
    "answer",
    "associated",
    "between",
    "compare",
    "difference",
    "discover",
    "discovered",
    "does",
    "famous",
    "known",
    "located",
    "person",
    "people",
    "place",
    "places",
    "president",
    "random",
    "tell",
    "unknown",
    "used",
    "what",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def _content_terms(text: str) -> set[str]:
    """Return query terms that should be visibly supported by context."""
    import re

    terms = set()
    for token in re.findall(r"[\w'-]+", text.lower()):
        if len(token) <= 3:
            continue
        if token in _GENERIC_QUERY_TERMS:
            continue
        terms.add(token)
    return terms


def context_supports_query(query: str, routed: RoutedQuery, hits: list[dict]) -> bool:
    """Conservative refusal guard for no-entity and out-of-corpus questions.

    Named entities already get explicit entity-filtered retrieval. For generic
    questions, require at least one meaningful query term to appear in the
    retrieved context before letting the LLM answer. This catches failure cases
    such as "president of Mars" and "John Doe" without blocking normal entity
    questions.
    """
    if routed.entities:
        return True
    terms = _content_terms(query)
    if not terms:
        return True
    context_terms: set[str] = set()
    for hit in hits:
        context_terms.update(_content_terms(hit.get("text", "")))
        metadata = hit.get("metadata", {})
        context_terms.update(_content_terms(str(metadata.get("entity_name", ""))))
        context_terms.update(_content_terms(str(metadata.get("title", ""))))
    return bool(terms & context_terms)


@dataclass
class IngestionReport:
    documents_ingested: int = 0
    chunks_written: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class IngestionPipeline:
    """Single-pass ingestion: fetch → chunk → embed → persist."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        db: MetadataDB | None = None,
        ingestor: WikipediaIngestor | None = None,
        batch_size: int = 16,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.db = db or MetadataDB()
        self.ingestor = ingestor or WikipediaIngestor()
        self.batch_size = batch_size

    def run(self, entities: Iterable[Entity] = ALL_ENTITIES, *, reset: bool = False) -> IngestionReport:
        if reset:
            logger.info("Resetting vector store and metadata database before ingestion")
            self.store.reset()
            self.db.reset()

        report = IngestionReport()
        start = time.perf_counter()
        self.embedder.health_check()

        for entity in entities:
            try:
                doc = self.ingestor.ingest_entity(entity)
            except Exception as exc:
                logger.error("Ingest failed for %s: %s", entity.name, exc)
                report.failures.append((entity.name, str(exc)))
                continue
            try:
                save_raw(doc)
                doc_id = self.db.upsert_document(doc)
                chunks = chunks_for_documents([doc])
                if not chunks:
                    logger.warning("No chunks produced for %s", entity.name)
                    continue
                self.db.replace_chunks_for(doc_id, chunks)
                self._embed_and_store(chunks)
                report.documents_ingested += 1
                report.chunks_written += len(chunks)
                logger.info("✓ %s — %d chunks", entity.name, len(chunks))
            except Exception as exc:
                logger.exception("Embedding/storage failed for %s", entity.name)
                report.failures.append((entity.name, str(exc)))

        report.elapsed_seconds = time.perf_counter() - start
        return report

    def _embed_and_store(self, chunks: list) -> None:
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.text for c in batch]
            vectors = self.embedder.embed_documents(texts)
            self.store.add(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors,
                documents=texts,
                metadatas=[metadata_for_chunk(c) for c in batch],
            )


@dataclass
class RAGAnswer:
    query: str
    answer: str
    routed: RoutedQuery
    chunks: list[dict]
    generation: GenerationResult


class RAGEngine:
    """One-shot question answering using the persisted index."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        db: MetadataDB | None = None,
        generator: LLMGenerator | None = None,
        top_k: int = TOP_K,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.db = db or MetadataDB()
        self.generator = generator or LLMGenerator()
        self.retriever = Retriever(self.embedder, self.store, top_k=top_k)
        self.top_k = top_k

    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
        routed, hits = self.retriever.retrieve(query, top_k=top_k)
        if not hits or not context_supports_query(query, routed, hits):
            generation = GenerationResult(
                answer="I don't know.",
                prompt="No sufficiently relevant context was retrieved.",
                model=self.generator.model,
                elapsed_seconds=0.0,
            )
            return RAGAnswer(query=query, answer=generation.answer, routed=routed, chunks=hits, generation=generation)

        generation = self.generator.generate(query, hits)
        answer = generation.answer or "I don't know."
        return RAGAnswer(
            query=query,
            answer=answer,
            routed=routed,
            chunks=hits,
            generation=generation,
        )

    def stream(self, query: str, top_k: int | None = None):
        """Yield (kind, payload) tuples — useful for incremental UIs."""
        routed, hits = self.retriever.retrieve(query, top_k=top_k)
        yield "routed", routed
        yield "chunks", hits
        if not hits or not context_supports_query(query, routed, hits):
            yield "token", "I don't know."
            yield "done", None
            return
        for token in self.generator.stream(query, hits):
            yield "token", token
        yield "done", None

    def stats(self) -> dict:
        store_count = self.store.count
        db_stats = self.db.stats()
        db_stats["vectors"] = store_count
        return db_stats
