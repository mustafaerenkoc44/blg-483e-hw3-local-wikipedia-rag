"""Custom sentence-aware chunker.

The brief asks us to define our own strategy and assume documents can be
large. We use *sentence-aware sliding-window* chunks:

1. Split the article into paragraphs on blank lines.
2. Within each paragraph, split into sentences with a small regex
   (no NLTK / spaCy — language-native logic only).
3. Greedily pack sentences into a window of approximately
   :data:`CHUNK_SIZE` characters; carry the trailing
   :data:`CHUNK_OVERLAP` characters into the next window so context is
   preserved across boundaries.
4. Chunks shorter than :data:`MIN_CHUNK_SIZE` are merged with their
   neighbour to avoid noisy single-sentence vectors.

Trade-offs (covered in ``recommendation.md``):

* Sentence-aware boundaries beat fixed character splits when the body
  contains short headings (e.g. "Early life") that would otherwise
  fragment a thought.
* Overlap improves recall on questions whose answer straddles two
  chunks, at the cost of ~20% extra storage and embedding compute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'À-ſ])")
_SECTION_RE = re.compile(r"\n=+\s*([^=\n]+?)\s*=+\n")
_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass
class Chunk:
    chunk_id: str
    entity_name: str
    type: str
    title: str
    url: str
    chunk_index: int
    text: str
    section: str
    char_start: int
    char_end: int


def _normalise(text: str) -> str:
    """Collapse repeated whitespace but preserve paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a Wikipedia plain-text extract into ``(section, body)`` pairs.

    Wikipedia ``extracts`` keep section headers like ``== Early life ==``.
    Anything before the first header lives in an implicit ``Overview``
    section.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("Overview", text.strip())]

    sections: list[tuple[str, str]] = []
    first = matches[0]
    intro = text[: first.start()].strip()
    if intro:
        sections.append(("Overview", intro))

    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((title, body))
    return sections


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentence-ish pieces using a regex heuristic."""
    pieces = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in pieces if p.strip()]


def _pack(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedy packing with character-based overlap."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if buffer_len + sentence_len <= chunk_size or not buffer:
            buffer.append(sentence)
            buffer_len += sentence_len
            continue

        chunks.append(" ".join(buffer).strip())

        carry: list[str] = []
        carry_len = 0
        for prev in reversed(buffer):
            if carry_len + len(prev) + 1 > overlap:
                break
            carry.insert(0, prev)
            carry_len += len(prev) + 1
        buffer = carry + [sentence]
        buffer_len = sum(len(s) + 1 for s in buffer)

    if buffer:
        chunks.append(" ".join(buffer).strip())
    return chunks


def _merge_small(chunks: list[str], min_size: int) -> list[str]:
    """Merge any chunk shorter than ``min_size`` into its neighbour."""
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for piece in chunks[1:]:
        if len(piece) < min_size:
            merged[-1] = (merged[-1] + " " + piece).strip()
        elif len(merged[-1]) < min_size:
            merged[-1] = (merged[-1] + " " + piece).strip()
        else:
            merged.append(piece)
    return merged


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_size: int = MIN_CHUNK_SIZE,
) -> list[tuple[str, str]]:
    """Chunk a single article. Returns list of ``(section, chunk_text)``."""
    text = _normalise(text)
    pieces: list[tuple[str, str]] = []
    for section, body in _split_sections(text):
        sentences = _split_sentences(body)
        if not sentences:
            continue
        raw_chunks = _pack(sentences, chunk_size=chunk_size, overlap=overlap)
        merged = _merge_small(raw_chunks, min_size=min_size)
        for chunk in merged:
            if chunk:
                pieces.append((section, chunk))
    return pieces


def make_chunks(
    *,
    entity_name: str,
    entity_type: str,
    title: str,
    url: str,
    text: str,
) -> list[Chunk]:
    """High-level helper that produces :class:`Chunk` records ready for storage."""
    raw_pairs = chunk_text(text)
    full = _normalise(text)
    chunks: list[Chunk] = []
    cursor = 0
    safe_name = entity_name.replace(" ", "_").replace("/", "-")
    for idx, (section, body) in enumerate(raw_pairs):
        located = full.find(body, cursor)
        if located == -1:
            located = full.find(body)
        char_start = max(located, 0)
        char_end = char_start + len(body)
        cursor = char_end
        chunks.append(
            Chunk(
                chunk_id=f"{safe_name}::{idx:03d}",
                entity_name=entity_name,
                type=entity_type,
                title=title,
                url=url,
                chunk_index=idx,
                text=body,
                section=section,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return chunks


def chunks_for_documents(documents: Iterable) -> list[Chunk]:
    """Convenience wrapper accepting :class:`ingest.WikiDocument` objects."""
    out: list[Chunk] = []
    for doc in documents:
        out.extend(
            make_chunks(
                entity_name=doc.entity_name,
                entity_type=doc.type,
                title=doc.title,
                url=doc.url,
                text=doc.text,
            )
        )
    return out
