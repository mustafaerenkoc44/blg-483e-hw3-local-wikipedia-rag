"""Query router and retriever.

Two responsibilities:

1. **Route the query** — decide whether the question is about a person,
   a place, both, or neither, using a small lexicon-based classifier
   over the curated entity list. This is intentionally simple per the
   brief ("Keyword based or rule based approaches are acceptable").
2. **Retrieve grounded chunks** — embed the query, run a single
   metadata-filtered vector search, and (when a specific entity is
   recognised) re-rank to prefer chunks for that entity so a question
   about Einstein cannot be answered by a Tesla chunk that happens to
   be semantically close.

Routing decisions are exposed in :class:`RoutedQuery` so the UI can
explain *why* certain chunks were retrieved.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from .config import TOP_K
from .embeddings import Embedder
from .entities import ALL_ENTITIES, PEOPLE, PLACES, Entity
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


_PERSON_HINTS = {
    "who", "person", "scientist", "artist", "footballer", "physicist", "musician",
    "painter", "inventor", "discovered", "invented", "wrote", "composed", "born",
    "died", "biography", "life of", "his", "her",
}
_PLACE_HINTS = {
    "where", "located", "place", "city", "country", "tower", "wall",
    "mountain", "river", "monument", "landmark", "building", "museum",
    "wonder", "structure", "located in", "in turkey", "in italy", "in egypt",
    "in china",
}
_COMPARE_HINTS = {"compare", "difference", "vs", "versus", "between", "or"}


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.lower())


def _detect_entities(query: str, pool: Iterable[Entity]) -> list[Entity]:
    q_low = query.lower()
    matches: list[Entity] = []
    for ent in pool:
        candidates = (ent.name.lower(), ent.wiki_title.lower(), *ent.aliases)
        if any(c and c in q_low for c in candidates):
            matches.append(ent)
            continue
        first_token = ent.name.split(" ", 1)[0].lower()
        if len(first_token) >= 4 and re.search(rf"\b{re.escape(first_token)}\b", q_low):
            matches.append(ent)
    return matches


@dataclass
class RoutedQuery:
    query: str
    target_types: tuple[str, ...]
    entities: list[Entity] = field(default_factory=list)
    is_comparison: bool = False
    rationale: str = ""

    def chroma_filter(self) -> dict | None:
        if not self.target_types or set(self.target_types) == {"person", "place"}:
            return None
        return {"type": self.target_types[0]}


def route_query(query: str) -> RoutedQuery:
    """Classify a question against the curated entity catalogue."""
    detected_people = _detect_entities(query, PEOPLE)
    detected_places = _detect_entities(query, PLACES)
    detected = detected_people + detected_places

    tokens = set(_tokenise(query))
    person_hint = bool(tokens & _PERSON_HINTS)
    place_hint = bool(tokens & _PLACE_HINTS)
    is_comparison = bool(tokens & _COMPARE_HINTS) or len(detected) >= 2

    if detected_people and detected_places:
        types = ("person", "place")
        rationale = "Detected at least one person AND one place — searching both subsets."
    elif detected_people:
        types = ("person",)
        rationale = "Recognised a person entity — restricting search to people."
    elif detected_places:
        types = ("place",)
        rationale = "Recognised a place entity — restricting search to places."
    elif person_hint and not place_hint:
        types = ("person",)
        rationale = "Person-shaped phrasing (e.g. 'who…') — restricting to people."
    elif place_hint and not person_hint:
        types = ("place",)
        rationale = "Place-shaped phrasing (e.g. 'where…') — restricting to places."
    else:
        types = ("person", "place")
        rationale = "Ambiguous — searching both people and places."

    return RoutedQuery(
        query=query,
        target_types=types,
        entities=detected,
        is_comparison=is_comparison,
        rationale=rationale,
    )


class Retriever:
    """High-level retrieval API used by the chat layer."""

    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = TOP_K) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def _boost_by_entity(self, hits: list[dict], entities: list[Entity]) -> list[dict]:
        if not entities:
            return hits
        target_names = {e.name for e in entities}
        boosted: list[dict] = []
        for hit in hits:
            new = dict(hit)
            if hit.get("metadata", {}).get("entity_name") in target_names:
                new["score"] = hit["score"] + 0.25
            boosted.append(new)
        boosted.sort(key=lambda h: h["score"], reverse=True)
        return boosted

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[RoutedQuery, list[dict]]:
        routed = route_query(query)
        embedding = self.embedder.embed_query(query)
        k = top_k or self.top_k
        if routed.is_comparison and len(routed.entities) >= 2:
            per_entity_k = max(2, k // max(len(routed.entities), 1) + 1)
            hits: list[dict] = []
            seen_ids: set[str] = set()
            for ent in routed.entities:
                where = {"entity_name": ent.name}
                results = self.store.query(embedding, top_k=per_entity_k, where=where)
                for r in results:
                    if r["id"] not in seen_ids:
                        hits.append(r)
                        seen_ids.add(r["id"])
            hits = sorted(hits, key=lambda h: h["score"], reverse=True)[:k]
        else:
            results = self.store.query(embedding, top_k=k * 2, where=routed.chroma_filter())
            hits = self._boost_by_entity(results, routed.entities)[:k]
        return routed, hits
