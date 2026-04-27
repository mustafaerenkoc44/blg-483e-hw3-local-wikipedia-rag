"""Wikipedia ingestion using the MediaWiki REST + Action APIs.

Per the homework brief we deliberately avoid wrapper libraries such as
``wikipedia`` or ``wikipedia-api`` and call the public endpoints with
``requests`` directly. Two endpoints are used:

* ``/w/api.php?action=query&prop=extracts`` returns plain-text article
  bodies (``explaintext=1`` strips wiki markup).
* ``/api/rest_v1/page/summary/{title}`` returns a short summary plus the
  canonical URL — convenient for citing the source.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests

from .config import RAW_DIR, WIKI_LANG, WIKI_TIMEOUT, WIKI_USER_AGENT, ensure_dirs
from .entities import ALL_ENTITIES, Entity

logger = logging.getLogger(__name__)


@dataclass
class WikiDocument:
    """A normalised Wikipedia article ready for chunking."""

    entity_name: str
    type: str
    title: str
    url: str
    summary: str
    text: str
    fetched_at: str

    @property
    def char_length(self) -> int:
        return len(self.text)


class WikipediaIngestor:
    """Polite, rate-limited fetcher for Wikipedia articles.

    We respect the Wikimedia API etiquette by sending a descriptive
    ``User-Agent`` header (required) and sleeping briefly between calls.
    """

    API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
    REST_SUMMARY_TEMPLATE = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

    def __init__(self, lang: str = WIKI_LANG, sleep_seconds: float = 0.4) -> None:
        self.lang = lang
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": WIKI_USER_AGENT,
                "Accept": "application/json",
            }
        )

    def fetch_extract(self, title: str) -> tuple[str, str]:
        """Return (canonical_title, plain_text_body) for the article."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
        url = self.API_URL_TEMPLATE.format(lang=self.lang)
        response = self.session.get(url, params=params, timeout=WIKI_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        if not pages:
            raise ValueError(f"No pages returned for title={title!r}")
        page = next(iter(pages.values()))
        if "missing" in page:
            raise ValueError(f"Wikipedia article {title!r} does not exist")
        canonical_title = page.get("title", title)
        extract = page.get("extract", "").strip()
        if not extract:
            raise ValueError(f"Empty extract for {title!r}")
        return canonical_title, extract

    def fetch_summary(self, title: str) -> tuple[str, str]:
        """Return (summary_text, canonical_url) using REST summary endpoint."""
        url = self.REST_SUMMARY_TEMPLATE.format(lang=self.lang, title=quote(title.replace(" ", "_")))
        response = self.session.get(url, timeout=WIKI_TIMEOUT)
        if response.status_code == 404:
            raise ValueError(f"Wikipedia summary 404 for {title!r}")
        response.raise_for_status()
        data = response.json()
        summary = data.get("extract", "").strip()
        canonical_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not canonical_url:
            canonical_url = f"https://{self.lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        return summary, canonical_url

    def ingest_entity(self, entity: Entity) -> WikiDocument:
        canonical_title, body = self.fetch_extract(entity.wiki_title)
        time.sleep(self.sleep_seconds)
        try:
            summary, url = self.fetch_summary(canonical_title)
        except Exception as exc:
            logger.warning("Summary fetch failed for %s: %s", canonical_title, exc)
            summary = ""
            url = f"https://{self.lang}.wikipedia.org/wiki/{quote(canonical_title.replace(' ', '_'))}"
        time.sleep(self.sleep_seconds)
        return WikiDocument(
            entity_name=entity.name,
            type=entity.type,
            title=canonical_title,
            url=url,
            summary=summary,
            text=body,
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def ingest_many(self, entities: Iterable[Entity]) -> list[WikiDocument]:
        documents: list[WikiDocument] = []
        for entity in entities:
            try:
                doc = self.ingest_entity(entity)
                logger.info("Fetched %s (%d chars)", entity.name, doc.char_length)
                documents.append(doc)
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", entity.name, exc)
        return documents


def save_raw(doc: WikiDocument, raw_dir: Path = RAW_DIR) -> Path:
    """Persist a single article to ``data/raw/<entity>.json`` for inspection."""
    ensure_dirs()
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = doc.entity_name.replace(" ", "_").replace("/", "-") + ".json"
    path = raw_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(doc), fh, ensure_ascii=False, indent=2)
    return path


def ingest_all(entities: Iterable[Entity] = ALL_ENTITIES) -> list[WikiDocument]:
    """Ingest every entity and write the raw JSON artefacts."""
    ensure_dirs()
    ingestor = WikipediaIngestor()
    documents = ingestor.ingest_many(entities)
    for doc in documents:
        save_raw(doc)
    return documents
