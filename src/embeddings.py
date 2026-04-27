"""Local embedding providers.

Two backends are supported and chosen via the ``RAG_EMBED_BACKEND``
environment variable:

* ``ollama`` (default) — calls ``POST /api/embeddings`` on the local
  Ollama daemon. Pull the model once with ``ollama pull nomic-embed-text``.
* ``sentence-transformers`` — uses ``sentence-transformers`` from
  Hugging Face (``all-MiniLM-L6-v2`` by default). Select it explicitly
  with ``RAG_EMBED_BACKEND=sentence-transformers``. The model is cached
  under ``~/.cache/huggingface`` after first download.

Both back-ends are fully local — no external API keys are needed and no
data leaves the machine.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import requests

from .config import EMBED_BACKEND, EMBED_MODEL, OLLAMA_HOST

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


class _OllamaEmbedder:
    """Thin wrapper around the Ollama HTTP embeddings endpoint."""

    def __init__(self, model: str = EMBED_MODEL, host: str = OLLAMA_HOST) -> None:
        self.model = model
        self.url = f"{host.rstrip('/')}/api/embeddings"
        self._dim: int | None = None

    def health_check(self) -> None:
        try:
            response = requests.get(f"{self.url.rsplit('/', 2)[0]}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as exc:
            raise EmbeddingError(
                f"Ollama is not reachable at {self.url}: {exc}. "
                "Is the daemon running? Try: ollama serve"
            ) from exc

    def embed_one(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingError("Refusing to embed empty text")
        try:
            response = requests.post(
                self.url,
                json={"model": self.model, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EmbeddingError(f"Ollama embeddings call failed: {exc}") from exc
        data = response.json()
        vec = data.get("embedding")
        if not vec:
            raise EmbeddingError(
                f"Ollama returned no embedding for model {self.model!r}. "
                "Did you run `ollama pull nomic-embed-text`?"
            )
        if self._dim is None:
            self._dim = len(vec)
        return list(vec)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class _SentenceTransformerEmbedder:
    """Optional fallback that uses ``sentence-transformers`` locally."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading sentence-transformers model %s ...", model_name)
        self._model = SentenceTransformer(model_name)
        self.model = model_name

    def health_check(self) -> None:
        return None

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


class Embedder:
    """Public interface used by the rest of the pipeline."""

    def __init__(self, backend: str = EMBED_BACKEND) -> None:
        backend = backend.lower()
        if backend == "ollama":
            self._impl = _OllamaEmbedder()
        elif backend in {"sentence-transformers", "st", "huggingface"}:
            self._impl = _SentenceTransformerEmbedder()
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")
        self.backend = backend

    @property
    def model_name(self) -> str:
        return self._impl.model

    def health_check(self) -> None:
        self._impl.health_check()

    def embed_query(self, query: str) -> list[float]:
        return self._impl.embed_one(query)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return self._impl.embed_many(list(texts))
