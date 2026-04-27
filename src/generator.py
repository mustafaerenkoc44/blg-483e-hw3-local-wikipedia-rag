"""Local LLM generator backed by the Ollama HTTP API.

We deliberately avoid wrapper libraries (LangChain, llama-index, the
``ollama`` Python SDK) and call the bare REST endpoint with
``requests``. Two endpoints are wired up:

* ``POST /api/generate`` — non-streaming completion; returns the full
  answer in one JSON payload. Used by the CLI.
* ``POST /api/generate`` with ``stream=True`` — newline-delimited JSON
  stream; used by the Streamlit UI to surface tokens as they arrive.

The system prompt enforces three rules from the brief:

1. Answers must be grounded in the provided context.
2. Hallucinations should be avoided — if the answer is not present,
   reply with exactly ``I don't know``.
3. Sources must be quotable, so we ask the model to reference them by
   the bracketed numbers we render in the prompt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, Iterator

import requests

from .config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE, LLM_TIMEOUT, OLLAMA_HOST

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a careful research assistant answering questions about famous people and places using ONLY the provided Wikipedia context.

Rules you MUST follow:
1. Use only facts contained in the numbered context blocks below. Do not invent details.
2. If the context does not contain the answer, reply with exactly: I don't know.
3. When citing a fact, mention the source like [1], [2] referring to the context block numbers.
4. Keep answers concise and factual. For comparison questions, use a short bullet list.
5. Never speculate about topics outside the provided context.
"""


@dataclass
class GenerationResult:
    answer: str
    prompt: str
    model: str
    elapsed_seconds: float
    eval_count: int | None = None


def _format_context(chunks: Iterable[dict]) -> str:
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        header = (
            f"[{idx}] Entity: {meta.get('entity_name', 'unknown')} "
            f"({meta.get('type', '?')}) — Section: {meta.get('section', 'Overview')} "
            f"— Source: {meta.get('url', '')}"
        )
        blocks.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


def build_prompt(query: str, chunks: Iterable[dict]) -> str:
    context = _format_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ===\n"
    )


class LLMGenerator:
    """Calls Ollama's ``/api/generate`` endpoint."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        host: str = OLLAMA_HOST,
        temperature: float = LLM_TEMPERATURE,
        num_ctx: int = LLM_NUM_CTX,
        timeout: int = LLM_TIMEOUT,
    ) -> None:
        self.model = model
        self.url = f"{host.rstrip('/')}/api/generate"
        self.tags_url = f"{host.rstrip('/')}/api/tags"
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    def health_check(self) -> dict:
        try:
            response = requests.get(self.tags_url, timeout=5)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.url}: {exc}. "
                "Start it with `ollama serve` and pull the model."
            ) from exc
        tags = response.json().get("models", [])
        names = {m.get("name", "").split(":")[0] for m in tags}
        return {"models": tags, "names": names}

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

    def generate(self, query: str, chunks: list[dict]) -> GenerationResult:
        import time

        prompt = build_prompt(query, chunks)
        start = time.perf_counter()
        try:
            response = requests.post(
                self.url,
                json=self._payload(prompt, stream=False),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama generate call failed: {exc}") from exc
        data = response.json()
        elapsed = time.perf_counter() - start
        return GenerationResult(
            answer=data.get("response", "").strip(),
            prompt=prompt,
            model=self.model,
            elapsed_seconds=elapsed,
            eval_count=data.get("eval_count"),
        )

    def stream(self, query: str, chunks: list[dict]) -> Iterator[str]:
        prompt = build_prompt(query, chunks)
        with requests.post(
            self.url,
            json=self._payload(prompt, stream=True),
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                token = payload.get("response", "")
                if token:
                    yield token
                if payload.get("done"):
                    break
