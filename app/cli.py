"""Command-line chat interface for the Local Wikipedia RAG.

Run from the repo root::

    python -m app.cli                # interactive chat
    python -m app.cli --ask "Who was Ada Lovelace?"
    python -m app.cli --stats        # corpus stats
    python -m app.cli --reset        # clear the vector store + SQLite

Built-in slash commands inside the chat:

    :sources         toggle "show retrieved chunks" mode
    :stats           print corpus stats
    :reset           wipe the index (asks for confirmation)
    :history         print this session's Q/A transcript
    :clear           clear the on-screen transcript
    :help            show this help
    :exit / :quit    leave the chat
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make ``src`` importable when launched as ``python -m app.cli`` or
# ``python app/cli.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLM_MODEL  # noqa: E402
from src.rag_pipeline import RAGEngine  # noqa: E402
from src.vector_store import MetadataDB, VectorStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


HELP_TEXT = """\
Slash commands:
  :sources    Toggle showing retrieved context with each answer
  :stats      Print corpus stats (documents, chunks, vectors)
  :reset      Wipe the vector store + SQLite (asks for confirmation)
  :history    Print the current session transcript
  :clear      Clear the on-screen transcript (keeps the index)
  :help       Show this help
  :exit       Leave the chat (alias: :quit)
"""


def print_banner(model: str, stats: dict) -> None:
    print("=" * 72)
    print(" Local Wikipedia RAG — interactive CLI")
    print(f" Model: {model}")
    print(
        f" Corpus: {stats.get('documents', 0)} docs "
        f"({stats.get('people', 0)} people, {stats.get('places', 0)} places) "
        f"— {stats.get('chunks', 0)} chunks, {stats.get('vectors', 0)} vectors"
    )
    print(" Type your question, or :help for commands.")
    print("=" * 72)


def render_chunks(chunks: list[dict]) -> None:
    if not chunks:
        print("  (no chunks retrieved)")
        return
    for idx, hit in enumerate(chunks, start=1):
        meta = hit.get("metadata", {})
        score = hit.get("score", 0.0)
        text = hit.get("text", "")
        snippet = text if len(text) <= 320 else text[:317] + "..."
        print(
            f"  [{idx}] {meta.get('entity_name', '?')} ({meta.get('type', '?')}) "
            f"§ {meta.get('section', 'Overview')} — score={score:.3f}"
        )
        print(f"      {snippet}")
        print(f"      {meta.get('url', '')}")


def cmd_stats(stats: dict) -> None:
    print(json.dumps(stats, indent=2))


def cmd_reset(force: bool = False) -> None:
    if not force:
        confirm = input("This will delete all vectors and SQLite rows. Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return
    store = VectorStore()
    store.reset()
    db_path = MetadataDB().path
    if db_path.exists():
        db_path.unlink()
    print("System reset. Re-run ingestion: python scripts/run_ingest.py")


def chat_loop(engine: RAGEngine, show_sources: bool = False) -> None:
    history: list[dict] = []
    print_banner(engine.generator.model, engine.stats())
    while True:
        try:
            query = input("\n you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n bye.")
            return
        if not query:
            continue
        if query.startswith(":"):
            cmd = query.lower()
            if cmd in {":exit", ":quit"}:
                return
            if cmd == ":help":
                print(HELP_TEXT)
                continue
            if cmd == ":sources":
                show_sources = not show_sources
                print(f"  show_sources = {show_sources}")
                continue
            if cmd == ":stats":
                cmd_stats(engine.stats())
                continue
            if cmd == ":reset":
                cmd_reset()
                return
            if cmd == ":history":
                if not history:
                    print("  (no history yet)")
                for turn in history:
                    print(f"\n you > {turn['query']}")
                    print(f" rag > {turn['answer']}")
                continue
            if cmd == ":clear":
                history.clear()
                print("  transcript cleared.")
                continue
            print(f"  unknown command. Try :help")
            continue

        try:
            result = engine.answer(query)
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {exc}")
            continue
        history.append({"query": query, "answer": result.answer})
        print(f"\n rag > {result.answer}")
        print(
            f"      [model={result.generation.model}  "
            f"latency={result.generation.elapsed_seconds:.2f}s  "
            f"top_k={len(result.chunks)}  "
            f"target_types={'/'.join(result.routed.target_types)}]"
        )
        print(f"      route: {result.routed.rationale}")
        if show_sources:
            print(" sources >")
            render_chunks(result.chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Wikipedia RAG CLI")
    parser.add_argument("--ask", help="One-shot question; print the answer and exit.")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved chunks alongside the answer.")
    parser.add_argument("--stats", action="store_true", help="Print corpus stats and exit.")
    parser.add_argument("--reset", action="store_true", help="Wipe the index (asks for confirmation unless --force).")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts (for scripting).")
    parser.add_argument("--top-k", type=int, default=None, help="Override TOP_K for retrieval.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.reset:
        cmd_reset(force=args.force)
        return 0

    engine = RAGEngine()

    if args.stats:
        cmd_stats(engine.stats())
        return 0

    if args.ask:
        result = engine.answer(args.ask, top_k=args.top_k)
        print(result.answer)
        print(
            f"\n[model={result.generation.model}  "
            f"latency={result.generation.elapsed_seconds:.2f}s  "
            f"top_k={len(result.chunks)}  "
            f"target_types={'/'.join(result.routed.target_types)}]"
        )
        print(f"route: {result.routed.rationale}")
        if args.show_sources:
            print("sources:")
            render_chunks(result.chunks)
        return 0

    try:
        chat_loop(engine, show_sources=args.show_sources)
    except KeyboardInterrupt:
        print("\n bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
