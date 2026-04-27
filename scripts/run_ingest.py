"""Run the full ingestion pipeline.

Usage::

    python scripts/run_ingest.py                # ingest the full default catalogue
    python scripts/run_ingest.py --reset        # wipe vectors + metadata and re-embed everything
    python scripts/run_ingest.py --only "Albert Einstein,Eiffel Tower"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.entities import ALL_ENTITIES, by_name  # noqa: E402
from src.rag_pipeline import IngestionPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia articles into the local RAG index.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the Chroma collection and SQLite metadata before ingesting.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated entity display names (e.g. 'Albert Einstein,Eiffel Tower').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        entities = []
        unknown = []
        for n in names:
            ent = by_name(n)
            if ent is None:
                unknown.append(n)
            else:
                entities.append(ent)
        if unknown:
            print(f"  Unknown entities (skipping): {', '.join(unknown)}")
        if not entities:
            print(" No valid entities to ingest.")
            return 1
    else:
        entities = list(ALL_ENTITIES)

    print(f" Ingesting {len(entities)} entities ...")
    pipeline = IngestionPipeline(batch_size=args.batch_size)
    try:
        report = pipeline.run(entities, reset=args.reset)
    except Exception as exc:  # noqa: BLE001
        print()
        print(" Ingestion could not start.")
        print(f" Reason: {exc}")
        print()
        print(" Check that Ollama is running and the embedding model is pulled:")
        print("   ollama serve")
        print("   ollama pull nomic-embed-text")
        return 2

    print()
    print("=" * 60)
    print(" Ingestion summary")
    print("=" * 60)
    print(f" documents : {report.documents_ingested}/{len(entities)}")
    print(f" chunks    : {report.chunks_written}")
    print(f" elapsed   : {report.elapsed_seconds:.1f}s")
    if report.failures:
        print()
        print(" Failures:")
        for name, reason in report.failures:
            print(f"   - {name}: {reason}")

    return 0 if not report.failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
