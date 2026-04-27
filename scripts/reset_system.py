"""Reset the persistent state — vector store and SQLite database.

Usage::

    python scripts/reset_system.py            # interactive confirmation
    python scripts/reset_system.py --force    # skip confirmation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHROMA_DIR, SQLITE_PATH  # noqa: E402
from src.vector_store import MetadataDB, VectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset Local Wikipedia RAG state.")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.force:
        confirm = input(
            f"Wipe {CHROMA_DIR} and {SQLITE_PATH}? Type 'yes' to confirm: "
        ).strip().lower()
        if confirm != "yes":
            print(" Aborted.")
            return 1

    VectorStore().reset()
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
        print(f" Deleted {SQLITE_PATH}")
    # Recreate the SQLite schema so the next ingest run can write straight away.
    MetadataDB()
    print(" Done. Re-run: python scripts/run_ingest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
