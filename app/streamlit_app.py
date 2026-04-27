"""Streamlit chat UI for the Local Wikipedia RAG.

Run from the repo root::

    streamlit run app/streamlit_app.py

Features:

* Streaming token-by-token answers from Ollama.
* Expandable "Retrieved context" panel showing each chunk, its
  similarity score, the section heading and the source URL.
* "Show route decision" so the user can see *why* the system chose
  to search people, places, or both.
* Chat history with a "Clear chat" button, plus a "Reset index"
  button that wipes the Chroma collection and SQLite database.
* Quick-pick example questions matching the homework brief.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLM_MODEL  # noqa: E402
from src.entities import PEOPLE, PLACES  # noqa: E402
from src.rag_pipeline import RAGEngine  # noqa: E402
from src.vector_store import MetadataDB, VectorStore  # noqa: E402


st.set_page_config(
    page_title="Local Wikipedia RAG",
    page_icon=":books:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading RAG engine ...")
def get_engine() -> RAGEngine:
    return RAGEngine()


def render_sources(chunks: list[dict]) -> None:
    if not chunks:
        st.info("No context was retrieved for this query.")
        return
    for idx, hit in enumerate(chunks, start=1):
        meta = hit.get("metadata", {})
        score = hit.get("score", 0.0)
        url = meta.get("url", "")
        title = (
            f"[{idx}] {meta.get('entity_name', '?')} — "
            f"{meta.get('type', '?')} · {meta.get('section', 'Overview')} · "
            f"score={score:.3f}"
        )
        with st.expander(title, expanded=False):
            st.write(hit.get("text", ""))
            if url:
                st.markdown(f"[Source on Wikipedia]({url})")


def sidebar(engine: RAGEngine) -> dict:
    st.sidebar.header("System")
    stats = engine.stats()
    st.sidebar.metric("Documents", stats.get("documents", 0))
    st.sidebar.metric("Chunks (SQLite)", stats.get("chunks", 0))
    st.sidebar.metric("Vectors (Chroma)", stats.get("vectors", 0))
    st.sidebar.write(f"People: **{stats.get('people', 0)}** · Places: **{stats.get('places', 0)}**")
    st.sidebar.divider()

    st.sidebar.header("Settings")
    show_sources = st.sidebar.checkbox("Show retrieved context", value=True)
    show_route = st.sidebar.checkbox("Show routing decision", value=True)
    use_streaming = st.sidebar.checkbox("Stream tokens", value=True)
    top_k = st.sidebar.slider("Top-K chunks", min_value=2, max_value=12, value=engine.top_k)
    st.sidebar.divider()

    st.sidebar.header("Quick examples")
    examples = [
        "Who was Albert Einstein and what is he known for?",
        "What did Marie Curie discover?",
        "Why is Nikola Tesla famous?",
        "Compare Lionel Messi and Cristiano Ronaldo.",
        "Where is the Eiffel Tower located?",
        "Why is the Great Wall of China important?",
        "Which famous place is located in Turkey?",
        "Compare Albert Einstein and Nikola Tesla.",
        "Who is the president of Mars?",
    ]
    for ex in examples:
        if st.sidebar.button(ex, key=f"ex::{ex}"):
            st.session_state.pending_query = ex

    st.sidebar.divider()
    st.sidebar.header("Maintenance")
    if st.sidebar.button("Clear chat"):
        st.session_state.history = []
        st.rerun()
    if st.sidebar.button("Reset index", help="Wipe Chroma collection and SQLite"):
        VectorStore().reset()
        db_path = MetadataDB().path
        if db_path.exists():
            db_path.unlink()
        st.cache_resource.clear()
        st.success("Index reset. Re-run `python scripts/run_ingest.py`.")

    return {
        "show_sources": show_sources,
        "show_route": show_route,
        "use_streaming": use_streaming,
        "top_k": top_k,
    }


def render_history() -> None:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("route"):
                st.caption(f"Route: {turn['route']}")
            if turn.get("sources"):
                with st.expander("Retrieved context", expanded=False):
                    render_sources(turn["sources"])


def handle_query(engine: RAGEngine, query: str, settings: dict) -> None:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        route_box = st.empty() if settings["show_route"] else None
        sources_box = st.empty() if settings["show_sources"] else None
        answer_box = st.empty()
        latency_box = st.empty()

        start = time.perf_counter()
        if settings["use_streaming"]:
            buffer = ""
            routed = None
            chunks: list[dict] = []
            for kind, payload in engine.stream(query, top_k=settings["top_k"]):
                if kind == "routed":
                    routed = payload
                    if route_box is not None:
                        route_box.caption(f"Route: {payload.rationale}")
                elif kind == "chunks":
                    chunks = payload
                    if sources_box is not None and chunks:
                        with sources_box.expander("Retrieved context", expanded=False):
                            render_sources(chunks)
                elif kind == "token":
                    buffer += payload
                    answer_box.markdown(buffer + " ▌")
                elif kind == "done":
                    answer_box.markdown(buffer or "I don't know.")
            elapsed = time.perf_counter() - start
            latency_box.caption(
                f"model={engine.generator.model} · latency={elapsed:.2f}s · top_k={len(chunks)}"
            )
            st.session_state.history.append(
                {
                    "query": query,
                    "answer": buffer or "I don't know.",
                    "route": routed.rationale if routed else "",
                    "sources": chunks,
                }
            )
        else:
            result = engine.answer(query, top_k=settings["top_k"])
            elapsed = time.perf_counter() - start
            if route_box is not None:
                route_box.caption(f"Route: {result.routed.rationale}")
            answer_box.markdown(result.answer)
            if sources_box is not None and result.chunks:
                with sources_box.expander("Retrieved context", expanded=False):
                    render_sources(result.chunks)
            latency_box.caption(
                f"model={result.generation.model} · latency={elapsed:.2f}s · top_k={len(result.chunks)}"
            )
            st.session_state.history.append(
                {
                    "query": query,
                    "answer": result.answer,
                    "route": result.routed.rationale,
                    "sources": result.chunks,
                }
            )


def main() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""

    st.title("Local Wikipedia RAG")
    st.caption(
        "Ask questions about famous people and places. Everything — embeddings, "
        "vector search, and the language model — runs on your machine."
    )

    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not initialise the RAG engine: {exc}")
        st.stop()

    settings = sidebar(engine)
    render_history()

    pending = st.session_state.pop("pending_query", "")
    user_input = st.chat_input("Ask about a famous person or place ...")
    query = user_input or pending or ""

    if query:
        handle_query(engine, query, settings)


if __name__ == "__main__":
    main()
