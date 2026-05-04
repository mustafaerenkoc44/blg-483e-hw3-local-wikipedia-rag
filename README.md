# Local Wikipedia RAG

> **BLG 483E — Homework 3**
> A simplified, ChatGPT-style retrieval-augmented generation system that runs **entirely on your laptop** — local embeddings, local vector store, local language model. Answers questions about 20 famous people and 20 famous places using Wikipedia as its only source of truth.

---

## What this system does

1. **Ingests** Wikipedia articles for 20 famous people + 20 famous places via the MediaWiki Action API (no `wikipedia` Python wrapper — bare HTTP).
2. **Chunks** each article with a sentence-aware sliding window so retrieval works across long sections.
3. **Embeds** every chunk locally with `nomic-embed-text` served by [Ollama](https://ollama.com), or with `sentence-transformers/all-MiniLM-L6-v2` as a fully local fallback.
4. **Stores** chunks in **two complementary layers**: SQLite for canonical text + provenance, ChromaDB for vector search.
5. **Routes** each query (person? place? both?) using a keyword/lexicon classifier built over the curated entity list.
6. **Generates** grounded answers with a local Ollama LLM (`llama3.2:3b` by default) using a strict "answer only from context, otherwise say *I don't know*" prompt.
7. **Surfaces** the retrieved context, the routing decision, and per-query latency in both a CLI and a Streamlit chat UI.

Nothing leaves your machine. No API keys are required.

---

## Repository layout

```
.
├── README.md                  # this file — install + run instructions
├── product_prd.md             # product requirements document
├── recommendation.md          # production-deployment recommendations
├── requirements.txt           # Python dependencies
├── .gitignore
├── src/
│   ├── config.py              # central configuration (env-overridable)
│   ├── entities.py            # 20 people + 20 places catalogue
│   ├── ingest.py              # MediaWiki API client + raw JSON dumper
│   ├── chunker.py             # sentence-aware sliding-window chunker
│   ├── embeddings.py          # Ollama + sentence-transformers backends
│   ├── vector_store.py        # SQLite metadata + Chroma vector store
│   ├── retriever.py           # query router + filtered retrieval
│   ├── generator.py           # Ollama HTTP generation (sync + streaming)
│   └── rag_pipeline.py        # IngestionPipeline + RAGEngine
├── app/
│   ├── cli.py                 # interactive CLI chat
│   └── streamlit_app.py       # Streamlit chat UI
├── scripts/
│   ├── run_ingest.py          # fetch + chunk + embed + persist
│   ├── reset_system.py        # wipe Chroma + SQLite
│   └── smoke_test.py          # offline integrity check (no Ollama needed)
└── data/
    └── raw/                   # raw Wikipedia JSON dumps (gitignored)
```

---

## Prerequisites

- **Python 3.10–3.12 recommended** (tested with 3.11/3.12; newer Python versions may need matching `torch` wheels).
- **Ollama** running locally on port `11434` — the standard default.
  Download: <https://ollama.com/download>.
- ~3 GB of free disk space for the LLM weights and ~1 GB for Chroma + SQLite data.

> No external services are used. The system runs without internet *after* ingestion, except for Wikipedia fetches you trigger explicitly.

---

## 1. Install Python dependencies

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The dependency list is intentionally small: `requests`, `chromadb`, `streamlit`, and `sentence-transformers` (used only if you opt out of Ollama embeddings).

---

## 2. Install and start Ollama

1. Install Ollama from <https://ollama.com/download>.
2. In a separate terminal, start the daemon (the desktop app launches it automatically):
   ```bash
   ollama serve
   ```
3. Pull the embedding model and the LLM:
   ```bash
   ollama pull nomic-embed-text     # 274 MB, used for chunk embeddings
   ollama pull llama3.2:3b          # 2.0 GB, used for answer generation
   ```
   You can substitute `phi3` or `mistral` for `llama3.2:3b` if you prefer — see "Configuration" below.

Sanity check:

```bash
curl http://localhost:11434/api/tags
```

You should see both models listed.

---

## 3. Ingest data

Run a single command to fetch every entity, chunk it, embed it, and persist it:

```bash
python scripts/run_ingest.py
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--reset` | Wipe Chroma vectors and SQLite metadata before ingesting. |
| `--only "Albert Einstein,Eiffel Tower"` | Ingest a subset. Quote the comma list. |
| `--batch-size N` | Embedding batch size; default 16. |

Re-running ingestion without `--reset` is idempotent — documents are upserted by `entity_name` and chunks are replaced, so changing the chunker and re-running just re-indexes the same articles.

A typical run on a laptop with `nomic-embed-text` finishes in **~3–5 minutes** for the full 40-entity catalogue.

You can verify the corpus without the LLM:

```bash
python scripts/smoke_test.py        # router + one Wikipedia fetch + temp SQLite write
python -m app.cli --stats           # prints corpus stats from SQLite + Chroma
```

---

## 4. Run the chat interface

### Option A — Streamlit UI (recommended for the demo)

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (typically <http://localhost:8501>). Features:

- Streaming token-by-token answers.
- Expandable "Retrieved context" panel showing the top-K chunks, their similarity score, the section they came from, and the source URL.
- "Show route decision" — see why the system chose `person`, `place`, or both.
- Sidebar buttons for **Clear chat** and **Reset index**.
- Quick-pick example questions matching the homework brief.

### Option B — CLI

```bash
python -m app.cli                 # interactive chat
python -m app.cli --ask "Who was Ada Lovelace?"
python -m app.cli --ask "Compare Lionel Messi and Cristiano Ronaldo." --show-sources
python -m app.cli --stats         # corpus stats
python -m app.cli --reset --force # wipe everything (no prompt)
```

In-chat slash commands:

```
:sources    toggle showing retrieved context
:stats      print corpus stats
:history    print this session's transcript
:clear      clear the on-screen transcript
:reset      wipe the index (asks for confirmation)
:help       show help
:exit       leave
```

---

## Example queries

**People**
- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Nikola Tesla famous?
- Compare Lionel Messi and Cristiano Ronaldo.
- What is Frida Kahlo known for?

**Places**
- Where is the Eiffel Tower located?
- Why is the Great Wall of China important?
- What is Machu Picchu?
- What was the Colosseum used for?
- Where is Mount Everest?

**Mixed / failure cases**
- Which famous place is located in Turkey? *(Hagia Sophia and Topkapı Palace are in the corpus.)*
- Which person is associated with electricity? *(Tesla.)*
- Compare Albert Einstein and Nikola Tesla.
- Compare the Eiffel Tower and the Statue of Liberty.
- Who is the president of Mars? *(Should refuse with "I don't know".)*
- Tell me about a random unknown person John Doe. *(Same — refuse.)*

---

## Configuration

Every setting in [`src/config.py`](src/config.py) is overridable via environment variables. The most useful:

| Variable | Default | Notes |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon URL. |
| `RAG_LLM_MODEL` | `llama3.2:3b` | Try `phi3`, `mistral`, `qwen2.5:3b`, etc. |
| `RAG_EMBED_BACKEND` | `ollama` | `sentence-transformers` to use a locally cached Hugging Face model. |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Any Ollama embed model when backend is `ollama`. |
| `RAG_CHUNK_SIZE` | `800` | Target chunk size in characters. |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap in characters. |
| `RAG_TOP_K` | `5` | Default retrieval depth. |
| `RAG_LLM_TEMPERATURE` | `0.2` | Lower = more grounded, less creative. |

Example — switch to Phi-3 for a lighter run:

```bash
export RAG_LLM_MODEL=phi3
ollama pull phi3
streamlit run app/streamlit_app.py
```

---

## How retrieval works (short version)

1. **Route** the query against the catalogue. We tokenise the query and look for entity names, common aliases (`messi`, `ataturk`, `cr7`, `pyramids`, …), and "shape" hints (`who`/`where`). The router decides `target_types ∈ {person, place, both}` and lists detected entities.
2. **Filter + search**. We embed the query once with the same model used at ingestion time, then run a single Chroma query with `where={"type": <target>}` if the route chose one type. For two-entity comparison questions (e.g. "Compare Messi and Ronaldo") we issue one filtered query per entity and merge by score.
3. **Boost** chunks whose `entity_name` matches a recognised entity — this prevents semantically similar but off-topic chunks from displacing the correct ones.
4. **Generate**. The grounded prompt (see `src/generator.py`) hands the model numbered context blocks and tells it to cite them. If retrieval returns nothing, the engine short-circuits to `"I don't know."` without calling the LLM.

The full design rationale lives in [`recommendation.md`](recommendation.md).

---

## Troubleshooting

**`Ollama is not reachable at http://localhost:11434/...`**
The daemon isn't running. Start it with `ollama serve`, or open the desktop app.

**`Ollama returned no embedding for model 'nomic-embed-text'`**
You haven't pulled the model. Run `ollama pull nomic-embed-text`.

**Streamlit shows old answers after re-ingesting.**
Hit **Reset index** in the sidebar (or `python scripts/reset_system.py`) and re-run ingestion. The Streamlit cache also resets when the app reloads.

**Corpus stats look wrong (`vectors=0`).**
Did you run `python scripts/run_ingest.py`? The vector store is intentionally empty until ingestion completes.

**Wikipedia fetch fails with HTTP 4xx.**
Wikipedia article titles are case-sensitive. If you see a 404 for an entity you just added, double-check `wiki_title` in [`src/entities.py`](src/entities.py) on `en.wikipedia.org`.

---

## Demo video

[Watch the 5-minute walkthrough on YouTube (unlisted)](https://youtu.be/0cD4cYTlSbU). It covers the system overview, live ingestion, live Q&A on people / places / mixed / failure cases, the routing decision, and the trade-offs. The same link is pinned at the top of [`recommendation.md`](recommendation.md).

---

## Licence

MIT — provided solely for the BLG 483E homework. Wikipedia content is CC BY-SA 3.0.
