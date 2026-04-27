# Product Requirements Document — Local Wikipedia RAG

**Course:** BLG 483E — Homework 3
**Owner:** Mustafa Eren Koç
**Status:** v1.0 — released for grading
**Last updated:** 2026-04-27

---

## 1. Problem statement

Course participants need a hands-on artefact that demonstrates a full
Retrieval-Augmented-Generation pipeline running with **only local
resources** — no third-party LLM API, no managed vector database. The
artefact must combine the indexing/retrieval ideas from Project 1 with
the AI-driven workflow design from Project 2 into a single, runnable
ChatGPT-style assistant.

The product addresses two questions an instructor asks during grading:

1. *Does the candidate understand how RAG actually works end-to-end?*
2. *Can the candidate make principled engineering trade-offs and
   explain them?*

## 2. Goals

| Goal | Why it matters | How we measure it |
| ---- | --------------- | ----------------- |
| G1 — Run entirely on a laptop | No-API requirement | `pip install -r requirements.txt && streamlit run …` works offline after model pulls |
| G2 — Answer the canonical example questions | Grading rubric | Each question in §6 returns a grounded answer; failure cases reply "I don't know" |
| G3 — Show *why* an answer was produced | Demonstrates understanding | UI exposes routing rationale + retrieved chunks |
| G4 — Be re-runnable from `README.md` alone | Instructor reproducibility | A clean checkout reaches a working chat in ≤ 10 commands |
| G5 — Stay near "language native" implementations | Brief explicitly says so | No LangChain / llama-index; raw HTTP for Wikipedia + Ollama; raw `chromadb` |

## 3. Non-goals

- Multi-user authentication, billing, or rate limiting.
- Production-grade monitoring or alerting.
- Cross-lingual ingestion (English Wikipedia only).
- Image / audio / table understanding within Wikipedia articles.
- Long-running background fine-tuning.

## 4. Personas

| Persona | Need |
| ------- | ---- |
| **Course instructor** | Verify functionality and architectural sensibility in under 15 minutes |
| **Classmate / future student** | Read source code to understand RAG without wading through framework wrappers |
| **Prospective production engineer** | Read `recommendation.md` to know what would change for a real deployment |

## 5. User stories

1. *As an instructor*, I clone the repo, follow the README, and within
   ten minutes I am asking the system questions in a browser.
2. *As an instructor*, when I ask "Who is the president of Mars?" the
   system answers exactly **"I don't know"** and shows me that no
   useful chunks were retrieved.
3. *As an instructor*, when I ask **"Which famous place is located in
   Turkey?"** the system identifies it as a place query, surfaces the
   Hagia Sophia and Topkapı Palace chunks, and answers from those
   chunks.
4. *As a classmate*, I open `src/retriever.py` and can read the routing
   logic top-to-bottom in five minutes.
5. *As a future maintainer*, I can swap `llama3.2:3b` for `phi3` by
   exporting an environment variable — no code changes.

## 6. Functional requirements

### 6.1 Ingestion

- **FR-1.1** Fetch Wikipedia articles for **at least 20 famous people
  and 20 famous places**, including the required minimum set (Einstein,
  Curie, da Vinci, Shakespeare, Lovelace, Tesla, Messi, Ronaldo, Swift,
  Kahlo / Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon,
  Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of
  Giza, Mount Everest).
- **FR-1.2** Use the Wikipedia MediaWiki Action API directly (no
  wrapper library that performs the core work for us).
- **FR-1.3** Store the raw plain-text article on disk for inspection
  (`data/raw/<entity>.json`).
- **FR-1.4** Be polite to Wikipedia: identifying `User-Agent`, ≥ 0.4 s
  delay between requests.

### 6.2 Chunking

- **FR-2.1** Split each article into chunks. Default: sentence-aware
  sliding window with **800-character target** and **150-character
  overlap**.
- **FR-2.2** Track `(entity_name, type, title, url, section,
  chunk_index, char_start, char_end)` for every chunk so the UI can
  cite sources.
- **FR-2.3** Configuration parameters live in `src/config.py` and are
  overridable via environment variables.

### 6.3 Embeddings + storage

- **FR-3.1** Generate embeddings locally — default `nomic-embed-text`
  via Ollama.
- **FR-3.2** Provide a fallback to `sentence-transformers/all-MiniLM-L6-v2`
  via env var so the project still works without Ollama embeddings.
- **FR-3.3** Persist embeddings in **a single Chroma collection with
  `type` metadata** (Option B from the brief). The reasoning is in §10.
- **FR-3.4** Persist canonical text + provenance in SQLite so the
  vector store can be rebuilt from local data without re-fetching
  Wikipedia.

### 6.4 Retrieval

- **FR-4.1** Determine whether a query targets `person`, `place`, or
  both, using a keyword/lexicon classifier seeded with the entity
  catalogue and "shape" hints (`who`, `where`, …).
- **FR-4.2** Apply Chroma metadata filtering when the type is decided.
- **FR-4.3** Detect comparison questions involving two named entities
  and run one filtered query per entity to ensure both sides have
  context.
- **FR-4.4** Boost chunks whose `entity_name` matches a recognised
  entity to prevent off-topic semantic neighbours from winning.

### 6.5 Generation

- **FR-5.1** Call a local Ollama LLM (default `llama3.2:3b`) via the
  bare `/api/generate` endpoint — no SDK wrappers.
- **FR-5.2** Prompt the model with a strict "answer only from context"
  instruction.
- **FR-5.3** When zero chunks are retrieved, short-circuit to
  `"I don't know"` without calling the LLM.
- **FR-5.4** Support a non-streaming and a streaming code path so both
  the CLI and the Streamlit UI feel responsive.

### 6.6 User interface

- **FR-6.1** Provide a CLI chat with slash commands (`:sources`,
  `:stats`, `:reset`, `:history`, `:clear`, `:help`, `:exit`).
- **FR-6.2** Provide a Streamlit chat with streaming answers,
  expandable retrieved-context viewer, routing-decision caption, and
  Clear chat / Reset index buttons.
- **FR-6.3** Both surfaces show per-query latency and the chosen model.

## 7. Non-functional requirements

| ID | Requirement |
| --- | ---------- |
| NFR-1 | Cold-start ingestion of 40 entities completes in < 8 min on a 2023-class laptop with `nomic-embed-text`. |
| NFR-2 | Single-question latency end-to-end (retrieval + generation) < 6 s with `llama3.2:3b` on CPU. |
| NFR-3 | Re-running ingestion is idempotent (upsert by `entity_name`). |
| NFR-4 | The repository runs offline once models are pulled — only Wikipedia fetches require internet. |
| NFR-5 | No third-party SaaS or managed service is used at any layer. |
| NFR-6 | All persistent state lives under the repository (`./data/`, `./chroma_db/`). |
| NFR-7 | Project runs on Windows 11, macOS, and Linux without code changes. |

## 8. Out-of-scope decisions captured

- **No LangChain / llama-index.** The brief says "to the greatest extent possible please use language native functionality rather than fully featured libraries that do the core work of the exercise out of the box." We use raw `requests`, raw `chromadb`, and raw `sqlite3`.
- **No `wikipedia` Python wrapper.** Same reasoning — we drive the MediaWiki API directly.
- **No re-ranker model.** A cross-encoder reranker (e.g. `bge-reranker-base`) would lift relevance further; called out as a recommended future improvement in `recommendation.md`.
- **No conversational memory.** Each question is independent; chat history is for display only. Memory is listed as an optional extension and is intentionally deferred.

## 9. Success metrics for the demo video

| Metric | Target |
| ------ | ------ |
| Canonical questions answered correctly | ≥ 80 % |
| "I don't know" returned for failure cases | 100 % |
| Retrieved chunks visibly correspond to the answer | true on every shown question |
| Response latency on a 3B model, CPU only | < 8 s typical |
| Time from `git clone` to first answer (instructor flow) | < 10 min |

## 10. Design decision: single collection vs. two collections

We chose **Option B — one Chroma collection with `type` metadata** for
four reasons:

1. **Mixed queries are first-class.** The brief lists "Which famous
   place is located in Turkey?" and "Compare Albert Einstein and
   Nikola Tesla." Single-collection means the retriever can run one
   filtered query for unambiguous cases and one unfiltered query for
   ambiguous ones. Two collections force a fan-out + merge step that
   we would have to write and tune.
2. **Cheaper to maintain.** A single collection means a single index,
   a single embedding-dimension constraint, and a single place to
   apply schema changes (e.g. adding a `country` field later).
3. **Filtering is essentially free in Chroma.** `where={"type":
   "person"}` is a metadata predicate evaluated on the candidate
   set — there's no cost to embedding-space comparison.
4. **Identical performance for type-restricted queries.** With cosine
   distance and HNSW, restricting via metadata is comparable in
   latency to using a smaller collection.

The trade-off — slightly worse cache locality vs. two collections — is
negligible at 40 documents. The decision and its reasoning live in code
comments at the top of `src/vector_store.py`.

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
| ---- | ---------- | ---------- |
| Ollama not installed on grader's laptop | Medium | README has an explicit install step; smoke test runs without Ollama |
| Wikipedia title drift (rename / disambiguation) | Low | `wiki_title` is decoupled from display name in `entities.py`; redirects are followed |
| LLM hallucination | Medium | Strict system prompt + zero-context short-circuit |
| 3B model gives weak comparison answers | Medium | Comparison route ensures both entities contribute chunks |
| Chunker drops headings | Low | Section-aware splitter preserves `==Heading==` boundaries |

## 12. Acceptance checklist (for the instructor)

- [ ] Cloning the repo and following `README.md` produces a working
      chat without additional guidance.
- [ ] `python scripts/smoke_test.py` passes.
- [ ] After `python scripts/run_ingest.py`, `python -m app.cli --stats`
      shows ≥ 40 documents and ≥ 200 chunks.
- [ ] Each canonical example question returns a coherent, sourced
      answer.
- [ ] Failure cases ("president of Mars", "John Doe") return
      "I don't know".
- [ ] Resetting the index and re-ingesting completes successfully.

---

*This PRD is delivered alongside the source code. The companion
`recommendation.md` covers what would change for a production
deployment.*
