# Production Deployment Recommendations

> **Demo video:** [Watch on YouTube (unlisted)](https://youtu.be/0cD4cYTlSbU)
> Length: ~5 minutes. Outline at the bottom of this file.

This document complements `README.md` (how to run) and `product_prd.md` (what we built and why) by answering one question:

> *If we had to take this exact system to production, what would we change, and why?*

The recommendations are grouped by layer. Each item lists the **trade-off the homework version makes**, the **production posture**, and the **expected effort** to migrate.

---

## 1. Ingestion

### 1.1 Move from synchronous fetch to a scheduled crawler
- **Today.** `scripts/run_ingest.py` runs once and pulls articles serially with a 0.4 s sleep.
- **Production.** Run a scheduled job (e.g. Airflow / Prefect / a simple GitHub Actions workflow on cron) that:
  - polls Wikipedia's `recentchanges` API or the Page Stream service so we only re-fetch articles that actually changed;
  - records each fetch in an immutable `ingestion_runs` table for auditability;
  - emits structured logs to a SIEM-compatible sink (JSON to stdout is enough).
- **Effort.** ~1 day. The `WikipediaIngestor` already has `User-Agent` plumbing and accepts a custom session — wrapping it in a scheduler is mechanical.

### 1.2 Validate ingested text before embedding
- **Today.** We accept whatever Wikipedia returns.
- **Production.** Add a content gate that rejects articles below a length threshold or above a "boilerplate" ratio (e.g. > 60 % infobox or list markup). Quarantine them for human review instead of silently embedding noise.
- **Effort.** Half a day, plus a small dashboard.

### 1.3 Use the `mwparserfromhell` AST instead of plain text
- **Today.** We rely on `explaintext=1` which strips all formatting, including section nesting and the lede sentence boundary.
- **Production.** Parse the wikitext into an AST so we can keep semantic markers (lede vs. section vs. infobox) and feed them to the chunker as features.
- **Effort.** 1–2 days. The chunker's section detection becomes much more robust.

---

## 2. Chunking

### 2.1 Calibrate chunk size against the embedding model
- **Today.** 800-character target with 150-character overlap, hand-picked.
- **Production.** Sweep `(chunk_size, overlap)` against a labelled retrieval evaluation set (Recall@k, MRR, Hit@1) and lock the winning combo per embedding model. Re-run when the embedding model changes.
- **Effort.** 1 day to build the eval harness; a few hours per sweep.

### 2.2 Add a semantic chunker as an alternative
- **Today.** Sentence-aware sliding window — fast, deterministic, but blind to topic shifts within a long section.
- **Production.** Try a "semantic chunker" that splits whenever the cosine distance between consecutive sentence embeddings exceeds a learned threshold. This produces fewer, denser chunks and tends to lift Recall@1 by 5–10 % on long biographical articles.
- **Effort.** Half a day, behind a config flag (`RAG_CHUNKER=semantic`).

---

## 3. Embeddings

### 3.1 Pick the embedding model on intent, not on convenience
- **Today.** `nomic-embed-text` (768-dim) because Ollama serves it for free.
- **Production.** Evaluate `bge-large-en-v1.5`, `gte-large`, and `text-embedding-3-large` (when an external API is permitted) on the same retrieval eval set. For a Wikipedia/biographical corpus, `bge-large-en-v1.5` typically wins by 3–5 points of Recall@5.
- **Effort.** 1 day per candidate, including swap-in via `RAG_EMBED_MODEL`.

### 3.2 Cache embeddings keyed by content hash
- **Today.** Re-embedding is cheap because the corpus is small.
- **Production.** Compute SHA-256 of the chunk text and key embeddings by hash. Skips ~99 % of recomputation when only metadata changes (title, URL).
- **Effort.** 2–3 hours.

### 3.3 Batch sized for the actual hardware
- **Today.** Batch size 16, optimised for CPU.
- **Production.** Detect GPU at startup and lift the batch to 64–128. The Embedder class already accepts a list, so the only change is in `IngestionPipeline._embed_and_store`.

---

## 4. Vector store

### 4.1 Keep one collection — but harden its schema
- **Today.** Single Chroma collection with `type` metadata (Option B). This works for 40 entities.
- **Production.** Continue with Option B but add: `language`, `country`, `last_modified`, `chunker_version`, `embedding_model`. Tag every vector with `embedding_model` so a model change becomes a metadata-filtered upgrade rather than a full rebuild.
- **Effort.** ~1 day, mostly migration code.

### 4.2 Move off Chroma when scale demands it
- **Today.** Chroma's persistent client comfortably handles tens of thousands of vectors on a laptop.
- **Production.** At ≥ 1 M vectors or under multi-tenant load, switch to **Qdrant** (best-of-class HNSW + payload filters, easy single-binary deploy) or **pgvector** if the rest of the stack is already on Postgres. Keep the `VectorStore` interface stable — that is precisely what the abstraction in `src/vector_store.py` exists for.
- **Effort.** 2–3 days for Qdrant; 1 day for pgvector.

### 4.3 Promote SQLite to a content-addressed object store
- **Today.** SQLite holds canonical text + chunk provenance.
- **Production.** Move article bodies to S3/GCS keyed by `sha256(article)` and keep only metadata + pointers in Postgres. SQLite stays only for fast local development.

---

## 5. Retrieval

### 5.1 Replace the keyword router with a small classifier
- **Today.** Keyword + alias match against the catalogue, plus "shape" hints (`who`/`where`).
- **Production.** Train a small (≤ 100 MB) intent classifier on logged queries. Outputs `{type, confidence, entities}`. Falls back to the keyword router when confidence is low — never produces worse routing than today.
- **Effort.** 2–3 days once we have ≥ a few hundred logged queries.

### 5.2 Add a cross-encoder reranker
- **Today.** Cosine similarity + entity boost.
- **Production.** Send the top-20 candidates from Chroma to `bge-reranker-base` (110 MB, runs on CPU at ~50 docs/s) and keep the top-5. Typically lifts answer quality measurably on long-tail questions.
- **Effort.** Half a day.

### 5.3 Hybrid retrieval (BM25 + vectors)
- **Today.** Pure vector search.
- **Production.** Combine BM25 (built into SQLite via FTS5) with vector scores using Reciprocal Rank Fusion. Catches the "exact phrase wins" cases vector search routinely misses (proper nouns, dates, rare terms).
- **Effort.** 1 day.

---

## 6. Generation

### 6.1 Larger model behind a router
- **Today.** `llama3.2:3b` for everything.
- **Production.** Default to `llama3.2:3b`. Route queries flagged as `is_comparison=True` or with > 8 retrieved chunks to a larger model (e.g. `llama3.1:8b`) — those are the queries where reasoning over multiple chunks matters most. Keep simple lookups on the small model to preserve latency.
- **Effort.** 2 hours.

### 6.2 Strict citation post-processing
- **Today.** The prompt asks for `[1]` style citations but doesn't enforce them.
- **Production.** Parse the answer, verify each cited number actually exists in the prompt, and refuse the answer if it cites a non-existent source. Prevents the most common kind of hallucination.
- **Effort.** Half a day.

### 6.3 Guardrails for refusal
- **Today.** "I don't know" is enforced by a strict prompt plus a deterministic guard: zero-hit retrieval, or no meaningful lexical overlap for generic no-entity questions, short-circuits before the LLM is called.
- **Production.** Add a calibrated score threshold: if the highest retrieval score is below a learned floor (e.g. cosine score < 0.15), short-circuit to "I don't know" without calling the LLM. This should be selected from an evaluation set rather than guessed.
- **Effort.** A few hours, plus eval to pick the threshold.

---

## 7. Observability and evaluation

### 7.1 Per-query traces
- Log every query as a structured event: `{query, route, chunk_ids,
  scores, model, latency_ms, answer_chars, refusal}`. Send to ClickHouse
  or even a local DuckDB file. Build a Grafana / Metabase dashboard for
  latency p50/p95, refusal rate, and "answers without citations".

### 7.2 Continuous evaluation
- Maintain a labelled set of ~200 question/answer pairs. Run the full
  pipeline against it on every change and gate releases on Recall@5 +
  exact-match.

### 7.3 Red-teaming
- Add a recurring job that submits known failure cases ("president of
  Mars", "John Doe") and asserts the system refuses. Today this is a
  manual `--ask` invocation; tomorrow it should be CI.

---

## 8. Security and privacy

- **Network egress is currently zero outside Wikipedia + Ollama.** Keep it that way: pin the Ollama host, deny outbound traffic from the worker except to whitelisted domains.
- **Don't log the user's raw query.** Hash it for analytics; store the raw text only when the user explicitly opts in.
- **Vector store metadata is PII-light today** but a production system might add fields like `submitted_by`. Encrypt them at rest.
- **Model files are large** — sign and verify them before loading; pull from your own mirror.

---

## 9. Cost & latency budget for production

| Layer | Today (laptop) | Production target | Notes |
| ----- | -------------- | ----------------- | ----- |
| Embedding (per chunk) | ~30 ms CPU | ≤ 5 ms GPU | batch 64 on a T4 |
| Chroma query | ~10 ms | ≤ 5 ms | Qdrant on a c7g.large keeps this |
| LLM generation | ~3–6 s CPU | ≤ 1.5 s | 8B model on an L4 |
| End-to-end p95 | ~8 s | ≤ 2 s | dominated by generation |

These numbers are educated estimates and should be re-measured against
the real corpus before any commitments are made.

---

## 10. Demo video — recommended outline (≤ 5 min)

| Time | Section | What to show |
| ---- | ------- | ------------- |
| 0:00–0:30 | **Pitch** | "A local, ChatGPT-style RAG over 40 Wikipedia articles." Show the README's repository layout. |
| 0:30–1:30 | **Live ingestion** | Run `python scripts/run_ingest.py --only "Albert Einstein,Eiffel Tower"` and narrate fetch → chunk → embed → persist. |
| 1:30–3:00 | **Live Q&A** | In the Streamlit UI, ask: a person question, a place question, a comparison, the "in Turkey" mixed question, and a failure case. For each, show the routing decision and the retrieved chunks. |
| 3:00–4:00 | **Architecture** | Walk through `src/rag_pipeline.py` and the design choice of one Chroma collection + metadata filtering. |
| 4:00–5:00 | **Trade-offs** | Cite three items from this document: keyword router → classifier, no reranker → cross-encoder, refusal threshold. End with "what I would do next". |

---

*If you are reviewing this for a grade — thank you. Run `streamlit run app/streamlit_app.py` after the README's setup steps; everything in §10 is reproducible from a clean checkout.*
