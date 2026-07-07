---
name: build-rag-app
description: Playbook for building or extending a RAG application, distilled from building GITAGPT — tech stack choices, chunking/retrieval/vector-store design, evaluation methodology, and the full performance optimization checklist. Use when building a new RAG app/feature, adding a document type or corpus, choosing embeddings/vector store/retrieval strategy, or wanting performance built in from day one instead of bolted on later.
---

# Building a RAG application — playbook from GITAGPT

Distilled from building GITAGPT (RAG over `Docs/Bhagavad-gita-As-It-Is.pdf`, LangChain/LangGraph + FastAPI).
Full rationale: [architecture.md](../../../architecture.md); build spec: [plan.md](../../../plan.md);
performance deep-dive + live diagnosis workflow: [PERFORMANCE.md](../../../PERFORMANCE.md) and the
`gitagpt-performance` skill.

## 1. Reference architecture

```
question → embed → hybrid retrieve (dense+BM25, k≈20, optional metadata filter) → rerank → top 5
        → LangGraph: grade → [rewrite query + re-retrieve if weak] → generate (grounded, cited) → stream

Ingestion (offline): PDF → parse → chunk-router → embed → persist to vector store
Serving: FastAPI (/config, /chat stream, /health, /ready) → Provider factory (LLM/embeddings) → RAG service
```

## 2. Tech stack decisions (and why)

| Layer | Choice | Why |
|---|---|---|
| Parse | `pymupdf4llm` (→Markdown); `unstructured`+OCR for scanned/messy PDFs | structure-preserving beats raw text extraction |
| Chunk | **Router**, not one strategy | rigid docs (verse/section headers) chunk best structurally; prose needs recursive/semantic splitting — one strategy cuts content mid-thought |
| Embeddings | Strong model, pluggable (`BAAI/bge-m3` local default; `gemini-embedding-001`/`voyage-3-large` hosted opt-in) | `all-MiniLM-L6-v2`-tier models are prototype-only; multilingual models handle transliteration/diacritics better |
| Reranker | Cross-encoder (`bge-reranker-base`) on top of a bi-encoder | a good bi-encoder + reranker beats a fancier embedder alone — invest here, not in embedder swaps |
| Vector store | Pluggable; Chroma default (embedded, zero-ops, rich metadata filters); quantized store (e.g. turbovec) only past millions of vectors | don't pay quantization's recall-approximation cost until scale demands it |
| Retrieval | Hybrid dense+BM25 (`EnsembleRetriever`) | keyword match rescues domain-specific terms (proper nouns, jargon) that dense embeddings under-rank |
| Orchestration | LangGraph CRAG (`retrieve→grade→[rewrite/retry]→generate→grounding_check`) | plain chains hallucinate on grounded QA; self-correction catches weak retrieval before generation |
| LLM | Pluggable factory (hosted + local) | lets users trade privacy (local) vs quality/speed (hosted) without an architecture rewrite |
| Eval | RAGAS against a hand-built golden set | "vibes" don't catch retrieval regressions; need objective numbers before/after any pipeline change |

## 3. Build order (checklist)

1. **Scaffold** — project structure, `requirements.txt`, `config.py` (pydantic-settings, sensible defaults so the app runs with zero `.env`), `.env.example`.
2. **Providers** — `llm_factory.get_llm(provider, api_key, model)` and `embedding_factory.get_embeddings(provider, api_key)`. Keep every provider behind a factory from day one, not after the second provider is requested.
3. **Vector store abstraction** — one interface (`add`, `similarity_search`, `as_retriever`, `persist`); pick the concrete backend via config, not by hardcoding imports elsewhere.
4. **Ingestion** — `parse → chunk_router → build_index` as a standalone CLI (`python -m app.ingest.build_index`), restart-safe, persists to disk. Every chunk carries generic metadata (`source, doc_type, title, section, page`) plus optional domain-specific extras (`chapter, verse, type`) — rich citations where structure exists, graceful degradation elsewhere.
5. **Retrieval + graph** — `retriever.py` (hybrid + optional metadata filter) → `reranker.py` (retrieve N → rerank → top-k) → `graph.py` (CRAG) → `prompts.py` (answer only from context, cite sources, explicit refusal when unsupported).
6. **API + UI** — streamed responses (SSE) for responsiveness; runtime provider/model switch; secrets (API keys) kept in server memory only, never logged or written to disk.
7. **Eval** — golden set (~20-30 question/answer/citation triples) **built early** — it decides every downstream chunking/embedding/retriever choice, not bolted on at the end.
8. **Performance** — apply the checklist in §5 *while building*, not after users complain. Warm-up, device pinning, and keep-alive are cheap to add at construction time and expensive to retrofit.

## 4. Non-obvious lessons (apply these before you rediscover them)

- **Embedding model used to build the index MUST equal the one used at query time** — different models produce incompatible vector spaces. Stamp the embed-model name into the store's metadata (or a sidecar file) and guard it at query time; changing `EMBED_MODEL` requires a full re-index, not a hot swap.
- **Reranking is the highest-leverage retrieval investment**, ahead of chasing a fancier embedder. Fix retrieval (measure Context Precision/Recall) *before* touching prompts.
- **E5/BGE-family embedders need `"query:"`/`"passage:"` prefixes and normalized vectors for cosine similarity** — both are silent recall-killers if skipped, with no error to warn you.
- **Chunking is a router, not a strategy.** Detect document structure (rigid/headed/prose/scanned) and pick a splitter per type; forcing one strategy across document types degrades whichever type it wasn't tuned for.
- **Corpus separation matters once you add a second document.** One collection + a `doc_type`/`source` filter is simplest for related topics; separate collections + query routing is safer once topics are unrelated (mixing causes retrieval drift).
- **A CRAG-style grade→rewrite→generate→grounding-check graph** is worth the extra complexity specifically for grounded/citation-required QA — plain retrieval-then-generate chains hallucinate more when the initial retrieval is weak.
- **Build the golden eval set before optimizing anything.** Every chunking/embedding/retriever/model change should be measured against it, not judged by spot-checking a few queries.

## 5. Performance optimization checklist (bake in from the start)

Full measured before/after numbers and code: [PERFORMANCE.md](../../../PERFORMANCE.md). The golden rule
behind all of it: **measure first, then change** — every knob below must default to old behavior, never
silently trade retrieval/answer quality for speed.

1. **Pin the reranker + embedding model to the fastest available device** via one shared helper
   (`cuda` > `mps` > `cpu`, overridable by env var), used everywhere a model is loaded — never let a
   library default silently place a cross-encoder on CPU when a GPU/MPS device is available.
2. **Instrument every pipeline sub-stage independently** (dense retrieve, keyword retrieve, rerank, generate)
   with wall-clock timing from day one. An opaque "retrieval = 24s" number tells you nothing; a breakdown
   immediately shows which stage to optimize (usually reranking).
3. **Keep the local LLM warm** (`keep_alive`-style setting) — the default eviction-between-requests behavior
   of most local model servers causes a full reload (seconds to tens of seconds) on the very next query.
4. **Warm up models in a background thread at startup, never in the request path.** Gate readiness behind
   a `/ready`-style endpoint the UI polls, so the server is reachable immediately instead of refusing
   connections or making the first real user pay the model-load cost.
5. **Gate any expensive retry/rewrite loop behind an explicit on/off switch.** Self-correction loops
   (rewrite-and-retry) add a blocking LLM call plus a second retrieval pass — great for quality, easy to
   forget as a latency multiplier on outlier queries. Default it on, but make it toggleable for measurement.
6. **Right-size context window and cap output tokens.** An oversized context window wastes memory/setup
   time for no benefit; capping generated tokens is often the single biggest *code-side* lever against
   pure hardware-bound decode speed.
7. **Harden hosted-LLM calls**: lower default retry counts (library defaults are often tuned for
   correctness, not latency — a retry storm can dominate wall-clock time), relax overly-aggressive safety
   filters that cause false refusals on legitimate domain content, and cap "thinking"/reasoning budget on
   models that support it if time-to-first-token matters more than deliberation depth.
8. **Capture token usage and timing uniformly across every provider**, not just the one that reports it by
   default. Some providers attach usage to streamed chunks; others only expose it via a callback — normalize
   both into the same log line so tok/s and cost are comparable across providers.

**Know the ceiling before chasing it further:** compute speedup (GPU/MPS) only helps stages that actually
run on that hardware (typically embedding + reranking); pure CPU-bound LLM decode speed is a hardware floor
that code changes cannot lower — the only real levers there are capping output length, switching to a
hosted/GPU-backed model, or using a smaller local model.

## 6. Evaluation methodology

| Stage | Metric | Tool |
|---|---|---|
| Embeddings/retrieval | Recall@k / Hit@k, Context Precision/Recall | RAGAS |
| Generation | Faithfulness, Answer Relevancy | RAGAS |
| End-to-end | Quality + latency across providers | RAGAS + tracing (e.g. LangSmith) |

Re-run the eval after **any** change to chunking, embeddings, retriever, or model — objective numbers, not
vibes, decide whether a change shipped.

## 7. Gotchas worth remembering

- Verify exact model tags/names with the provider's own listing command (e.g. `ollama list`) rather than
  assuming a name from memory or docs — model naming schemes change often enough that a guessed name
  either fails outright or silently resolves to a different, similarly-named model.
- A source document's cosmetic extraction artifacts (e.g. encoding/font issues that garble non-ASCII
  characters) don't always need fixing — check whether the text you actually embed and answer from
  (translation/body text) is affected before treating it as a bug.
- Keep a short "corrections log" of wrong assumptions caught during design (wrong model names, wrong
  library capabilities, chunking strategies that seemed fine until a second document type arrived) — it
  prevents re-making the same wrong assumption in a later session.
