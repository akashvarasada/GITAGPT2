# GITAGPT — Architecture & Design Reasoning

Companion to [plan.md](plan.md). This holds the *why* behind each choice, trade-offs, diagrams, gotchas, and
evaluation methodology. Read this before making design changes.

---

## 1. Context

Source is **Bhagavad-gita As It Is** — a highly regular document: 18 chapters → verses, each with Sanskrit,
transliteration, word-by-word, **translation**, and **purport** (commentary). This structure is the single
biggest lever in the design: it lets us chunk by verse and produce exact `BG chapter.verse` citations. The
design must also stay correct when *less-structured* PDFs are added later — hence the pluggable/router choices.

---

## 2. Diagrams

### Component view
```
┌─────────────────────── FastAPI App ───────────────────────┐
│  /config   /chat (stream)   /health   UI (Jinja2)           │
└──────────────┬───────────────────────────┬────────────────┘
   provider+key│                           │question
               ▼                           ▼
     ┌───────────────────┐      ┌──────────────────────────┐
     │ Provider Factory   │◄─────┤  RAG Service (LangGraph)  │
     │ llm / embeddings   │      │ retrieve→grade→generate   │
     └──────┬─────────────┘      └────────────┬─────────────┘
 Gemini ◄───┤                                 │ query embedding
 Ollama ◄───┘                                 ▼
                              ┌────────────────────────────┐
                              │ Retriever: hybrid + rerank   │
                              └────────────┬───────────────┘
                                           ▼
                              ┌────────────────────────────┐
                              │ Vector Store (Chroma|turbovec)│
                              └────────────┬───────────────┘
                                           ▲ built offline
        ┌──────────────────────────────────┘
        │ Ingestion: PDF→parse→chunk-router→embed→index
        └───────────────────────────────────────────────
```

### Query flow
```
question → embed → hybrid retrieve (dense+BM25, k≈20, optional filter) → rerank →5
        → LangGraph grade→[rewrite/retry if weak]→generate (grounded, cited) → stream
```

---

## 3. Key Decisions & Rationale

### 3.1 LLM & embeddings are pluggable behind factories
The whole app is provider-agnostic; the UI picks Gemini or local Ollama at runtime. This satisfies the
"versatile" requirement and keeps privacy (local) vs quality (cloud) a user choice, not an architecture rewrite.

**Gemini models:** `gemini-3-flash-preview` (default, fast/cheap) and `gemini-3.1-pro-preview` (quality).
The models' **Jan 2025 knowledge cutoff is irrelevant** — RAG injects the Gita as context, so the model never
relies on training knowledge of the scripture. Cutoff only matters for closed-book questions, which this isn't.

**Local:** Ollama running Gemma. Verify the exact tag with `ollama list` (`gemma3` etc.).

**Hard rule:** the embedding model used to build the index MUST equal the one used at query time — different
models produce incompatible vector spaces. Store the embed-model name in collection metadata; switching
embeddings requires a re-index.

### 3.2 Embeddings: use a strong model, not MiniLM
"Sentence-transformers" spans mediocre→SOTA. `all-MiniLM-L6-v2` is prototype-grade only.

| Tier | Model | Why |
|---|---|---|
| Best hosted | `gemini-embedding-001`, `voyage-3-large` | Top MTEB, no GPU, aligns with Gemini |
| Best local | **`BAAI/bge-m3` ← current default** (multilingual → handles Sanskrit transliteration/diacritics), `gte-Qwen2-1.5B`, `mxbai-embed-large` | Near-SOTA |
| Fast local (CPU) | `bge-base-en-v1.5`, `nomic-embed-text` | Prototyping |
| Avoid | `all-MiniLM-L6-v2` | Weak retrieval |

**Current choice:** `BAAI/bge-m3` (local) — keeps the app fully offline/private by default; switch to hosted
`gemini-embedding-001` later only if eval shows a quality gap. Remember: switching embeddings requires a re-index.

**Insight:** a solid bi-encoder **+ a cross-encoder reranker** beats a fancier embedder alone — so invest in
reranking, don't over-optimize the embedder. **Gotchas:** E5/BGE need `"query:"`/`"passage:"` prefixes;
**normalize embeddings** for cosine. Both are silent recall-killers if missed.

### 3.3 Chunking is a ROUTER, not one strategy
Verse-per-chunk works *because* the Gita is rigid. For arbitrary PDFs, detect type and pick a splitter:

| Doc type | Strategy |
|---|---|
| Rigid (Gita) | Split on `TEXT/TRANSLATION/PURPORT`; one chunk per verse-unit; long purports → recursive |
| Headings/tables | Layout-aware (`unstructured`) → chunk by section |
| Flowing prose | `RecursiveCharacterTextSplitter` (~800–1000 tok, ~150 overlap) or semantic chunking |
| Scanned/image | OCR first (`pytesseract`/`unstructured`), then above |

- **Why not fixed-size everywhere:** it cuts purports/sections mid-thought → poor retrieval.
- **Why not semantic-only:** overkill for the Gita (author already chunked by verse) and can break verse↔purport link.
- **Parent-document retriever** is a good cross-type default: retrieve on small chunks, feed larger parent to LLM.

**Generalized metadata:** every chunk carries `source, doc_type, title, section, page`; `chapter/verse/type`
are *optional* extras the Gita fills. Result: rich citations where structure exists, graceful degradation elsewhere.

### 3.4 Vector store: Chroma default, turbovec for scale
| Option | Verdict |
|---|---|
| **Chroma** | Default — embedded, LangChain-native, persistent, rich metadata filtering, zero-ops |
| FAISS | Fast exact search, thin metadata |
| **turbovec** | Rust+Python on Google **TurboQuant** quantization, MIT, local-only, LangChain drop-in. **16× compression** wins at *millions* of vectors — but for one ~2 MB book, exact search is instant and quantization only adds recall approximation. Adopt when GITAGPT scales to a large multi-text corpus. |
| Qdrant | Local service; great filtering/quantization if you want a service |
| Vertex AI Vector Search | Managed cloud (ScaNN); overkill/cost for this scale |

Kept pluggable (`vectorstores/base.py`) so switching Chroma↔turbovec is one config line.

### 3.5 Corpus separation (multiple PDFs)
Mixing unrelated topics in one collection causes retrieval drift. Two patterns:
- **One collection + `doc_type`/`source` filter** — simplest; fine when topics are related.
- **Separate collections + query routing** — safer for unrelated docs; add a "search scope" selector in UI.

### 3.6 Retrieval is the highest-leverage stage
1. **Hybrid dense+BM25** — Sanskrit terms (dharma, karma, atma) benefit hugely from keyword match.
2. **Cross-encoder rerank** (`bge-reranker-base`): retrieve 20 → rerank → 5.
3. **Metadata filter** — scope to a chapter/doc.
Fix retrieval (measure Context Precision/Recall) *before* tuning prompts.

### 3.7 LangGraph over a plain chain
Scripture answers must not hallucinate. A **CRAG-style graph** —
`retrieve → grade_docs → (rewrite query + re-retrieve if weak) → generate → grounding_check` — adds
self-correction. Prompt forces answers strictly from context, `BG x.y` citations, and explicit refusal when
unsupported.

---

## 4. Evaluation Methodology
Build the **golden set early** (~20–30 question/answer/verse triples); it decides every downstream choice.

| Stage | Metric | Tool |
|---|---|---|
| Embeddings/retrieval | Recall@k / Hit@k, Context Precision/Recall | RAGAS |
| Generation | Faithfulness, Answer Relevancy | RAGAS |
| End-to-end | Compare Gemini vs local (quality + latency) | RAGAS + LangSmith trace |

Re-run `eval/evaluate.py` after any change to chunking, embeddings, retriever, or model. Objective numbers, not vibes.

---

## 5. Security & Ops Notes
- API key lives in **server session/memory only** — never logged, never written to disk. Defaults from `.env`.
- Stream responses (`sse-starlette`) for responsive chat.
- Local path (Ollama + local embeddings + Chroma) is **fully offline/private** — a genuine selling point.

---

## 6. Corrections Log (assumptions fixed during design)
- "Turbo Quant from Google" → real: **TurboQuant** quantization algo; **turbovec** is the index built on it.
- "Gemini 3.5" → doesn't exist; available models are `gemini-3-flash-preview` / `gemini-3.1-pro-preview`.
- MiniLM was an early "fast" pick; upgraded to `bge-m3`/hosted for real quality.
- "One chunk per verse" generalized to a chunking router once non-Gita PDFs entered scope.
