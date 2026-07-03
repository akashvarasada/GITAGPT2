# GITAGPT — Build Spec

RAG app over `Docs/Bhagavad-gita-As-It-Is.pdf`. Python + LangChain/LangGraph, FastAPI UI, pluggable
LLM/embeddings (**Gemini 3** cloud OR **Ollama Gemma** local). Reasoning & trade-offs: see [architecture.md](architecture.md).

## Tech Stack (final)
- **Parse:** `pymupdf4llm` (→Markdown); `unstructured`(+OCR `pytesseract`) for unstructured/scanned PDFs.
- **Chunk:** strategy router — verse-aware (Gita) / layout / recursive / semantic. `langchain-text-splitters`, `langchain-experimental`.
- **Embeddings (pluggable):** **default local `BAAI/bge-m3`** (normalize + query/passage prefixes); hosted `gemini-embedding-001` / `voyage-3-large` as opt-in. NOT MiniLM.
- **Vector store (pluggable):** `Chroma` (default, persistent) via `langchain-chroma`; `turbovec` backend as scale option.
- **Retrieval:** hybrid dense+BM25 (`EnsembleRetriever`, `rank_bm25`) + cross-encoder rerank (`bge-reranker-base`).
- **Orchestration:** `langgraph` (CRAG: retrieve→grade→[rewrite/retry]→generate→grounding-check).
- **LLM (pluggable):** Gemini `gemini-3-flash-preview` / `gemini-3.1-pro-preview` (`langchain-google-genai`); local Gemma via `langchain-ollama` `ChatOllama`.
- **Eval:** `ragas` (+ optional `langsmith`).
- **UI/API:** `fastapi`, `uvicorn`, `jinja2`, `sse-starlette`, `pydantic-settings`, `python-dotenv`.

## Project Structure
```
GITAGPT/
├─ Docs/                      # source PDFs
├─ app/
│  ├─ ingest/                 # parse.py, chunk_router.py, build_index.py
│  ├─ rag/                    # retriever.py, reranker.py, graph.py, prompts.py
│  ├─ providers/              # llm_factory.py, embedding_factory.py
│  ├─ vectorstores/           # base.py, chroma_store.py, turbovec_store.py
│  ├─ api/                    # main.py, routes.py, schemas.py
│  └─ ui/                     # templates/, static/
├─ eval/                      # golden_set.jsonl, evaluate.py
├─ storage/                   # persisted index
├─ config.py                  # pydantic-settings
├─ .env.example
└─ requirements.txt
```

## Build Steps (in order)

**0. Scaffold.** Create structure above, `requirements.txt`, `config.py` (pydantic-settings), `.env.example`
(`GOOGLE_API_KEY=`, `VOYAGE_API_KEY=`, `EMBED_PROVIDER=local`, `EMBED_MODEL=BAAI/bge-m3`, `VECTOR_BACKEND=chroma`, `OLLAMA_MODEL=gemma3`).

**1. Providers.**
- `providers/llm_factory.py` → `get_llm(provider, api_key, model)`: `gemini`→`ChatGoogleGenerativeAI`, `local`→`ChatOllama`.
- `providers/embedding_factory.py` → `get_embeddings(provider, api_key)`: `gemini`→`GoogleGenerativeAIEmbeddings(gemini-embedding-001)`, `voyage`→`VoyageAIEmbeddings(voyage-3-large)`, `local`→`HuggingFaceEmbeddings(BAAI/bge-m3, normalize_embeddings=True)`.
- Rule: index & query embeddings must match — store embed-model name in collection metadata.

**2. Vector store abstraction.**
- `vectorstores/base.py` interface (`add`, `similarity_search`, `as_retriever`, `persist`).
- `chroma_store.py` (default, persist to `storage/`); `turbovec_store.py` (optional, same interface).

**3. Ingestion.**
- `ingest/parse.py`: PDF→Markdown (`pymupdf4llm`); OCR path for scanned.
- `ingest/chunk_router.py`: pick splitter by doc type. Gita = split on `TEXT/TRANSLATION/PURPORT` headers, one chunk per verse-unit, long purports via recursive splitter. **Metadata every chunk:** `source, doc_type, title, section, page` + optional `chapter, verse, type`.
- `ingest/build_index.py`: parse→chunk→embed→store. CLI: `python -m app.ingest.build_index`.

**4. Retrieval + graph.**
- `rag/retriever.py`: `EnsembleRetriever`(dense+BM25), optional metadata filter (chapter/doc_type).
- `rag/reranker.py`: cross-encoder rerank 20→5.
- `rag/prompts.py`: answer ONLY from context; cite `BG chapter.verse`; say "not addressed in the provided text" if unsupported.
- `rag/graph.py`: LangGraph `retrieve→grade_docs→[rewrite+retry]→generate→grounding_check`.

**5. API + UI.**
- `api/main.py`+`routes.py`: `POST /config` (provider, api_key, model — kept in server memory, never logged/persisted), `POST /chat` (streamed answer+citations via `sse-starlette`), `GET /health`.
- `ui/templates`: chat page with provider dropdown (Local | Gemini), API-key field (shown only for Gemini), model dropdown, chat box, answer panel rendering `BG x.y` citations.

**6. Eval.**
- `eval/golden_set.jsonl` (~20–30 question/answer/verse triples).
- `eval/evaluate.py`: RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall). Run vs both providers.

## Acceptance Criteria
- `build_index` produces a persisted store; restart-safe.
- Every chunk has required metadata; no verse split from its purport.
- `/chat` answers with correct `BG x.y` citations and refuses out-of-scope questions.
- Provider/model switch works at runtime from the UI (Gemini with key, local Ollama without).
- Answers stream; API key never written to disk or logs.
- RAGAS runs and reports metrics for both providers.

## Confirm before building
- Exact Ollama Gemma tag (`ollama list`); Gemini embedding model name in AI Studio (only if switching to hosted later).
- Corpus strategy if more PDFs added: one collection + `doc_type` filter vs. per-corpus collections (see architecture.md §Corpus).
