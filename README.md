# GITAGPT

RAG application over **Bhagavad-gita As It Is**. Python + LangChain/LangGraph, a
FastAPI chat UI, and a pluggable LLM/embedding layer (local **Ollama Gemma** or
**Gemini 3**). Design details: [plan.md](plan.md) · [architecture.md](architecture.md).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env                                 # optional; defaults are local-only
```

Defaults are fully local: `bge-m3` embeddings + Chroma + Ollama Gemma. Nothing
in `.env` is required unless you switch to Gemini/Voyage.

### Local LLM (default)
Install [Ollama](https://ollama.com), then pull a Gemma model and confirm the tag
matches `OLLAMA_MODEL` in `.env` (default: `gemma4:12b-mlx` — check with `ollama list`):
```bash
ollama pull gemma4:12b-mlx
```

### Gemini (optional)
Set `GOOGLE_API_KEY` in `.env`, or just paste the key in the UI and pick "Gemini".

## Build the index

```bash
python -m app.ingest.build_index --dry-run     # inspect chunks, no embedding
python -m app.ingest.build_index               # embed + persist to storage/
```

> The embedding model is stamped into `storage/embedding_id.txt`. If you change
> `EMBED_MODEL`/`EMBED_PROVIDER`, re-run build_index (vector spaces aren't compatible).

## Run

```bash
uvicorn app.api.main:app --reload --reload-dir app
# open http://127.0.0.1:8000
```

> `--reload-dir app` matters: without it, uvicorn watches the whole project including
> `storage/` (Chroma's on-disk index), which gets touched on every query and would
> otherwise retrigger a full server restart (and a slow model reload) on each request.

Pick the LLM (Local / Gemini), paste a key if using Gemini, ask a question.
Answers are grounded in the text and cite verses as `(BG chapter.verse)`.

## Evaluate

```bash
pip install ragas
python -m eval.evaluate --limit 3
```

Reports retrieval Hit@k plus RAGAS Faithfulness / Answer Relevancy / Context
Precision / Context Recall. Re-run after changing chunking, embeddings, or model.

## Layout
```
app/ingest   parse + chunk-router + build_index
app/rag      retriever (hybrid) + reranker + LangGraph (CRAG) + service
app/providers  llm_factory + embedding_factory (pluggable)
app/vectorstores  base + chroma (default) + turbovec (scale stub)
app/api      FastAPI routes + main
app/ui       Jinja2 chat UI
eval         golden_set.jsonl + evaluate.py
```
