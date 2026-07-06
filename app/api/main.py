"""FastAPI entrypoint.  Run:  uvicorn app.api.main:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build + warm the RAG service at startup so the FIRST user query doesn't eat
    # the index load + model warm-up (embeddings, reranker, MPS kernel compile).
    # Fail soft: if the index isn't built yet, leave lazy init in routes to surface
    # a clear error on the first /chat instead of blocking server startup.
    try:
        from app.rag.service import RagService

        rag = RagService()
        rag.warmup()
        app.state.rag = rag
        print("[startup] RAG service warmed and ready.", flush=True)
    except Exception as exc:
        print(f"[startup] RAG warm-up skipped ({exc}); will init lazily.", flush=True)
    yield


app = FastAPI(title="GITAGPT",
              description="RAG over the Bhagavad-gita As It Is",
              lifespan=lifespan)
app.include_router(router)
