"""FastAPI entrypoint.  Run:  uvicorn app.api.main:app --reload"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router


def _warm(app: FastAPI) -> None:
    """Build + warm the RAG service (models, MPS kernels). Runs in a background
    thread so the server accepts connections immediately; the UI polls /ready."""
    try:
        from app.rag.service import RagService

        with app.state.rag_lock:
            if app.state.rag is None:
                rag = RagService()
                rag.warmup()
                app.state.rag = rag
        app.state.rag_ready = True
        print("[startup] RAG service warmed and ready.", flush=True)
    except Exception as exc:  # surfaced to the UI via /ready
        app.state.rag_error = str(exc)
        print(f"[startup] RAG warm-up failed: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = None
    app.state.rag_ready = False
    app.state.rag_error = None
    app.state.rag_lock = threading.Lock()
    # Non-blocking warm-up: server is reachable at once, models load in background.
    threading.Thread(target=_warm, args=(app,), daemon=True).start()
    yield


app = FastAPI(title="GITAGPT",
              description="RAG over the Bhagavad-gita As It Is",
              lifespan=lifespan)
app.include_router(router)
