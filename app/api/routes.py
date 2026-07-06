"""API routes: config, chat (streaming), health.

The chosen LLM provider/key/model live in server memory only (app.state) and are
never logged or written to disk. Embeddings are fixed by config (bge-m3) and are
independent of the LLM choice, so the RAG index loads once and is reused.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from app.api.schemas import ChatRequest, ConfigRequest
from config import ROOT, settings

router = APIRouter()
templates = Jinja2Templates(directory=str(ROOT / "app" / "ui" / "templates"))


def _get_rag(request: Request):
    """Lazily build and cache the RagService (loads embeddings + index)."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        from app.rag.service import RagService

        rag = RagService()
        request.app.state.rag = rag
    return rag


def _get_config(request: Request) -> dict:
    return getattr(request.app.state, "config",
                   {"provider": settings.llm_provider, "api_key": None})


@router.get("/")
def index(request: Request):
    from app.providers.llm_factory import list_ollama_models

    return templates.TemplateResponse(request, "index.html", {
        "ollama_models": list_ollama_models(),       # all installed local models
        "gemini_display": settings.gemini_display,
    })


@router.get("/health")
def health():
    return {"status": "ok", "embed": settings.embed_model, "backend": settings.vector_backend}


@router.post("/config")
def set_config(cfg: ConfigRequest, request: Request):
    if cfg.provider == "gemini" and not cfg.api_key:
        return {"ok": False, "error": "Gemini requires an API key."}
    request.app.state.config = cfg.model_dump()
    # Never echo the key back.
    from app.providers.llm_factory import describe_ollama_model

    if cfg.provider == "gemini":
        model = settings.gemini_display
    else:
        model = cfg.model or describe_ollama_model()
    return {"ok": True, "provider": cfg.provider, "model": model}


@router.post("/chat")
def chat(req: ChatRequest, request: Request):
    cfg = _get_config(request)

    def event_stream():
        try:
            rag = _get_rag(request)
            for event in rag.stream(
                req.question,
                provider=cfg.get("provider"),
                api_key=cfg.get("api_key"),
                model=cfg.get("model"),
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # surface errors to the UI instead of hanging
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
