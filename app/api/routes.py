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

from app.api.schemas import (
    ChatRequest,
    ConfigRequest,
    GarakGenerateRequest,
    GarakRunRequest,
)
from config import ROOT, settings

router = APIRouter()
templates = Jinja2Templates(directory=str(ROOT / "app" / "ui" / "templates"))


def _get_rag(request: Request):
    """Return the cached RagService. Normally built by the startup warm-up thread;
    if a request beats it (or warm-up was skipped), build under a lock so we never
    construct two copies. Blocks until the in-flight warm-up finishes."""
    app = request.app
    rag = getattr(app.state, "rag", None)
    if rag is not None:
        return rag
    lock = getattr(app.state, "rag_lock", None)
    if lock is None:
        import threading
        lock = app.state.rag_lock = threading.Lock()
    with lock:
        if getattr(app.state, "rag", None) is None:
            from app.rag.service import RagService

            rag = RagService()
            rag.warmup()
            app.state.rag = rag
            app.state.rag_ready = True
        return app.state.rag


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


@router.get("/ready")
def ready(request: Request):
    """UI polls this to know when the warm-up thread has models loaded."""
    st = request.app.state
    return {"ready": bool(getattr(st, "rag_ready", False)),
            "error": getattr(st, "rag_error", None)}


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


# --------------------------------------------------------------------------- #
# Garak security scanner                                                      #
# --------------------------------------------------------------------------- #
@router.get("/garak")
def garak_page(request: Request):
    from app.garak.runner import PROBE_CATALOG, garak_available
    from app.providers.llm_factory import list_ollama_models

    return templates.TemplateResponse(request, "garak.html", {
        "ollama_models": list_ollama_models(),
        "gemini_display": settings.gemini_display,
        "gemini_model": settings.gemini_model,
        "probes": PROBE_CATALOG,
        "garak_available": garak_available(),
    })


@router.post("/garak/generate")
def garak_generate(req: GarakGenerateRequest, request: Request):
    """Target endpoint garak's REST generator POSTs each attack prompt to.

    Routes through the same get_llm()/RagService the app uses, driven by the
    target config stashed in app.state when the scan started. Errors are
    returned as text (not raised) so garak records them as normal outputs.
    """
    target = getattr(request.app.state, "garak_target", None) or {}
    provider = target.get("provider")
    api_key = target.get("api_key")
    model = target.get("model")
    mode = target.get("mode", "raw")
    prompt = req.prompt or ""

    try:
        if mode == "rag":
            rag = _get_rag(request)
            out = rag.answer(prompt, provider=provider, api_key=api_key, model=model)
            text = out.get("answer", "")
        else:
            from langchain_core.messages import HumanMessage

            from app.providers.llm_factory import get_llm

            llm = get_llm(provider, api_key, model)
            text = llm.invoke([HumanMessage(content=prompt)]).text
    except Exception as exc:  # noqa: BLE001
        text = f"[generation error: {exc}]"
    return {"output": text}


@router.post("/garak/run")
def garak_run(req: GarakRunRequest, request: Request):
    if req.provider == "gemini" and not (req.api_key or settings.google_api_key):
        def err():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Gemini requires an API key.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    # One scan at a time: stash the target so /garak/generate knows how to answer.
    request.app.state.garak_target = {
        "mode": req.mode, "provider": req.provider,
        "api_key": req.api_key, "model": req.model,
    }
    label = req.model or (settings.gemini_display if req.provider == "gemini" else req.provider)

    def event_stream():
        from app.garak.runner import run_scan

        try:
            for event in run_scan(req.probes, req.generations, f"{label} ({req.mode})"):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
