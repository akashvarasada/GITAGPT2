"""Pluggable chat-LLM factory.

The rest of the app calls `get_llm(...)` and never imports a provider directly,
so swapping Gemini <-> local Ollama is a one-argument change (driven by the UI
or config). Imports are lazy so you only need the deps for the provider you use.
"""
from __future__ import annotations

import json
import urllib.request

from config import settings


def get_llm(provider: str | None = None,
            api_key: str | None = None,
            model: str | None = None):
    """Return a LangChain chat model.

    provider: "local" (Ollama) or "gemini". Defaults to config.
    api_key:  Gemini API key (ignored for local).
    model:    override the model name.
    """
    provider = (provider or settings.llm_provider).lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key or settings.google_api_key
        if not key:
            raise ValueError("Gemini selected but no API key provided.")
        return ChatGoogleGenerativeAI(
            model=model or settings.gemini_model,
            google_api_key=key,
            temperature=0.2,
        )

    if provider == "local":
        from langchain_ollama import ChatOllama

        name = model or settings.ollama_model
        if not name or name.lower() == "auto":
            name = resolve_ollama_model()
        return ChatOllama(
            model=name,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r} (use 'local' or 'gemini').")


# --------------------------------------------------------------------------- #
# Ollama model auto-detection                                                 #
# --------------------------------------------------------------------------- #
# Ollama has no built-in "default model" concept, so when OLLAMA_MODEL is
# "auto" (or blank) we pick one from what's installed. This keeps the config
# portable -- the exact tag differs per machine (e.g. gemma4:e4b vs a Mac's
# gemma4:12b-mlx) without editing anything.

def _list_ollama_models(endpoint: str) -> list[str]:
    url = settings.ollama_base_url.rstrip("/") + endpoint
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.load(resp)
    return [m.get("name") or m.get("model") for m in data.get("models", [])]


def resolve_ollama_model() -> str:
    """Pick a model: prefer one already loaded (no cold start), else first installed."""
    for endpoint in ("/api/ps", "/api/tags"):   # ps = running, tags = installed
        try:
            models = _list_ollama_models(endpoint)
        except Exception:
            models = []
        if models:
            return models[0]
    raise ValueError(
        "No Ollama models available. Pull one first, e.g. `ollama pull gemma4:e4b`, "
        "or set OLLAMA_MODEL explicitly."
    )


def describe_ollama_model() -> str:
    """Human-friendly label for the UI. Never raises (used at page load)."""
    try:
        return resolve_ollama_model()
    except Exception:
        return "no model found"


def list_ollama_models() -> list[str]:
    """All installed Ollama model tags, for the UI dropdown. Never raises."""
    try:
        return _list_ollama_models("/api/tags")
    except Exception:
        return []
