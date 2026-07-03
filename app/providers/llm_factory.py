"""Pluggable chat-LLM factory.

The rest of the app calls `get_llm(...)` and never imports a provider directly,
so swapping Gemini <-> local Ollama is a one-argument change (driven by the UI
or config). Imports are lazy so you only need the deps for the provider you use.
"""
from __future__ import annotations

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

        return ChatOllama(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r} (use 'local' or 'gemini').")
