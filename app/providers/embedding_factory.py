"""Pluggable embedding factory.

Default is local BAAI/bge-m3 (multilingual, near-SOTA, fully offline). Gemini
and Voyage are opt-in. Whatever builds the index MUST also serve queries -- see
`embedding_id()`, which we stamp into the vector collection so a mismatch is
caught early instead of silently returning garbage.

Note on prefixes: bge-m3 does NOT need "query:"/"passage:" instruction prefixes
(unlike bge-*-v1.5 / e5). We only normalize embeddings for cosine similarity.
"""
from __future__ import annotations

from config import settings


def get_embeddings(provider: str | None = None,
                   api_key: str | None = None,
                   model: str | None = None):
    """Return a LangChain embeddings object."""
    provider = (provider or settings.embed_provider).lower()

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        from app.providers.device import get_device

        # Pin the device (mps on Apple Silicon, cuda if present, else cpu) instead
        # of relying on auto-detection -- same weights + math, just faster hardware.
        return HuggingFaceEmbeddings(
            model_name=model or settings.embed_model,
            model_kwargs={"device": get_device()},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        key = api_key or settings.google_api_key
        if not key:
            raise ValueError("Gemini embeddings selected but no API key provided.")
        return GoogleGenerativeAIEmbeddings(
            model=model or "models/gemini-embedding-001",
            google_api_key=key,
        )

    if provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        key = api_key or settings.voyage_api_key
        if not key:
            raise ValueError("Voyage embeddings selected but no API key provided.")
        return VoyageAIEmbeddings(model=model or "voyage-3-large", api_key=key)

    raise ValueError(f"Unknown embed provider: {provider!r}.")


def embedding_id(provider: str | None = None, model: str | None = None) -> str:
    """Stable identifier stamped into the collection to guard against mixing
    embeddings from different models at index vs query time."""
    provider = (provider or settings.embed_provider).lower()
    if provider == "local":
        model = model or settings.embed_model
    elif provider == "gemini":
        model = model or "gemini-embedding-001"
    elif provider == "voyage":
        model = model or "voyage-3-large"
    return f"{provider}:{model}"
