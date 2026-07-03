"""Central configuration. Values come from environment / .env file.

Everything has a sensible default for the fully-local setup (bge-m3 + Ollama +
Chroma), so the app runs with no .env at all. The UI can override the LLM /
embedding provider at runtime without touching these defaults.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = folder containing this file.
ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    llm_provider: str = "local"          # "local" | "gemini"
    ollama_model: str = "gemma4:12b-mlx"  # model name used in the Ollama API in mac
    ollama_display: str = "Gemma 4"      # human-readable label shown in the UI
    ollama_base_url: str = "http://localhost:11434"
    gemini_model: str = "gemini-3-flash-preview"
    gemini_display: str = "3 Flash Preview"
    google_api_key: str = ""

    # --- Embeddings ---
    embed_provider: str = "local"        # "local" | "gemini" | "voyage"
    embed_model: str = "BAAI/bge-m3"
    voyage_api_key: str = ""

    # --- Vector store ---
    vector_backend: str = "chroma"       # "chroma" | "turbovec"
    chroma_dir: str = "storage/chroma"

    # --- Retrieval ---
    reranker_model: str = "BAAI/bge-reranker-base"
    retrieve_k: int = 20                 # candidates fetched before rerank
    top_k: int = 5                       # docs kept after rerank
    # CRAG: if the best rerank score is below this, rewrite the query and retry once.
    rerank_relevance_threshold: float = -3.0

    # --- Docs ---
    docs_dir: str = "Docs"

    # Absolute-path helpers -------------------------------------------------
    @property
    def chroma_path(self) -> Path:
        return ROOT / self.chroma_dir

    @property
    def docs_path(self) -> Path:
        return ROOT / self.docs_dir


settings = Settings()
