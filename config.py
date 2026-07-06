"""Central configuration. Values come from environment / .env file.

Everything has a sensible default for the fully-local setup (bge-m3 + Ollama +
Chroma), so the app runs with no .env at all. The UI can override the LLM /
embedding provider at runtime without touching these defaults.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = folder containing this file.
ROOT = Path(__file__).resolve().parent

# pydantic-settings below only parses .env for its OWN declared fields below --
# it does not push values into the real process environment. Some libraries
# (e.g. LangSmith tracing) read os.environ directly, so we load .env for real too.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    llm_provider: str = "local"          # "local" | "gemini"
    ollama_model: str = "auto"           # "auto" = pick an installed Ollama model
    ollama_base_url: str = "http://localhost:11434"
    # How long Ollama keeps the model resident after a request. Default (5m) lets
    # it get evicted between queries -> a cold reload (seen as ollama_model_load_s
    # of 1.5-36s) on the next one. Keep it warm to eliminate that reload cost.
    ollama_keep_alive: str = "30m"
    # Optional Ollama generation controls (None = use the server's own default):
    #   num_ctx     -- context window. Right-size to prompt+answer; an oversized
    #                  window (e.g. 16k for ~2.5k-token prompts) wastes KV-cache
    #                  memory and slows prefill setup.
    #   num_predict -- hard cap on generated tokens (safety net against runaway
    #                  answers; each token costs ~0.3s on CPU).
    ollama_num_ctx: int | None = None
    ollama_num_predict: int | None = None

    gemini_model: str = "gemini-3-flash-preview"
    gemini_display: str = "3 Flash Preview"
    google_api_key: str = ""
    # Gemini hardening (targets the observed 150s "retry storm" + wrong fallback):
    #   max_retries      -- LangChain default is 6; a storm of backoff retries can
    #                       balloon latency. 2 fails fast instead of hanging.
    #   safety           -- "relaxed" sets the main harm categories to BLOCK_NONE so
    #                       Gita content (war, killing kinsmen) isn't falsely refused;
    #                       "default" leaves Google's defaults.
    #   thinking_budget  -- None = model default. gemini-3-flash is a thinking model;
    #                       unbounded thinking inflates time-to-first-token. Set a
    #                       small int (e.g. 0 or 512) to cap it.
    #   max_output_tokens-- None = model default; cap to bound answer length.
    gemini_max_retries: int = 2
    gemini_safety: str = "relaxed"       # "relaxed" | "default"
    gemini_thinking_budget: int | None = None
    gemini_max_output_tokens: int | None = None

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
    # Master switch for the CRAG rewrite retry. It fires a *blocking* LLM call
    # inside the retrieval graph (+ a second retrieve), so on a slow local model it
    # can dominate LangGraph latency. Keep on for quality; turn off to measure/trim.
    enable_rewrite: bool = True

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
