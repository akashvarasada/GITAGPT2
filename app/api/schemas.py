"""Request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel


class ConfigRequest(BaseModel):
    provider: str = "local"           # "local" | "gemini"
    api_key: str | None = None        # required only for gemini
    model: str | None = None          # chosen local model tag (local only)


class ChatRequest(BaseModel):
    question: str


class GarakGenerateRequest(BaseModel):
    """Body garak's REST generator POSTs to /garak/generate."""
    prompt: str = ""


class GarakRunRequest(BaseModel):
    """Kick off a garak scan against a chosen provider/model."""
    provider: str = "gemini"          # "local" | "gemini"
    api_key: str | None = None        # required for gemini
    model: str | None = None          # model tag/name (local tag or gemini model)
    mode: str = "raw"                 # "raw" (LLM only) | "rag" (full pipeline)
    probes: list[str] = []            # garak probe specs; empty -> promptinject
    generations: int = 1              # attempts per probe prompt
