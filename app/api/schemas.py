"""Request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel


class ConfigRequest(BaseModel):
    provider: str = "local"           # "local" | "gemini"
    api_key: str | None = None        # required only for gemini
    model: str | None = None          # chosen local model tag (local only)


class ChatRequest(BaseModel):
    question: str
