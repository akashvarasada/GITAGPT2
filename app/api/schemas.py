"""Request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel


class ConfigRequest(BaseModel):
    provider: str = "local"           # "local" | "gemini"
    api_key: str | None = None        # required only for gemini


class ChatRequest(BaseModel):
    question: str
