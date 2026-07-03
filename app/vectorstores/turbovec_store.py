"""Turbovec backend (scale option) -- stub.

turbovec (https://github.com/RyanCodrai/turbovec) shines at millions of vectors
via TurboQuant quantization. For a single book Chroma is the right default, so
this is intentionally a stub that documents the integration points. Flesh out
when the corpus grows large; it must satisfy the same VectorStore interface.
"""
from __future__ import annotations

from langchain_core.documents import Document


class TurbovecStore:
    def __init__(self, embeddings):
        raise NotImplementedError(
            "Turbovec backend not implemented yet. Use VECTOR_BACKEND=chroma. "
            "See architecture.md 3.4 -- adopt turbovec when scaling to a large "
            "multi-text corpus."
        )

    def add(self, docs: list[Document]) -> None: ...
    def similarity_search(self, query: str, k: int, filter: dict | None = None): ...
    def as_retriever(self, k: int, filter: dict | None = None): ...
    def all_documents(self) -> list[Document]: ...
    def count(self) -> int: ...
