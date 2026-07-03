"""Vector-store interface.

Keeps the rest of the app independent of the concrete backend. Chroma is the
default; turbovec is a scale option. `get_store(...)` is the single entry point.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document

from config import settings


class VectorStore(ABC):
    """Minimal surface the ingestion + retrieval layers depend on."""

    @abstractmethod
    def add(self, docs: list[Document]) -> None:
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int,
                          filter: dict | None = None) -> list[Document]:
        ...

    @abstractmethod
    def as_retriever(self, k: int, filter: dict | None = None):
        """Return a LangChain retriever for use in ensembles/chains."""
        ...

    @abstractmethod
    def all_documents(self) -> list[Document]:
        """Every stored document -- used to build the BM25 keyword index."""
        ...

    @abstractmethod
    def count(self) -> int:
        ...


def get_store(embeddings, backend: str | None = None) -> VectorStore:
    """Factory: return the configured vector store bound to `embeddings`."""
    backend = (backend or settings.vector_backend).lower()

    if backend == "chroma":
        from app.vectorstores.chroma_store import ChromaStore
        return ChromaStore(embeddings)

    if backend == "turbovec":
        from app.vectorstores.turbovec_store import TurbovecStore
        return TurbovecStore(embeddings)

    raise ValueError(f"Unknown vector backend: {backend!r} (use 'chroma' or 'turbovec').")
