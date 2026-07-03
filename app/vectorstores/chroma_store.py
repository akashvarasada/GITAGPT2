"""Chroma-backed vector store (default). Persists to `settings.chroma_dir`."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings

COLLECTION = "gitagpt"


class ChromaStore:
    def __init__(self, embeddings):
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._db = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_path),
        )

    def add(self, docs: list[Document]) -> None:
        # Chroma metadata values must be scalars; drop None to avoid errors.
        for d in docs:
            d.metadata = {k: v for k, v in d.metadata.items() if v is not None}
        self._db.add_documents(docs)

    def similarity_search(self, query: str, k: int,
                          filter: dict | None = None) -> list[Document]:
        return self._db.similarity_search(query, k=k, filter=filter)

    def as_retriever(self, k: int, filter: dict | None = None):
        kwargs: dict = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return self._db.as_retriever(search_kwargs=kwargs)

    def all_documents(self) -> list[Document]:
        raw = self._db.get(include=["documents", "metadatas"])
        return [
            Document(page_content=txt, metadata=meta or {})
            for txt, meta in zip(raw["documents"], raw["metadatas"])
        ]

    def count(self) -> int:
        return self._db._collection.count()
