"""Hybrid retrieval: dense (vector) + BM25 (keyword), unioned then reranked.

Keyword matching matters here because Sanskrit terms (dharma, karma, atma) are
rare tokens a dense model can smear together. We build BM25 once over the whole
corpus and reuse it; per-query metadata filtering is applied to both arms.
"""
from __future__ import annotations

import time

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import settings


class HybridRetriever:
    def __init__(self, store):
        self._store = store
        self._all_docs = store.all_documents()
        self._bm25 = BM25Retriever.from_documents(self._all_docs)
        self._bm25.k = settings.retrieve_k

    def retrieve(self, query: str, filter: dict | None = None,
                 timings: dict | None = None) -> list[Document]:
        # `timings` (optional): filled with per-arm seconds for profiling. The
        # dense arm's time includes the query embedding (Chroma embeds internally).
        k = settings.retrieve_k
        t0 = time.perf_counter()
        dense = self._store.similarity_search(query, k=k, filter=filter)
        t1 = time.perf_counter()
        keyword = self._bm25.invoke(query)
        if filter:
            keyword = [d for d in keyword if _matches(d, filter)]
        t2 = time.perf_counter()
        if timings is not None:
            timings["dense_s"] = round(t1 - t0, 3)
            timings["bm25_s"] = round(t2 - t1, 3)
        return _dedup(dense + keyword)


def _matches(doc: Document, filter: dict) -> bool:
    return all(doc.metadata.get(key) == val for key, val in filter.items())


def _dedup(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("reference"), d.metadata.get("part"), d.page_content[:60])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out
