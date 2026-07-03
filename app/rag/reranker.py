"""Cross-encoder reranker.

A bi-encoder (embeddings) is recall-oriented; a cross-encoder re-scores the
query against each candidate for precision. Model is loaded once and cached.
"""
from __future__ import annotations

from langchain_core.documents import Document

from config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(settings.reranker_model)
    return _model


def rerank(query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
    """Return the top_k (document, score) pairs, most relevant first."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = _get_model().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda ds: ds[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_k]]
