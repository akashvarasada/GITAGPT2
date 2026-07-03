"""High-level RAG service used by both the API and the eval harness.

Loads the index + retriever once, runs the LangGraph retrieval brain per query,
then generates a grounded answer (blocking or streamed) with the chosen LLM.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.providers.embedding_factory import embedding_id, get_embeddings
from app.rag import prompts
from app.rag.graph import build_graph
from app.rag.retriever import HybridRetriever
from app.vectorstores.base import get_store
from config import ROOT, settings


class RagService:
    def __init__(self):
        self._check_embedding_match()
        embeddings = get_embeddings()
        store = get_store(embeddings)
        if store.count() == 0:
            raise RuntimeError(
                "Vector index is empty. Run: python -m app.ingest.build_index"
            )
        self._retriever = HybridRetriever(store)
        self._graph = build_graph(self._retriever)

    # -- retrieval + prompt assembly ---------------------------------------
    def prepare(self, question, provider=None, api_key=None, model=None, filter=None):
        state = self._graph.invoke({
            "question": question,
            "query": question,
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "filter": filter,
            "tries": 0,
        })
        docs = state.get("documents", [])
        messages = [
            SystemMessage(content=prompts.ANSWER_SYSTEM),
            HumanMessage(content=prompts.ANSWER_USER.format(
                context=state.get("context", ""), question=question)),
        ]
        return messages, docs

    # -- generation --------------------------------------------------------
    def answer(self, question, provider=None, api_key=None, model=None, filter=None):
        from app.providers.llm_factory import get_llm

        messages, docs = self.prepare(question, provider, api_key, model, filter)
        llm = get_llm(provider, api_key, model)
        text = llm.invoke(messages).text
        return {
            "answer": text,
            "sources": _sources(docs),
            "contexts": [d.page_content for d in docs],   # for RAGAS
        }

    def stream(self, question, provider=None, api_key=None, model=None, filter=None):
        """Yield answer text chunks, then a final dict with sources."""
        from app.providers.llm_factory import get_llm

        messages, docs = self.prepare(question, provider, api_key, model, filter)
        llm = get_llm(provider, api_key, model)
        for chunk in llm.stream(messages):
            # chunk.content can be a plain string or a list of content blocks
            # (LangChain 1.x); .text normalizes either shape to plain text.
            text = chunk.text
            if text:
                yield {"type": "token", "text": text}
        yield {"type": "sources", "sources": _sources(docs)}

    # -- guards ------------------------------------------------------------
    @staticmethod
    def _check_embedding_match():
        marker = ROOT / "storage" / "embedding_id.txt"
        if marker.exists():
            indexed = marker.read_text(encoding="utf-8").strip()
            current = embedding_id()
            if indexed != current:
                raise RuntimeError(
                    f"Embedding mismatch: index built with '{indexed}' but query "
                    f"uses '{current}'. Re-run build_index or restore the provider."
                )


def _sources(docs) -> list[dict]:
    return [
        {"reference": d.metadata.get("reference"),
         "chapter": d.metadata.get("chapter"),
         "verse": d.metadata.get("verse"),
         "page": d.metadata.get("page")}
        for d in docs
    ]

