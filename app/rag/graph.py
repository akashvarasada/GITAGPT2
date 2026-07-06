"""LangGraph retrieval brain (CRAG-lite).

Flow:  retrieve -> grade -> (rewrite -> retrieve)? -> finalize

Grading is based on the cross-encoder's top score rather than an extra LLM call:
it's cheaper, deterministic, and easy to debug (scores are inspectable). If the
best candidate scores below the threshold, we rewrite the query with the LLM and
retry retrieval once. The graph outputs the selected documents + a formatted,
citable context string; generation/streaming happens in service.py.
"""
from __future__ import annotations

import time
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.rag import prompts
from app.rag.reranker import rerank
from app.rag.retriever import HybridRetriever
from config import settings


class RagState(TypedDict, total=False):
    question: str
    query: str                 # current search query (may be rewritten)
    filter: dict | None
    documents: list[Document]
    context: str
    tries: int
    relevant: bool
    timings: dict              # per-stage seconds (dense/bm25/rerank), for profiling

    # NOTE: the API key is deliberately NOT part of the state. State becomes the
    # traced `inputs` of every step (local AND cloud tracers), so putting secrets
    # here leaks them. The LLM is injected via config["configurable"]["llm"].


def build_graph(retriever: HybridRetriever):
    """Compile the retrieval graph bound to a HybridRetriever."""

    def retrieve(state: RagState) -> RagState:
        timings: dict = {}
        candidates = retriever.retrieve(state["query"], filter=state.get("filter"),
                                        timings=timings)
        t0 = time.perf_counter()
        scored = rerank(state["query"], candidates, settings.top_k)
        timings["rerank_s"] = round(time.perf_counter() - t0, 3)
        timings["candidates"] = len(candidates)
        # A rewrite retry runs this node twice; sum so `timings` reflects total work.
        prev = state.get("timings") or {}
        for key in ("dense_s", "bm25_s", "rerank_s"):
            timings[key] = round(prev.get(key, 0) + timings.get(key, 0), 3)
        best = scored[0][1] if scored else float("-inf")
        return {
            "documents": [d for d, _ in scored],
            "relevant": best >= settings.rerank_relevance_threshold,
            "timings": timings,
        }

    def decide(state: RagState) -> str:
        # enable_rewrite off -> never fire the blocking in-graph LLM rewrite.
        if (not settings.enable_rewrite
                or state.get("relevant")
                or state.get("tries", 0) >= 1):
            return "finalize"
        return "rewrite"

    def rewrite(state: RagState, config) -> RagState:
        # LLM comes from config (not state) so the API key never enters traced inputs.
        llm = config["configurable"]["llm"]
        msg = llm.invoke([
            SystemMessage(content=prompts.REWRITE_SYSTEM),
            HumanMessage(content=state["question"]),
        ])
        return {"query": msg.text.strip(), "tries": state.get("tries", 0) + 1}

    def finalize(state: RagState) -> RagState:
        return {"context": prompts.format_context(state.get("documents", []))}

    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite)
    g.add_node("finalize", finalize)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", decide,
                            {"rewrite": "rewrite", "finalize": "finalize"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("finalize", END)
    return g.compile()
