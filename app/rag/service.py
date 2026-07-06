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

    def warmup(self) -> None:
        """Run one tiny retrieve+rerank to trigger model loads and MPS kernel
        compilation at startup, so the first real query doesn't pay that cost."""
        from app.rag.reranker import rerank

        docs = self._retriever.retrieve("warmup")
        rerank("warmup", docs[:5], 5)

    # -- retrieval + prompt assembly ---------------------------------------
    def prepare(self, question, llm, filter=None, callbacks=None):
        # The LLM is passed via config (for the rewrite node) -- NOT via state --
        # so the API key never becomes part of any tracer's recorded inputs.
        config = {"configurable": {"llm": llm}}
        if callbacks:
            config["callbacks"] = callbacks
        state = self._graph.invoke({
            "question": question,
            "query": question,
            "filter": filter,
            "tries": 0,
        }, config=config)
        docs = state.get("documents", [])
        messages = [
            SystemMessage(content=prompts.ANSWER_SYSTEM),
            HumanMessage(content=prompts.ANSWER_USER.format(
                context=state.get("context", ""), question=question)),
        ]
        return messages, docs, state.get("timings", {})

    # -- generation --------------------------------------------------------
    def answer(self, question, provider=None, api_key=None, model=None, filter=None):
        from app.providers.llm_factory import get_llm

        llm = get_llm(provider, api_key, model)
        messages, docs, _timings = self.prepare(question, llm, filter)
        text = llm.invoke(messages).text
        return {
            "answer": text,
            "sources": _sources(docs),
            "contexts": [d.page_content for d in docs],   # for RAGAS
        }

    def stream(self, question, provider=None, api_key=None, model=None, filter=None):
        """Yield answer text chunks, then a final dict with sources."""
        import time
        import uuid

        from langchain_core.callbacks import UsageMetadataCallbackHandler

        from app.providers.llm_factory import get_llm
        from app.rag.local_tracer import LocalTracer
        from app.rag.query_log import log_query

        # One tracer per query -> full per-step span tree in logs/traces.jsonl,
        # shared across the retrieval graph and the generation call.
        tracer = LocalTracer(query_id=uuid.uuid4().hex[:8])
        # Provider-agnostic token accounting: Ollama attaches usage to streamed
        # chunks, but Gemini reports it only via callback -- this captures both.
        usage_cb = UsageMetadataCallbackHandler()

        llm = get_llm(provider, api_key, model)   # built once, reused by graph + generation

        t0 = time.perf_counter()
        messages, docs, timings = self.prepare(question, llm, filter, callbacks=[tracer])
        t_retrieved = time.perf_counter()

        first_token_t = None
        parts: list[str] = []
        meta: dict = {}
        full = None   # accumulated message -> correct aggregated usage_metadata
        for chunk in llm.stream(messages, config={"callbacks": [tracer, usage_cb]}):
            # Sum chunks: Gemini emits usage_metadata INCREMENTALLY (the final chunk
            # is 0/0), so a single chunk's usage is never the total -- accumulate.
            full = chunk if full is None else full + chunk
            # chunk.content can be a plain string or a list of content blocks
            # (LangChain 1.x); .text normalizes either shape to plain text.
            text = chunk.text
            if text:
                if first_token_t is None:
                    first_token_t = time.perf_counter()   # prefill ends here
                parts.append(text)
                yield {"type": "token", "text": text}
            # Ollama's engine durations (prompt_eval_duration, etc.) arrive on the
            # "done" chunk's response_metadata -- capture whenever present.
            if getattr(chunk, "response_metadata", None):
                meta = chunk.response_metadata
        t_end = time.perf_counter()

        # Token counts: the callback aggregates correctly across providers; fall back
        # to the accumulated message's usage_metadata.
        usage = _usage_from_callback(usage_cb) or (getattr(full, "usage_metadata", None) or {})

        from app.rag.local_tracer import write_trace

        record = self._log(log_query, question, provider, llm, docs, parts,
                            t0, t_retrieved, first_token_t, t_end, usage, meta, timings)
        # Unified per-query trace: summary metrics + full span tree, one line.
        write_trace(tracer.query_id, record, tracer.roots)
        yield {"type": "sources", "sources": _sources(docs)}

    @staticmethod
    def _log(log_query, question, provider, llm, docs, parts,
             t0, t_retrieved, first_token_t, t_end, usage, meta, timings=None):
        prompt_tokens = usage.get("input_tokens") or meta.get("prompt_eval_count")
        completion_tokens = usage.get("output_tokens") or meta.get("eval_count")

        # Wall-clock prefill/decode (works for any provider, but the client can
        # buffer streamed tokens, skewing the split -- prefer engine timing below).
        prefill = (first_token_t - t_retrieved) if first_token_t else None
        decode = (t_end - first_token_t) if first_token_t else None

        # Ollama reports engine-measured durations in nanoseconds -- more precise
        # than wall-clock (no client buffering artifacts).
        def _ns(key):
            v = meta.get(key)
            return round(v / 1e9, 2) if isinstance(v, (int, float)) else None

        ollama_prefill = _ns("prompt_eval_duration")
        ollama_decode = _ns("eval_duration")

        # Throughput: prefer engine timing over wall-clock (no buffering skew).
        prefill_basis = ollama_prefill or prefill
        decode_basis = ollama_decode or decode
        prefill_tok_per_s = (prompt_tokens / prefill_basis) if (prompt_tokens and prefill_basis) else None
        decode_tok_per_s = (completion_tokens / decode_basis) if (completion_tokens and decode_basis) else None

        from app.providers.device import get_device

        timings = timings or {}
        record = {
            "question": question,
            "provider": provider,
            "model": getattr(llm, "model", None),
            "retrieval_s": round(t_retrieved - t0, 2),
            # Retrieval sub-stage breakdown (seconds) -- pinpoints embed vs rerank.
            "device": get_device(),
            "dense_s": timings.get("dense_s"),
            "bm25_s": timings.get("bm25_s"),
            "rerank_s": timings.get("rerank_s"),
            "candidates": timings.get("candidates"),
            "prefill_s": round(prefill, 2) if prefill else None,
            "decode_s": round(decode, 2) if decode else None,
            "generation_s": round(t_end - t_retrieved, 2),
            "total_s": round(t_end - t0, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefill_tok_per_s": round(prefill_tok_per_s, 2) if prefill_tok_per_s else None,
            "decode_tok_per_s": round(decode_tok_per_s, 2) if decode_tok_per_s else None,
            "answer_chars": len("".join(parts)),
            "answer": "".join(parts),
            "sources": [d.metadata.get("reference") for d in docs],
            # engine-reported, Ollama only (authoritative when present)
            "ollama_prefill_s": ollama_prefill,
            "ollama_decode_s": ollama_decode,
            "ollama_model_load_s": _ns("load_duration"),
        }
        log_query(record)      # queries.jsonl + terminal summary
        return record          # reused by the unified trace writer

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


def _usage_from_callback(cb) -> dict:
    """Flatten UsageMetadataCallbackHandler.usage_metadata (keyed per model) into
    the standard {input_tokens, output_tokens} shape the logger expects."""
    data = getattr(cb, "usage_metadata", None) or {}
    inp = sum((u.get("input_tokens") or 0) for u in data.values())
    out = sum((u.get("output_tokens") or 0) for u in data.values())
    return {"input_tokens": inp, "output_tokens": out} if (inp or out) else {}


def _sources(docs) -> list[dict]:
    return [
        {"reference": d.metadata.get("reference"),
         "chapter": d.metadata.get("chapter"),
         "verse": d.metadata.get("verse"),
         "page": d.metadata.get("page")}
        for d in docs
    ]

