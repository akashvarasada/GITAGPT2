"""Local span-tree tracer -- LangSmith-style tracing, natively, no cloud.

LangChain's callback system builds a full run tree for every invocation (each
chain / retriever / LLM step as a nested Run with its exact inputs, outputs, and
timing). By subclassing BaseTracer we receive that tree via `_persist_run` and
keep it in memory; the service then writes ONE unified record per query to
logs/traces.jsonl -- the summary metrics plus the complete span tree, so a single
line contains everything (mirrors what a LangSmith trace shows in one view).

Attach a tracer instance via `config={"callbacks": [tracer]}` on any invoke/stream.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime

from langchain_core.tracers.base import BaseTracer

from config import ROOT

_TRACE_PATH = ROOT / "logs" / "traces.jsonl"
_lock = threading.Lock()

# Never write these to disk (the graph state carries the API key through it).
_SECRET_KEYS = {"api_key", "google_api_key", "voyage_api_key"}


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("***" if k in _SECRET_KEYS else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


class LocalTracer(BaseTracer):
    """Collects each completed run tree in memory (self.roots)."""

    def __init__(self, query_id: str, **kwargs):
        super().__init__(**kwargs)
        self.query_id = query_id
        self.roots: list[dict] = []

    def _persist_run(self, run) -> None:
        self.roots.append(_node(run))


def _node(run) -> dict:
    """One Run -> a serializable span with nested children."""
    latency = None
    if run.start_time and run.end_time:
        latency = round((run.end_time - run.start_time).total_seconds(), 3)
    node = {
        "name": run.name,
        "run_type": run.run_type,
        "latency_s": latency,
        "inputs": _redact(run.inputs),
        "outputs": _redact(run.outputs),
        "error": run.error,
        "children": [_node(c) for c in run.child_runs],
    }
    if run.run_type == "llm":
        node["token_usage"] = _llm_usage(run)
    return node


def _llm_usage(run) -> dict | None:
    """Best-effort per-LLM-span token counts / engine durations from outputs."""
    try:
        info = run.outputs["generations"][0][0].get("generation_info") or {}
    except (TypeError, KeyError, IndexError, AttributeError):
        return None
    if not info:
        return None
    return {
        "prompt_tokens": info.get("prompt_eval_count"),
        "completion_tokens": info.get("eval_count"),
        "prompt_eval_s": round(info["prompt_eval_duration"] / 1e9, 2)
        if info.get("prompt_eval_duration") else None,
        "eval_s": round(info["eval_duration"] / 1e9, 2)
        if info.get("eval_duration") else None,
    }


def write_trace(query_id: str, summary: dict, roots: list[dict]) -> None:
    """Write one unified per-query record: metrics + the full span tree."""
    payload = {
        "query_id": query_id,
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "spans": roots,
    }
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with _TRACE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")
