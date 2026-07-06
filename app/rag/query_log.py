"""Per-query metrics logger.

Appends one JSON object per query to logs/queries.jsonl -- easy to tail live or
load into pandas later. Also prints a one-line human summary to the console.

Metrics captured per query:
  retrieval_s  : time in the LangGraph retrieval brain (embed + hybrid + rerank)
  prefill_s    : wall-clock time to the FIRST generated token (prompt processing)
  decode_s     : wall-clock time from first to last token (token generation)
  generation_s : prefill_s + decode_s
  total_s      : end-to-end
  prompt/completion_tokens, decode_tok_per_s, sources, answer_chars
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime

from config import ROOT

_LOG_PATH = ROOT / "logs" / "queries.jsonl"
_lock = threading.Lock()


def log_query(record: dict) -> dict:
    """Append a query record (with timestamp) to the log file. Thread-safe."""
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _print_summary(record)
    return record


def _fmt(v, unit=""):
    return f"{v:.1f}{unit}" if isinstance(v, (int, float)) else "n/a"


def _safe(text: str) -> str:
    """Make text printable on any console (the PDF's Sanskrit glyphs break cp1252)."""
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, "replace").decode(enc)


def _print_summary(r: dict) -> None:
    line = (
        f"[query] {r.get('provider')}/{r.get('model')} "
        f"retrieval={_fmt(r.get('retrieval_s'), 's')} "
        f"total={_fmt(r.get('total_s'), 's')} "
        f"tokens={r.get('prompt_tokens')}+{r.get('completion_tokens')} "
        f"prefill={_fmt(r.get('prefill_tok_per_s'))} tok/s "
        f"decode={_fmt(r.get('decode_tok_per_s'))} tok/s"
    )
    q = r.get("question", "")
    a = r.get("answer", "")
    print(f"\n{line}\nQ: {_safe(q)}\nA: {_safe(a)}\n", flush=True)
