"""Export recent LangSmith traces for the 'gitagpt' project to a JSON file.

    python -m eval.export_traces                     # last 20 runs -> traces.json
    python -m eval.export_traces --limit 50 --out my_traces.json

Reads LANGSMITH_API_KEY from .env (loaded by config). Share the resulting JSON
file for analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config  # noqa: F401  -- imported for its load_dotenv() side effect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = "gitagpt"


def _latency(run) -> float | None:
    if run.start_time and run.end_time:
        return (run.end_time - run.start_time).total_seconds()
    return None


def _engine_info(outputs) -> dict:
    """Dig out Ollama's generation_info (token counts + ns durations) from a
    ChatOllama run's outputs. Returns {} for non-Ollama or missing data."""
    try:
        return outputs["generations"][0][0]["generation_info"] or {}
    except (TypeError, KeyError, IndexError):
        return {}


def _throughput(outputs) -> dict:
    """Compute prefill/decode tok/sec from engine timings, mirroring the live
    query log. Durations are nanoseconds; None when data is absent."""
    info = _engine_info(outputs)
    prompt_tok = info.get("prompt_eval_count")
    completion_tok = info.get("eval_count")
    prefill_s = (info.get("prompt_eval_duration") or 0) / 1e9 or None
    decode_s = (info.get("eval_duration") or 0) / 1e9 or None
    return {
        "prefill_tok_per_s": round(prompt_tok / prefill_s, 2) if prompt_tok and prefill_s else None,
        "decode_tok_per_s": round(completion_tok / decode_s, 2) if completion_tok and decode_s else None,
    }


def main(limit: int, out: str):
    from langsmith import Client

    client = Client()
    runs = list(client.list_runs(project_name=PROJECT, limit=limit))

    data = []
    for r in runs:
        data.append({
            "name": r.name,
            "run_type": r.run_type,
            "start_time": r.start_time.isoformat() if r.start_time else None,
            "latency_s": _latency(r),
            "total_tokens": getattr(r, "total_tokens", None),
            "prompt_tokens": getattr(r, "prompt_tokens", None),
            "completion_tokens": getattr(r, "completion_tokens", None),
            **_throughput(r.outputs),   # prefill_tok_per_s + decode_tok_per_s (Ollama)
            "error": r.error,
            "inputs": r.inputs,
            "outputs": r.outputs,
            "trace_id": str(r.trace_id) if r.trace_id else None,
            "parent_run_id": str(r.parent_run_id) if r.parent_run_id else None,
        })

    Path(out).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(data)} runs to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="traces.json")
    args = ap.parse_args()
    main(args.limit, args.out)
