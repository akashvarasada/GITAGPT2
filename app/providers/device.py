"""Pick the best available torch device for local models (embeddings + reranker).

Order: explicit TORCH_DEVICE override > CUDA > Apple MPS > CPU. Probed once and
cached, so importing this is cheap and the choice is stable for the process.

Why this matters: the embedding model (bge-m3) and the cross-encoder reranker
(bge-reranker-base) are transformer forward passes -- exactly the parallel matmul
workload a GPU accelerates. On Apple Silicon, sentence-transformers frequently
lands the CrossEncoder on CPU by default, which is 5-15x slower for reranking
~40 candidates than the M-series GPU. The device only changes WHERE the math
runs, not the math itself, so scores/rankings are identical -- a pure latency win
with no retrieval-quality change.
"""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_device() -> str:
    """Return "cuda" | "mps" | "cpu". Set TORCH_DEVICE to force a specific one."""
    override = os.getenv("TORCH_DEVICE", "").strip().lower()
    if override:
        return override
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
