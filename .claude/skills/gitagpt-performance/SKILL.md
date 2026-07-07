---
name: gitagpt-performance
description: Diagnose and optimize latency in the GITAGPT RAG pipeline (retrieval, reranking, LLM generation, startup/warm-up). Use when the user reports slowness, asks to tune latency, add a new perf knob, or investigate a "why is this query slow" question.
---

# GITAGPT performance workflow

Full history and measured numbers live in [PERFORMANCE.md](../../../PERFORMANCE.md) — **read it first**
before assuming a bottleneck is new. Most obvious wins (cold model loads, reranker device, rewrite-loop
cost, oversized context) are already fixed; re-diagnosing them wastes a turn.

## Golden rule (do not violate)

**Measure first, then change.** Every perf knob in this project defaults to the old/safe behavior and
must never silently degrade retrieval quality (see `config.py` comments — each knob explains its
tradeoff). If a proposed change trades quality for speed, make it opt-in via `.env`, not a new default.

## Step 1 — get a number before touching code

Every query is already instrumented end-to-end. Don't add new timers before checking these first:

- `logs/queries.jsonl` — one line per query with the breakdown format:
  `retrieval=Xs (dense=Xs bm25=Xs rerank=Xs on <device>) total=Xs tokens=in+out prefill=X tok/s decode=X tok/s`
- `logs/traces.jsonl` — raw trace detail
- LangSmith (if `LANGSMITH_*` set in `.env`) for per-node graph timing

Reproduce the slow query and read the log line before hypothesizing. `app/rag/query_log.py` is where the
console summary is assembled if you need to add a field.

## Step 2 — know the bottleneck ranking (already established)

1. **Reranking dominates on CPU** (~98% of retrieval span at `RETRIEVE_K=20`). Dense embed and BM25 are
   near-free (0.0–0.2s). Don't waste time optimizing the embedder or BM25 — optimize rerank or reduce
   candidate count (`RETRIEVE_K`).
2. **LLM decode is hardware-bound**, not a code problem. `gemma*:e*` on CPU runs ~4 tok/s; a 1000-token
   answer is ~250s regardless of refactoring. The only real levers: cap `OLLAMA_NUM_PREDICT`, switch to
   Gemini (hosted GPU), or use a local GPU / smaller model.
3. **Overhead (cold loads, reloads, thrash) is fixable everywhere**; **compute speed (GPU/MPS) only helps
   the reranker/embedder**, not CPU-bound LLM decode. Don't conflate the two when explaining a result —
   see the Mac-vs-Windows table in PERFORMANCE.md §"GPU/MPS vs non-GPU".

## Step 3 — check existing knobs before writing new code

```bash
TORCH_DEVICE=mps                # force device (auto = cuda>mps>cpu) — app/providers/device.py
OLLAMA_KEEP_ALIVE=30m           # keep model resident, avoids 1.5-36s reload per query
OLLAMA_NUM_CTX=6144             # right-size context window to prompt+answer size
OLLAMA_NUM_PREDICT=512          # cap generated tokens (~halves CPU decode time)
RETRIEVE_K=10                   # candidates per arm before rerank; rerank cost scales ~linearly with this
ENABLE_REWRITE=true             # CRAG rewrite retry — a *blocking* in-graph LLM call when triggered
GEMINI_MAX_RETRIES=2            # avoid retry-storm backoff
GEMINI_SAFETY=relaxed           # BLOCK_NONE — avoids false refusals on war/violence content in the Gita
GEMINI_THINKING_BUDGET=0        # disable thinking to cut Gemini time-to-first-token
GEMINI_MAX_OUTPUT_TOKENS=768    # cap Gemini answer length
```

If the user's complaint maps to one of these, adjust the `.env` value first — cheaper than a code change.

## Step 4 — if a genuinely new bottleneck is found

1. Instrument it the same way as existing stages (`time.perf_counter()`, thread into `query_log.py`).
2. Make the fix a config-gated knob defaulting to current behavior, mirroring the pattern in `config.py`
   (see how `ollama_num_predict`, `enable_rewrite`, `gemini_thinking_budget` are all `None`/safe by default).
3. Re-run `eval/evaluate.py` (RAGAS) to confirm retrieval/answer quality didn't regress — speed wins that
   quietly hurt Context Precision/Recall or Faithfulness are not acceptable.
4. Capture before/after numbers from `logs/queries.jsonl` on whichever machine you tested (note CPU vs
   MPS/CUDA explicitly — the two have very different ceilings, see Step 2).
5. Append a new numbered section to `PERFORMANCE.md` in the existing style: **Why it was slow** → **Change**
   (with code snippet) → **Effect** (measured before/after), and add the file to "Files touched" and any
   new `.env` knob to the "Config quick-reference" block.

## Common traps specific to this codebase

- Don't pin embeddings/reranker to a device without going through `app/providers/device.py`'s
  `get_device()` — it's the single shared source of truth (`TORCH_DEVICE` override → `cuda` → `mps` → `cpu`).
- Changing the embedding model or device does **not** require a re-index; changing the embedding *model*
  does (index and query embeddings must match — see [architecture.md](../../../architecture.md) §3.1).
- A slow query on Windows is very likely just the CPU rerank floor (~9s), not a regression — check the
  `on <device>` field in the log line before assuming something broke.
- The CRAG rewrite loop (`ENABLE_REWRITE`) adds a second retrieve+rerank *and* a blocking LLM call — this
  is the usual explanation for outlier queries that are 2-4x slower than the norm.
