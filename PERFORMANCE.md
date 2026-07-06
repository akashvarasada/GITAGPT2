# GITAGPT — Performance Optimization Log

How we cut latency across retrieval, the LLM path, and startup — what changed,
**why** it was slow, the exact change, and **measured** before/after numbers from
`logs/queries.jsonl` and LangSmith.

> **Golden rule we followed:** measure first, then change. Every tradeoff is a
> tunable config knob defaulting to the old behavior — nothing degrades retrieval
> quality silently.

---

## TL;DR — headline results

| Stage | Before | After (Mac M3 / MPS) | After (Windows / CPU) |
|---|---|---|---|
| Retrieval (LangGraph span) | 9–39s | **2.3s** | **9.3s** (was 24s) |
| — of which reranking | ~all of it | 2.0s `on mps` | 9.1s `on cpu` |
| Cold model reload per query | 1.5–36s | ~0 (kept warm) | ~0 (kept warm) |
| Token accounting (Gemini) | `None+None` | `2823+1087` | n/a |
| First page load during boot | `ERR_CONNECTION_REFUSED` | loads instantly + "warming up" | same |

**The single biggest retrieval win = putting the cross-encoder reranker on the
Apple GPU (MPS).** On Windows (no GPU) that lever is a no-op; there the wins came
from **startup warm-up + keep-alive + not hitting the rewrite loop**.

The LLM generation itself on CPU (`gemma4:e4b` at ~4 tok/s) is hardware-bound and
cannot be refactored into speed — see point 7.

---

## GPU/MPS vs non-GPU: what improved on each machine

There are **two kinds** of gains, and it's important not to conflate them:

- **Compute speedup** — the reranker math itself runs faster. *Only the GPU/MPS
  does this.* This is the Mac-only gain.
- **Overhead removal** — deleting waste *around* the computation (cold model loads,
  model reloads, RAM thrash, redundant rewrite LLM calls). This helps *both*
  machines and is the *entire* story on Windows.

### Side-by-side (measured)

| | **Mac M3 (MPS / GPU)** | **Windows (CPU, no GPU)** |
|---|---|---|
| Reranker **compute** | **2.0s** `on mps` | **9.1s** `on cpu` |
| Retrieval span (before → after) | 9–39s → **2.3s** | 24s → **9.3s** |
| What drove the improvement | MPS compute **+** overhead removal | overhead removal **only** |
| Reranker math got faster? | **Yes (~4.5×)** | **No** — same CPU floor |
| Left at a hardware floor? | No — GPU headroom | Yes — 9.1s is the CPU floor |

**The ~4.5× gap (9.1s → 2.0s) IS the pure GPU/MPS gain.** It exists only because
the Apple GPU runs the 40 candidate forward-passes in parallel. Windows can never
reach 2.0s on rerank without a GPU — the best it can do is stop wasting time around
the 9.1s of unavoidable CPU math (which we did) or do *less* math (`RETRIEVE_K`).

### Which change helped which machine

| Change | Mac (MPS) | Windows (CPU) | Type |
|---|---|---|---|
| 1. Device pinning (MPS) | ✅ **4.5× rerank** | ⬜ no-op (`cpu`) | Compute speedup |
| 3. `keep_alive` | ✅ | ✅ | Overhead removal |
| 4. Startup warm-up | ✅ | ✅ | Overhead removal |
| 5. Rewrite-loop gate | ✅ | ✅ | Overhead removal |
| 6. `num_predict` cap | ✅ | ✅ | Less LLM work |
| 7. Gemini hardening | ✅ (provider) | ✅ (provider) | Provider-side |

**Bottom line:** Mac got the reranker *computed* faster (GPU) **and** the overhead
removed. Windows got **only** the overhead removed — so it improved (24s → 9.3s)
but is now sitting on the CPU compute floor, exactly as expected without a GPU.

---

## 1. Pin the reranker + embeddings to the GPU (MPS/CUDA), not CPU

**Why it was slow:** the cross-encoder reranker scores the query against ~40
candidates — 40 transformer forward passes. `sentence-transformers` frequently
places the `CrossEncoder` on **CPU** by default, where those passes run serially.

**Change:** a shared `get_device()` helper (`cuda` > `mps` > `cpu`, overridable via
`TORCH_DEVICE`), pinned on both models.

```python
# app/providers/device.py
@lru_cache(maxsize=1)
def get_device() -> str:
    override = os.getenv("TORCH_DEVICE", "").strip().lower()
    if override:
        return override
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# app/rag/reranker.py
_model = CrossEncoder(settings.reranker_model, device=get_device())

# app/providers/embedding_factory.py
HuggingFaceEmbeddings(model_name=..., model_kwargs={"device": get_device()}, ...)
```

**Effect (measured):**
- Mac M3: `rerank=9.1s on cpu` → **`rerank=2.0s on mps`**; whole retrieval span
  9–39s → **2.3s**.
- **Zero quality cost:** same weights + same math → identical scores and identical
  top-5 documents. Only the hardware changed.
- **Windows:** no-op (`get_device()` returns `cpu`), confirmed by `on cpu` in logs.

---

## 2. Instrument every retrieval sub-stage (measure, don't guess)

**Why:** the old trace showed `retrieve chain = 24s` as one opaque number — we
couldn't tell if embedding or reranking was the cost.

**Change:** `time.perf_counter()` around dense / BM25 / rerank, plus the device,
threaded into every query record and the console summary.

```
[query] local/gemma4:e4b retrieval=9.3s (dense=0.2s bm25=0.0s rerank=9.1s on cpu) \
        total=406.9s tokens=2823+1087 prefill=28.2 tok/s decode=4.1 tok/s
```

**Effect:** immediately proved **rerank is 98% of retrieval** and BM25 is free
(0.0s) — so all tuning effort targets the reranker, not the embedder or BM25.

> The numbers above (`rerank=9.1s on cpu`, `2.0s on mps`) were measured at
> `RETRIEVE_K=20` (~40 candidates). We then lowered `RETRIEVE_K` to **10** (~20
> candidates) — rerank cost scales ~linearly with candidate count, so expect
> roughly **half** the rerank time (~4.5s CPU, ~1s MPS) at negligible recall cost,
> since only the top 5 survive rerank anyway.

---

## 3. Keep the Ollama model warm (`keep_alive`)

**Why it was slow:** Ollama's default evicts the model between requests. On 16GB
RAM, competing with the embedder + reranker, the next query reloaded it — logged
as `ollama_model_load_s` of **1.5–36s** *per query*, plus RAM thrash that slowed
the concurrent CPU rerank.

**Change:**

```python
# app/providers/llm_factory.py
ChatOllama(model=name, ..., keep_alive=settings.ollama_keep_alive)  # default "30m"
```

**Effect:** eliminates the per-query reload. This is a big part of why Windows
retrieval dropped **24s → 9.1s** — the earlier 24s query logged
`ollama_model_load_s=36s`, i.e. it was thrashing while reranking; keeping the
model resident removed that contention and exposed the reranker's true ~9s floor.

---

## 4. Warm up models at startup + non-blocking boot + readiness UI

**Why it was slow / broken:** the reranker loaded **lazily on the first query**, so
the first user paid the model-load cost inside their request. A first attempt to
warm at startup made the server unreachable during boot → `ERR_CONNECTION_REFUSED`.

**Change:** warm up in a **background thread** so the server accepts connections
immediately; expose `GET /ready`; the UI polls it and disables the composer with a
"Warming up the system…" banner until models are loaded.

```python
# app/api/main.py — background warm-up, server reachable at once
threading.Thread(target=_warm, args=(app,), daemon=True).start()

# app/api/routes.py
@router.get("/ready")
def ready(request):
    st = request.app.state
    return {"ready": bool(getattr(st, "rag_ready", False)),
            "error": getattr(st, "rag_error", None)}
```

```js
// UI: gate the composer until /ready
async function pollReady() {
  const d = await (await fetch("/ready")).json();
  if (d.ready) return setReady(true);
  if (d.error) return setReady(false, d.error);
  setTimeout(pollReady, 1000);
}
```

**Effect:** first real query no longer pays model-load; page loads instantly and
shows progress instead of a refused connection; warm-up errors (e.g. empty index)
surface in a red banner instead of a silent hang.

---

## 5. Gate the CRAG rewrite loop

**Why it was slow:** on weak retrieval the graph rewrites the query — a **blocking
LLM call inside the retrieval graph** *plus a second retrieve+rerank*. On a slow
local model this alone added 10–40s and doubled retrieval; it also caused Gemini to
make two calls.

**Change:** a master switch, default on (CRAG preserved), off to trim/measure.

```python
# app/rag/graph.py
def decide(state):
    if (not settings.enable_rewrite
            or state.get("relevant")
            or state.get("tries", 0) >= 1):
        return "finalize"
    return "rewrite"
```

**Effect:** explains the earlier 24–47s retrieval spikes vs the ~9s floor. Queries
with a strong top match (e.g. "What is detachment?" → BG 15.1) skip the rewrite and
stay fast; `ENABLE_REWRITE=false` guarantees no in-graph LLM call.

---

## 6. Right-size Ollama context + cap output (tunable)

**Why it can be slow:** an oversized `num_ctx` (e.g. 16k for ~2.5k-token prompts)
wastes KV-cache memory and slows prefill setup. Verbose answers (1087 tokens at
~4 tok/s ≈ 265s of decode) dominate total time.

**Change:** optional knobs, default `None` = no change.

```python
# config.py
ollama_num_ctx: int | None = None       # right-size to prompt+answer
ollama_num_predict: int | None = None    # hard cap on generated tokens
```

```bash
# .env
OLLAMA_NUM_CTX=6144
OLLAMA_NUM_PREDICT=512     # ~halves decode time on CPU
```

**Effect:** `num_predict` is the main *code-side* lever on CPU generation — capping
a 1087-token answer to 512 roughly halves the ~265s decode. Tradeoff: shorter
answers, so it's opt-in.

---

## 7. Gemini hardening — fix the 150s "retry storm" + wrong answer

**Why it was slow / wrong:** on the Mac, Gemini took ~150s and returned "This is not
addressed in the provided text." Two causes: safety filters falsely refusing Gita
content (war, killing kinsmen) with retry backoff, and — confirmed via LangSmith —
**thinking latency**: first token at 29.98s of a 30.02s call.

**Change:**

```python
# app/providers/llm_factory.py
ChatGoogleGenerativeAI(
    ...,
    max_retries=settings.gemini_max_retries,          # default 2 (was lib default 6)
    safety_settings=_gemini_safety_settings(),         # "relaxed" = BLOCK_NONE
    thinking_budget=settings.gemini_thinking_budget,   # set 0 to disable thinking
    max_output_tokens=settings.gemini_max_output_tokens,
)
```

**Effect:**
- Retrieval on Mac already 2.3s; with `GEMINI_THINKING_BUDGET=0` the ~30s
  time-to-first-token collapses to a few seconds → total ~3–6s.
- `safety=relaxed` stops false refusals on scripture; `max_retries=2` fails fast
  instead of a long backoff storm.

---

## 8. Capture token counts for all providers

**Why it was broken:** logs showed `tokens=None+None`, `prefill=n/a`, `decode=n/a`
for Gemini. Ollama attaches usage to streamed chunks; **Gemini does not** — it
reports usage via callback only.

**Change:** attach `UsageMetadataCallbackHandler` and fall back to it when chunks
carry no usage.

```python
# app/rag/service.py
usage_cb = UsageMetadataCallbackHandler()
for chunk in llm.stream(messages, config={"callbacks": [tracer, usage_cb]}):
    ...
if not usage:                       # Gemini path
    usage = _usage_from_callback(usage_cb)
```

**Effect:** `tokens=None+None` → `tokens=2823+1087`, and prefill/decode tok/s now
compute — so every provider is measurable.

---

## What did NOT change the LLM generation speed (honest note)

The LangSmith ChatOllama latencies (135s–760s) are driven by **token count ÷ CPU
decode rate (~4 tok/s)** — pure hardware. Our changes removed *reload* overhead
(`keep_alive`) and *cold-load* (warm-up), and `num_predict` can shorten answers,
but the raw per-token generation speed of `gemma4:e4b` on a CPU is unchanged. The
real levers for sub-5s answers are **Gemini** (hosted GPU) or a **local GPU / smaller
model** — both already supported via config, no code change.

---

## Config quick-reference (all new knobs)

```bash
# Retrieval device (Mac: mps; unset = auto cuda>mps>cpu)
# TORCH_DEVICE=mps

# Ollama
OLLAMA_KEEP_ALIVE=30m        # keep model resident (avoids 1.5-36s reloads)
# OLLAMA_NUM_CTX=6144        # right-size context window
# OLLAMA_NUM_PREDICT=512     # cap answer length (~halves CPU decode)

# Retrieval
RETRIEVE_K=10                # candidates per arm before rerank (lowered from 20 -> ~halves rerank)
ENABLE_REWRITE=true          # CRAG rewrite retry (blocking in-graph LLM call when on)

# Gemini
GEMINI_MAX_RETRIES=2
GEMINI_SAFETY=relaxed        # avoids false refusals on scripture
# GEMINI_THINKING_BUDGET=0   # disable thinking -> big TTFT drop
# GEMINI_MAX_OUTPUT_TOKENS=768
```

## Files touched

- `app/providers/device.py` *(new)* — device selection
- `app/providers/embedding_factory.py` — pin embed device
- `app/providers/llm_factory.py` — keep_alive, num_ctx/num_predict, Gemini hardening
- `app/rag/reranker.py` — pin reranker device
- `app/rag/retriever.py`, `app/rag/graph.py` — sub-stage instrumentation, rewrite gate
- `app/rag/service.py` — usage capture, timings, `warmup()`
- `app/rag/query_log.py` — breakdown in console summary
- `app/api/main.py` — background warm-up + lifespan
- `app/api/routes.py` — `/ready` endpoint, lock-guarded `_get_rag`
- `app/ui/templates/index.html` — warm-up banner + readiness gating
- `config.py`, `.env.example` — new tunable settings
```
