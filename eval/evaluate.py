"""RAGAS evaluation harness.

    python -m eval.evaluate                 # eval with configured LLM provider
    python -m eval.evaluate --limit 3       # quick subset

Reports Faithfulness, Answer Relevancy, Context Precision, Context Recall, plus
a simple retrieval Hit@k (did any expected verse appear in the sources?). Re-run
after any change to chunking, embeddings, retriever, or model.

Requires: pip install ragas   (kept optional in requirements.txt)
"""
from __future__ import annotations

import argparse
import json
import sys

from config import ROOT, settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLDEN = ROOT / "eval" / "golden_set.jsonl"


def load_golden(limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def hit_at_k(expected_refs: list[str], sources: list[dict]) -> bool:
    found = {str(s.get("verse")) for s in sources}
    # match on either exact verse or the chapter.verse fragment
    refs_in_sources = {str(s.get("reference", "")).replace("BG ", "") for s in sources}
    return any(r in found or r in refs_in_sources for r in expected_refs)


def main(limit: int | None):
    from app.rag.service import RagService

    rag = RagService()
    rows = load_golden(limit)
    print(f"Evaluating {len(rows)} questions with LLM provider "
          f"'{settings.llm_provider}', embeddings '{settings.embed_model}'.\n")

    records, hits = [], 0
    for i, row in enumerate(rows, 1):
        out = rag.answer(row["question"])
        hit = hit_at_k(row.get("expected_refs", []), out["sources"])
        hits += hit
        refs = ", ".join(s["reference"] for s in out["sources"] if s.get("reference"))
        print(f"[{i}/{len(rows)}] hit@k={hit}  sources: {refs}")
        records.append({
            "user_input": row["question"],
            "response": out["answer"],
            "retrieved_contexts": out["contexts"],
            "reference": row["reference"],
        })

    print(f"\nRetrieval Hit@k: {hits}/{len(rows)} = {hits / len(rows):.0%}")
    _run_ragas(records)


def _run_ragas(records: list[dict]):
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (answer_relevancy, context_precision,
                                    context_recall, faithfulness)
    except ImportError:
        print("\n(ragas not installed -- `pip install ragas` for LLM-graded metrics. "
              "Retrieval Hit@k above still gives a useful signal.)")
        return

    from app.providers.embedding_factory import get_embeddings
    from app.providers.llm_factory import get_llm

    judge = LangchainLLMWrapper(get_llm())
    embed = LangchainEmbeddingsWrapper(get_embeddings())
    dataset = EvaluationDataset.from_list(records)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embed,
    )
    print("\nRAGAS metrics:")
    print(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.limit)
