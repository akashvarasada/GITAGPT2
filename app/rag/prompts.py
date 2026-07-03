"""Prompt templates. Kept in one place so they are easy to tune."""
from __future__ import annotations

from langchain_core.documents import Document

ANSWER_SYSTEM = """You are GITAGPT, a study assistant for the Bhagavad-gita As It Is.

Rules:
- Answer ONLY using the provided context passages. Do not use outside knowledge.
- Cite the verses you rely on in the form (BG chapter.verse), e.g. (BG 2.47).
- If the context does not contain the answer, say exactly:
  "This is not addressed in the provided text."
- Be clear and faithful to the text; do not invent teachings or verse numbers.
"""

ANSWER_USER = """Context passages:
{context}

Question: {question}

Answer (with verse citations):"""

# Used by the query-rewrite node when initial retrieval looks weak.
REWRITE_SYSTEM = """Rewrite the user's question into a single, keyword-rich search
query optimized for retrieving relevant Bhagavad-gita verses and commentary.
Return only the rewritten query, nothing else."""


def format_context(docs: list[Document]) -> str:
    """Render retrieved docs as a numbered, citable context block."""
    blocks = []
    for i, d in enumerate(docs, 1):
        ref = d.metadata.get("reference", "?")
        blocks.append(f"[{i}] ({ref})\n{d.page_content}")
    return "\n\n".join(blocks)
