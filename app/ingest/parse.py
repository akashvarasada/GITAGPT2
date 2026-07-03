"""PDF parsing helpers.

Two extraction modes:
  * extract_pages()    -> per-page plain text (used by the Gita verse parser,
                          which needs line/page structure).
  * extract_markdown() -> structure-preserving Markdown (used for generic /
                          unstructured PDFs).

The Bhagavad-gita PDF has a font-encoding quirk that garbles Sanskrit diacritics,
but the English translation/purport text extracts cleanly -- which is what we
embed and answer from.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Return [{'page': 1-based int, 'text': str}, ...]."""
    doc = fitz.open(str(pdf_path))
    pages = [{"page": i + 1, "text": doc[i].get_text()} for i in range(doc.page_count)]
    doc.close()
    return pages


def extract_markdown(pdf_path: str | Path) -> str:
    """Whole document as Markdown (headings preserved where detectable)."""
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(pdf_path))


def looks_like_gita(pages: list[dict]) -> bool:
    """Heuristic: the Gita layout is dense with TRANSLATION/PURPORT/TEXT markers."""
    blob = "\n".join(p["text"] for p in pages[:120])
    return (
        blob.count("TRANSLATION") >= 5
        and blob.count("PURPORT") >= 5
        and blob.count("TEXT") >= 5
    )
