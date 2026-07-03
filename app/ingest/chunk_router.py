"""Chunking strategy router.

Picks a splitter per document:
  * Gita (rigid structure) -> one chunk per verse-unit {translation + purport}.
  * Everything else        -> recursive character splitting (generic default).

Every chunk carries a common metadata schema so citations are rich where
structure exists and degrade gracefully where it doesn't:
    source, doc_type, title, section, page   (+ optional chapter, verse, part)
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest.parse import extract_pages, looks_like_gita

# Split a purport into sub-chunks once it gets this long (chars ~ 4x tokens).
_LONG_PURPORT_CHARS = 3500

_CHAPTER_RE = re.compile(r"^CHAPTER\s+([A-Z]+)$")
_TEXT_RE = re.compile(r"^TEXTS?\s+([\d]+(?:\s*[-–]\s*\d+)?)$")

_WORDS = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6,
    "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12,
    "THIRTEEN": 13, "FOURTEEN": 14, "FIFTEEN": 15, "SIXTEEN": 16,
    "SEVENTEEN": 17, "EIGHTEEN": 18,
}


def chunk_document(pdf_path: str | Path) -> list[Document]:
    """Route a single PDF to the right chunking strategy."""
    path = Path(pdf_path)
    pages = extract_pages(path)
    if looks_like_gita(pages):
        return _chunk_gita(pages, source=path.name)
    return _chunk_generic(pages, source=path.name)


# --------------------------------------------------------------------------- #
# Gita: structure-aware verse chunking                                        #
# --------------------------------------------------------------------------- #
def _chunk_gita(pages: list[dict], source: str) -> list[Document]:
    docs: list[Document] = []

    chapter_num: int | None = None
    chapter_title_parts: list[str] = []
    capturing_title = False

    verse: str | None = None
    verse_page: int | None = None
    section: str | None = None                       # TRANSLATION | PURPORT | HEAD
    buf = {"TRANSLATION": [], "PURPORT": []}

    def flush_verse():
        nonlocal verse, buf
        if verse is None:
            return
        translation = " ".join(buf["TRANSLATION"]).strip()
        purport = "\n".join(buf["PURPORT"]).strip()
        if translation or purport:
            docs.extend(
                _build_verse_docs(
                    source=source,
                    chapter=chapter_num,
                    verse=verse,
                    title=" ".join(chapter_title_parts).strip(),
                    page=verse_page,
                    translation=translation,
                    purport=purport,
                )
            )
        verse = None
        buf = {"TRANSLATION": [], "PURPORT": []}

    for pg in pages:
        for raw in pg["text"].splitlines():
            line = raw.strip()
            if not line or line.isdigit():           # skip blanks & page numbers
                continue

            m_ch = _CHAPTER_RE.match(line)
            if m_ch:
                flush_verse()
                chapter_num = _WORDS.get(m_ch.group(1))
                chapter_title_parts = []
                capturing_title = True
                section = None
                continue

            m_tx = _TEXT_RE.match(line)
            if m_tx:
                flush_verse()
                capturing_title = False
                verse = re.sub(r"\s*[-–]\s*", "-", m_tx.group(1))
                verse_page = pg["page"]
                section = "HEAD"                     # skip Sanskrit + word-by-word
                continue

            if line == "TRANSLATION":
                section = "TRANSLATION"
                continue
            if line == "PURPORT":
                section = "PURPORT"
                continue

            if capturing_title:
                chapter_title_parts.append(line)
            elif section in ("TRANSLATION", "PURPORT"):
                buf[section].append(line)
            # HEAD lines (Sanskrit/transliteration/synonyms) are intentionally dropped.

    flush_verse()
    return docs


def _build_verse_docs(source, chapter, verse, title, page,
                      translation, purport) -> list[Document]:
    """One verse -> one or more Documents (purport split if very long)."""
    ref = f"BG {chapter}.{verse}" if chapter else f"BG ?.{verse}"
    base_meta = {
        "source": source,
        "doc_type": "gita",
        "title": title or None,
        "chapter": chapter,
        "verse": verse,
        "page": page,
        "reference": ref,
    }
    header = f"{ref}"
    if title:
        header += f" ({title})"

    # Short verse: single chunk with translation + purport together.
    full_purport = purport
    if len(purport) <= _LONG_PURPORT_CHARS:
        content = _compose(header, translation, full_purport)
        meta = {**base_meta, "section": f"{chapter}.{verse}", "type": "verse"}
        return [Document(page_content=content, metadata=meta)]

    # Long purport: keep translation with part 1, split the rest, mark continuations.
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    parts = splitter.split_text(full_purport)
    out: list[Document] = []
    for i, part in enumerate(parts):
        content = _compose(header, translation if i == 0 else "", part)
        meta = {
            **base_meta,
            "section": f"{chapter}.{verse}",
            "type": "verse",
            "part": i + 1,
            "is_continuation": i > 0,
        }
        out.append(Document(page_content=content, metadata=meta))
    return out


def _compose(header: str, translation: str, purport: str) -> str:
    blocks = [header]
    if translation:
        blocks.append(f"TRANSLATION: {translation}")
    if purport:
        blocks.append(f"PURPORT: {purport}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Generic: recursive splitting for unstructured PDFs                          #
# --------------------------------------------------------------------------- #
def _chunk_generic(pages: list[dict], source: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs: list[Document] = []
    for pg in pages:
        text = pg["text"].strip()
        if not text:
            continue
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "doc_type": "generic",
                        "title": None,
                        "section": None,
                        "page": pg["page"],
                    },
                )
            )
    return docs
