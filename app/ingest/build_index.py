"""Build the vector index from every PDF in the Docs folder.

    python -m app.ingest.build_index            # use configured providers
    python -m app.ingest.build_index --dry-run   # chunk only, no embedding

The embedding model id is written to storage/embedding_id.txt so retrieval can
detect an index/query mismatch (see architecture.md 3.1).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

# Windows consoles default to cp1252 and choke on the PDF's Sanskrit glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.ingest.chunk_router import chunk_document
from app.providers.embedding_factory import embedding_id, get_embeddings
from app.vectorstores.base import get_store
from config import ROOT, settings


def build(dry_run: bool = False) -> None:
    pdfs = sorted(settings.docs_path.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {settings.docs_path}")

    all_docs = []
    for pdf in pdfs:
        print(f"Parsing + chunking: {pdf.name}")
        docs = chunk_document(pdf)
        by_type = Counter(d.metadata.get("doc_type") for d in docs)
        print(f"  -> {len(docs)} chunks {dict(by_type)}")
        all_docs.extend(docs)

    sizes = [len(d.page_content) for d in all_docs]
    print(f"\nTotal chunks: {len(all_docs)} | "
          f"chars min/avg/max: {min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)}")

    if dry_run:
        print("\n--dry-run: skipping embedding. Sample chunk:\n")
        print(all_docs[0].page_content[:600])
        print("\nmetadata:", all_docs[0].metadata)
        return

    print(f"\nEmbedding with '{embedding_id()}' and writing to {settings.vector_backend} ...")
    embeddings = get_embeddings()
    store = get_store(embeddings)
    store.add(all_docs)

    (ROOT / settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    (ROOT / "storage" / "embedding_id.txt").write_text(embedding_id(), encoding="utf-8")
    print(f"Done. Index now holds {store.count()} vectors.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="chunk only, no embedding")
    args = ap.parse_args()
    build(dry_run=args.dry_run)
