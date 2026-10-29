"""Stage 1 - Ingest (and the glue that runs stages 1-4).

Ingest only answers: "what text do I have, and where did it come from?"
It reads files into `Document`s (one per PDF page, one per Markdown file) and
keeps the provenance (source path, page number) that retrieval later shows.
Splitting those documents into chunks is the NEXT stage (chunk.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Document:
    source: str  # path relative to the ingested folder
    page: int | None  # 1-based page for PDFs, None for text files
    text: str
    entity: str = ""  # company entity from the folder layout, e.g. "hk-hq"
    department: str = ""  # department from the folder layout, e.g. "hr"


def org_metadata(rel_path: str) -> tuple[str, str]:
    """<entity>/<department>/file -> (entity, department). The folders are the org chart."""
    parts = rel_path.split("/")
    if len(parts) >= 3:
        return parts[0], parts[1]
    return ("", parts[0]) if len(parts) == 2 else ("", "")


def load_documents(folder: str | Path) -> list[Document]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {root}")
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        entity, dept = org_metadata(rel)
        if suffix in TEXT_SUFFIXES:
            docs.append(Document(rel, None, path.read_text(encoding="utf-8"), entity, dept))
        elif suffix == ".pdf":
            for n, page in enumerate(PdfReader(path).pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:  # scanned pages have no text layer; this demo has no OCR
                    docs.append(Document(rel, n, text, entity, dept))
    return docs


def build_index(
    folder: str | Path,
    cfg,
    embedder,
    store,
    log: Callable[[str], None] = lambda _msg: None,
    batch_size: int = 64,
) -> int:
    """Run Ingest -> Chunk -> Embed -> Store. Returns the number of chunks stored."""
    from .chunk import chunk_documents  # local import: chunk.py imports Document from here

    docs = load_documents(folder)
    log(f"[1/4 ingest]  {len(docs)} documents")
    chunks = chunk_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
    log(f"[2/4 chunk]   {len(chunks)} chunks (size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

    for source in {c.source for c in chunks}:  # re-ingesting a file replaces its old chunks
        store.delete_source(source)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.embed([c.text for c in batch])
        store.add(batch, vectors)
    log(f"[3/4 embed]   {embedder.name} ({embedder.dim} dims)")
    log(f"[4/4 store]   {store.path} via {store.backend}")
    return len(chunks)
