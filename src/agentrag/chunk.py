"""Stage 2 - Chunking.  (STANDALONE: `split_text` is pure Python, copy this file into any project.)

Why chunking is a RAG-layer concern, not an embedding-API setting:
an embedding endpoint maps ONE string to ONE vector (and truncates or rejects
strings over its token limit). It never splits your document. So you decide the
unit of retrieval here, before anything is embedded.

Strategy: character windows of `size`, preferring to end on a paragraph / line /
sentence / word boundary found in the last 40% of the window, with `overlap`
characters repeated at the start of the next chunk so ideas cut at a boundary
still appear whole in at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # chunk.py has no runtime dependency on the rest of the package
    from .ingest import Document

_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "; ", ", ", " ")


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int | None
    index: int  # position of the chunk within its document page
    text: str
    entity: str = ""
    department: str = ""


def split_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    out: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            end = _best_cut(text, start, end, size)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always make progress
    return out


def _best_cut(text: str, start: int, end: int, size: int) -> int:
    lo = start + int(size * 0.6)
    for sep in _SEPARATORS:
        i = text.rfind(sep, lo, end)
        if i != -1:
            return i + len(sep)
    return end


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(split_text(doc.text, size, overlap)):
            chunks.append(Chunk(doc.source, doc.page, i, piece, doc.entity, doc.department))
    return chunks
