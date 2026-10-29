"""Adapter: the SQLite/sqlite-vec store + an embedder, exposed through the `Searcher` interface."""

from __future__ import annotations

from .tools import Snippet, SourceInfo


class StoreSearcher:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def search(self, query: str, k: int, department: str | None = None) -> list[Snippet]:
        if self.store.count() == 0:
            return []
        vector = self.embedder.embed([query])[0]
        hits = self.store.search(vector, k * 5 if department else k)  # org filter runs after vector search
        if department:
            hits = [h for h in hits if h.department == department]
        return [
            Snippet(source=h.source, page=h.page, entity=h.entity, department=h.department, text=h.text, score=round(h.score, 4))
            for h in hits[:k]
        ]

    def sources(self) -> list[SourceInfo]:
        rows = self.store.db.execute(
            "SELECT source, entity, department, COUNT(*) FROM chunks GROUP BY source, entity, department ORDER BY source"
        ).fetchall()
        return [SourceInfo(source=s, entity=e, department=d, chunks=n) for s, e, d, n in rows]
