"""Adapter: the SQLite/sqlite-vec store + an embedder, exposed through the `Searcher` interface.

Thread safety: when the model asks for several tools at once (parallel tool calls), Pydantic AI
runs sync tools concurrently in worker threads. One sqlite3 connection must never be used by two
threads at the same time - that corrupts results - so every call goes through one lock.
"""

from __future__ import annotations

import threading

from .tools import Snippet, SourceInfo


class StoreSearcher:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder
        self._lock = threading.Lock()

    def search(self, query: str, k: int, department: str | None = None) -> list[Snippet]:
        with self._lock:
            return self._search(query, k, department)

    def sources(self) -> list[SourceInfo]:
        with self._lock:
            return self._sources()

    def _search(self, query: str, k: int, department: str | None) -> list[Snippet]:
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

    def _sources(self) -> list[SourceInfo]:
        rows = self.store.db.execute(
            "SELECT source, entity, department, COUNT(*) FROM chunks GROUP BY source, entity, department ORDER BY source"
        ).fetchall()
        return [SourceInfo(source=s, entity=e, department=d, chunks=n) for s, e, d, n in rows]
