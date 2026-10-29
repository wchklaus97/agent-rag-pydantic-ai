"""Stage 4 - Vector store (one SQLite file).

Tables
  meta(key, value)        which embedder built the index (vectors from different
                          models are not comparable, so we refuse to mix them)
  chunks(id, source, page, chunk_index, text, embedding)
  vec_chunks              sqlite-vec `vec0` virtual table; rowid == chunks.id

If the sqlite-vec extension cannot be loaded (some Python builds disable
extension loading), the store falls back to a documented pure-Python scan: the
vector is kept in chunks.embedding and cosine similarity is computed in Python.
Same API, just O(n) per query, fine for demos.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hit:
    id: int
    source: str
    page: int | None
    chunk_index: int
    text: str
    score: float  # cosine similarity, higher = closer
    entity: str = ""
    department: str = ""


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


class Store:
    def __init__(self, path: str | Path, embedder_name: str, dim: int, force_python: bool = False):
        self.path = str(path)
        self.dim = dim
        self.db = sqlite3.connect(self.path, check_same_thread=False)  # tools run in worker threads; StoreSearcher serialises access with a lock
        self.backend = "python-fallback"
        if not force_python:
            self._try_load_vec()
        self._init_schema(embedder_name)

    # --- setup -----------------------------------------------------------
    def _try_load_vec(self) -> None:
        try:
            import sqlite_vec

            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            self.backend = "sqlite-vec"
        except (ImportError, AttributeError, sqlite3.OperationalError):
            pass

    def _init_schema(self, embedder_name: str) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, page INTEGER,
                chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB,
                entity TEXT NOT NULL DEFAULT '', department TEXT NOT NULL DEFAULT '');
            CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
            """
        )
        row = self.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        if row and row[0] != f"{embedder_name}|{self.dim}":
            raise RuntimeError(
                f"Index was built with {row[0]!r} but you are using {embedder_name!r}|{self.dim}. "
                "Embeddings from different models can't be compared. Run `rag reset` and re-ingest."
            )
        self.db.execute("INSERT OR REPLACE INTO meta VALUES('embedder', ?)", (f"{embedder_name}|{self.dim}",))
        if self.backend == "sqlite-vec":
            self.db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                f"embedding float[{self.dim}] distance_metric=cosine)"
            )
        self.db.commit()

    # --- write -----------------------------------------------------------
    def add(self, chunks, vectors: list[list[float]]) -> None:
        for c, vec in zip(chunks, vectors, strict=True):
            blob = _pack(vec)
            cur = self.db.execute(
                "INSERT INTO chunks(source, page, chunk_index, text, embedding, entity, department) VALUES(?,?,?,?,?,?,?)",
                (c.source, c.page, c.index, c.text, None if self.backend == "sqlite-vec" else blob, c.entity, c.department),
            )
            if self.backend == "sqlite-vec":
                self.db.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES(?,?)", (cur.lastrowid, blob))
        self.db.commit()

    def delete_source(self, source: str) -> None:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE source=?", (source,))]
        if self.backend == "sqlite-vec":
            self.db.executemany("DELETE FROM vec_chunks WHERE rowid=?", [(i,) for i in ids])
        self.db.execute("DELETE FROM chunks WHERE source=?", (source,))
        self.db.commit()

    # --- read ------------------------------------------------------------
    def search(self, vector: list[float], k: int) -> list[Hit]:
        if self.backend == "sqlite-vec":
            rows = self.db.execute(
                "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (_pack(vector), k),
            ).fetchall()
            scored = [(rid, 1.0 - dist) for rid, dist in rows]  # cosine distance -> similarity
        else:
            scored = []
            for rid, blob in self.db.execute("SELECT id, embedding FROM chunks"):
                scored.append((rid, _cosine(vector, _unpack(blob))))
            scored = sorted(scored, key=lambda t: t[1], reverse=True)[:k]

        hits = []
        for rid, score in scored:
            r = self.db.execute("SELECT id, source, page, chunk_index, text, entity, department FROM chunks WHERE id=?", (rid,)).fetchone()
            hits.append(Hit(*r[:5], score=score, entity=r[5], department=r[6]))
        return hits

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def sources(self) -> list[tuple[str, int]]:
        return self.db.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source").fetchall()

    def embedder_info(self) -> str:
        return self.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()[0]

    def close(self) -> None:
        self.db.close()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def reset_db(path: str | Path) -> bool:
    """Delete the index file (and SQLite sidecar files). Returns True if anything was removed."""
    removed = False
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
            removed = True
    return removed
