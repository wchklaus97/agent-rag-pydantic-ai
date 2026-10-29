"""The retrieve tool's contract - STANDALONE (needs only pydantic; no agent, no SQLite).

Lift this file into any agent framework: it defines WHAT the model may ask for and what
comes back, independent of how the search is implemented.

  Searcher        the interface a backend implements (SQLite here; could be anything)
  Snippet         one retrieved passage + where it came from (the model cites these)
  retrieve_snippets / list_sources   the safe, typed functions the agent tools wrap

Design rules that make a retrieval tool agent-friendly:
  * never raise on "nothing found": return [] so the model can say so
  * clamp k so a model can't request 10,000 passages
  * return the source with every passage, otherwise the model can't cite
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

MAX_K = 10


class Snippet(BaseModel):
    source: str  # file path relative to the ingested folder
    page: int | None = None
    entity: str = ""  # company entity folder, e.g. hk-hq
    department: str = ""  # department folder, e.g. finance
    text: str
    score: float  # cosine similarity, higher = closer


class SourceInfo(BaseModel):
    source: str
    entity: str = ""
    department: str = ""
    chunks: int


class Searcher(Protocol):
    def search(self, query: str, k: int, department: str | None = None) -> list[Snippet]: ...

    def sources(self) -> list[SourceInfo]: ...


def retrieve_snippets(searcher: Searcher, query: str, k: int = 4, department: str | None = None) -> list[Snippet]:
    query = query.strip()
    if not query:
        return []
    return searcher.search(query, max(1, min(k, MAX_K)), department or None)


def list_sources(searcher: Searcher) -> list[SourceInfo]:
    return searcher.sources()
