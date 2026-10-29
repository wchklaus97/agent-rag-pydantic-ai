"""Stage 3 - Embedding.

An embedder maps a list of strings to a list of float vectors, one per string.
That is all. It does not split text (see chunk.py) and it does not know about
your database (see store.py).

Providers:
  local   sentence-transformers (default; needs `pip install -e .[local]`)
  hash    dependency-free hashed bag-of-words; NOT semantic, for tests/offline demos
  openai  OPENAI_API_KEY      (text-embedding-3-small)
  gemini  GEMINI_API_KEY      (gemini-embedding-001)
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys

from ._http import post_json

DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        if not hasattr(self, "_dim"):
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim


class HashEmbedder(Embedder):
    """Hashed bag of words (+ CJK character bigrams), L2-normalised.

    Only matches shared vocabulary, so it is much weaker than a real model, but it
    needs no downloads, which makes tests and the first run instant.
    """

    _TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]")

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.name = f"hash:{dim}"

    def _features(self, text: str) -> list[str]:
        toks = self._TOKEN.findall(text.lower())
        is_cjk = lambda t: "一" <= t <= "鿿"  # noqa: E731
        cjk_bigrams = [a + b for a, b in zip(toks, toks[1:]) if is_cjk(a) and is_cjk(b)]
        return toks + cjk_bigrams

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self._dim
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                vec[h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model)
        self.name = f"local:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str):
        self._key = os.environ["OPENAI_API_KEY"]
        self._model = model
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/embeddings"
        data = post_json(url, {"model": self._model, "input": texts}, {"Authorization": f"Bearer {self._key}"})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


class GeminiEmbedder(Embedder):
    def __init__(self, model: str):
        self._key = os.environ["GEMINI_API_KEY"]
        self._model = model
        self.name = f"gemini:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:batchEmbedContents"
        reqs = [{"model": f"models/{self._model}", "content": {"parts": [{"text": t}]}} for t in texts]
        data = post_json(url, {"requests": reqs}, {"x-goog-api-key": self._key})
        return [e["values"] for e in data["embeddings"]]


def get_embedder(provider: str, model: str = "") -> Embedder:
    provider = provider.lower()
    if provider == "hash":
        return HashEmbedder()
    if provider == "local":
        try:
            return SentenceTransformerEmbedder(model or DEFAULT_LOCAL_MODEL)
        except ImportError:
            print(
                "warning: sentence-transformers not installed; falling back to the 'hash' embedder "
                "(keyword-ish, not semantic). Install real embeddings with: pip install -e '.[local]'",
                file=sys.stderr,
            )
            return HashEmbedder()
    if provider == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    if provider == "gemini":
        return GeminiEmbedder(model or "gemini-embedding-001")
    raise ValueError(f"unknown embed_provider {provider!r} (local | hash | openai | gemini)")
