"""Configuration. Index-builder settings (chunking, embedding) and agent settings are separate
concerns: the agent only ever calls a tool, it knows nothing about chunk size."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    db_path: str = "agentrag.db"
    chunk_size: int = 800
    chunk_overlap: int = 120
    embed_provider: str = "local"
    embed_model: str = ""
    top_k: int = 4
    agent_model: str = "openai:gpt-4o-mini"


_ENV = {"RAG_DB": "db_path", "EMBED_PROVIDER": "embed_provider", "EMBED_MODEL": "embed_model", "AGENT_MODEL": "agent_model"}


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("AGENTRAG_CONFIG", "agentrag.toml"))
    values = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in values.items() if k in known})
    for env, attr in _ENV.items():
        if os.environ.get(env):
            setattr(cfg, attr, os.environ[env])
    return cfg
