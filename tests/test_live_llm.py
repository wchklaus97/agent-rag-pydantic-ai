"""Opt-in tests: the real agent loop against a real LLM (OpenAI or OpenAI-compatible, e.g. OpenRouter).

Skipped by default. Run with:
    RUN_LIVE_LLM=1 LIVE_MODEL=openai/gpt-4o-mini uv run pytest tests/test_live_llm.py
The model must support tool calling.
"""

import os
from pathlib import Path

import pytest
from pydantic_ai.messages import ToolCallPart

from agentrag.agent import Deps, build_agent
from agentrag.config import Config
from agentrag.embed import HashEmbedder
from agentrag.ingest import build_index
from agentrag.searcher import StoreSearcher
from agentrag.store import Store

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_LLM") and os.environ.get("OPENAI_API_KEY")),
    reason="live LLM test: set RUN_LIVE_LLM=1 and OPENAI_API_KEY",
)
MODEL = "openai:" + os.environ.get("LIVE_MODEL", "openai/gpt-4o-mini")
HK = Path(__file__).parent.parent / "examples" / "harbour-lane"


@pytest.fixture(scope="module")
def deps(tmp_path_factory):
    emb = HashEmbedder()
    store = Store(tmp_path_factory.mktemp("live") / "t.db", emb.name, emb.dim)
    build_index(HK, Config(), emb, store)
    return Deps(StoreSearcher(store, emb))


def _calls(result):
    return [p.tool_name for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)]


def test_agent_retrieves_and_cites(deps):
    result = build_agent(MODEL).run_sync("Who must approve a taxi fare over HKD 300?", deps=deps)
    assert "retrieve" in _calls(result)
    assert "expense-claims.md" in result.output


def test_agent_uses_list_sources_for_inventory_question(deps):
    result = build_agent(MODEL).run_sync("What documents does the Shenzhen subsidiary have?", deps=deps)
    assert "list_sources" in _calls(result)
    assert "supplier-onboarding" in result.output.lower().replace(" ", "-")


def test_agent_parallel_retrieves_for_two_topic_question(deps):
    """Exercises parallel tool calls (the bug that shared one SQLite connection across threads)."""
    result = build_agent(MODEL).run_sync("報銷同年假各有咩規定？", deps=deps)
    assert _calls(result).count("retrieve") >= 1
    assert "staff-handbook.md" in result.output or "expense-claims.md" in result.output
