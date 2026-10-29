from pathlib import Path

import pytest
from pydantic_ai.messages import SystemPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from agentrag.agent import SYSTEM_PROMPT, Deps, build_agent, offline_model
from agentrag.config import Config
from agentrag.embed import HashEmbedder
from agentrag.ingest import build_index
from agentrag.searcher import StoreSearcher
from agentrag.store import Store
from agentrag.tools import MAX_K, Snippet, SourceInfo, list_sources, retrieve_snippets

HK = Path(__file__).parent.parent / "examples" / "harbour-lane"


def make_searcher(tmp_path, folder=HK):
    emb = HashEmbedder()
    store = Store(tmp_path / "t.db", emb.name, emb.dim)
    if folder:
        build_index(folder, Config(chunk_size=300, chunk_overlap=50), emb, store)
    return StoreSearcher(store, emb)


# --- tool contract -----------------------------------------------------------
def test_retrieve_on_empty_index_returns_empty_list(tmp_path):
    assert retrieve_snippets(make_searcher(tmp_path, folder=None), "annual leave") == []


def test_retrieve_blank_query_returns_empty_list(tmp_path):
    assert retrieve_snippets(make_searcher(tmp_path), "   ") == []


def test_retrieve_clamps_k(tmp_path):
    s = make_searcher(tmp_path)
    assert len(retrieve_snippets(s, "policy", k=10_000)) <= MAX_K
    assert len(retrieve_snippets(s, "policy", k=-5)) == 1


def test_retrieve_returns_cite_ready_snippets_and_department_filter(tmp_path):
    s = make_searcher(tmp_path)
    out = retrieve_snippets(s, "taxi fare approval HKD 300", k=3, department="finance")
    assert out and all(isinstance(x, Snippet) and x.source and x.department == "finance" for x in out)
    assert retrieve_snippets(s, "anything", department="no-such-dept") == []


def test_list_sources(tmp_path):
    srcs = list_sources(make_searcher(tmp_path))
    assert all(isinstance(s, SourceInfo) for s in srcs)
    assert {"hr", "finance", "compliance", "operations"} <= {s.department for s in srcs}


# --- agent -------------------------------------------------------------------
def test_system_prompt_requires_citations_and_is_sent_to_the_model():
    assert "Cite" in SYSTEM_PROMPT and "source" in SYSTEM_PROMPT
    agent = build_agent(TestModel(call_tools=[]))
    result = agent.run_sync("hi", deps=Deps(searcher=None))
    sent = [p for m in result.all_messages() for p in m.parts if isinstance(p, SystemPromptPart)]
    assert sent and "Cite every fact" in sent[0].content


def test_agent_registers_retrieve_and_list_sources_tools():
    agent = build_agent(TestModel())
    result = agent.run_sync("hi", deps=Deps(searcher=None), model=TestModel(call_tools=[]))
    assert result.output
    assert {"retrieve", "list_sources"} <= set(agent._function_toolset.tools)


def test_agent_calls_retrieve_tool_and_cites_source(tmp_path):
    agent = build_agent(offline_model())
    result = agent.run_sync("年假 申請 要提前幾多日", deps=Deps(StoreSearcher(*_parts(tmp_path))))
    calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)]
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [c.tool_name for c in calls] == ["retrieve"] and returns
    assert "[hk-hq/hr/staff-handbook.md]" in result.output


def test_agent_says_not_found_when_index_is_empty(tmp_path):
    searcher = make_searcher(tmp_path, folder=None)
    result = build_agent(offline_model()).run_sync("anything", deps=Deps(searcher))
    assert "could not find" in result.output


def _parts(tmp_path):
    s = make_searcher(tmp_path)
    return s.store, s.embedder


def test_parallel_tool_calls_do_not_corrupt_shared_connection(tmp_path):
    """Regression: parallel tool calls hit one sqlite connection from several threads at once."""
    from concurrent.futures import ThreadPoolExecutor

    s = make_searcher(tmp_path)
    queries = ["annual leave", "taxi fare HKD 300", "supplier onboarding", "personal data"] * 25
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: retrieve_snippets(s, q, k=3), queries))
    assert all(len(r) == 3 and all(x.source.endswith(".md") for x in r) for r in results)
