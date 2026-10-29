"""The agent: R and G of RAG, as an agent loop.

  R (retrieval)  = the `retrieve` tool the model chooses to call (and may call several times
                   with different queries, or not at all for small talk)
  G (generation) = the agent's LLM writing the final answer from tool results

Pydantic AI does NOT ingest, chunk or embed anything. Those happen in the index builder
(ingest.py / chunk.py / embed.py / store.py) BEFORE the agent ever runs. The agent only sees
the tool's name, its docstring/arguments schema, and whatever the tool returns.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .tools import Searcher, Snippet, SourceInfo, list_sources as _list_sources, retrieve_snippets

SYSTEM_PROMPT = """\
You are an assistant for the internal documents of a Hong Kong company group.
Rules:
1. For any question about company policy or procedure, call the `retrieve` tool first. Do not answer from memory.
2. Cite every fact with its source in square brackets taken from the tool result, e.g. [hk-hq/hr/staff-handbook.md].
3. If `retrieve` returns an empty list or nothing relevant, say you could not find it. Do not guess.
4. Use `list_sources` when the user asks what documents or departments exist.
5. Answer in the language of the question (Cantonese/Traditional Chinese or English).
"""


@dataclass
class Deps:
    searcher: Searcher


def build_agent(model=None) -> Agent[Deps, str]:
    agent = Agent(model, deps_type=Deps, system_prompt=SYSTEM_PROMPT, defer_model_check=True)

    @agent.tool
    def retrieve(ctx: RunContext[Deps], query: str, k: int = 4, department: str | None = None) -> list[Snippet]:
        """Search the company documents and return the best-matching passages with their sources.

        Args:
            query: What to look for, phrased as a short search query.
            k: How many passages to return (1-10).
            department: Optional department folder to restrict to, e.g. "hr", "finance", "compliance", "operations".
        """
        return retrieve_snippets(ctx.deps.searcher, query, k, department)

    @agent.tool
    def list_sources(ctx: RunContext[Deps]) -> list[SourceInfo]:
        """List every indexed document with its entity, department and chunk count."""
        return _list_sources(ctx.deps.searcher)

    return agent


def _as_dicts(content) -> list[dict]:
    items = content if isinstance(content, list) else [content]
    return [i.model_dump() if hasattr(i, "model_dump") else i for i in items if i]


def offline_model() -> FunctionModel:
    """A scripted 'LLM' for running without an API key: calls `retrieve` once with the user's
    question, then quotes the top passage with a citation. It shows the agent/tool loop, not intelligence."""

    def script(messages, info: AgentInfo) -> ModelResponse:
        returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_name == "retrieve"]
        if not returned:
            question = next(p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart))
            return ModelResponse(parts=[ToolCallPart("retrieve", {"query": str(question), "k": 3})])
        snippets = _as_dicts(returned[-1].content)
        if not snippets:
            return ModelResponse(parts=[TextPart("I could not find this in the indexed documents.")])
        top = snippets[0]
        return ModelResponse(parts=[TextPart(f"[offline demo, no LLM] Closest passage: {' '.join(top['text'].split())[:300]} [{top['source']}]")])

    return FunctionModel(script)
