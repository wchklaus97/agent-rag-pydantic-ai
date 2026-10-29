# agent-rag-pydantic-ai

**Theme: agent-first RAG. Retrieval is a typed *tool* the agent decides to call, not a fixed step in a chain.**

A small, runnable demo with [Pydantic AI](https://ai.pydantic.dev/). The index lives outside the agent (SQLite + sqlite-vec). The agent has a `retrieve` tool (and a `list_sources` tool), must cite the sources the tool returns, and prints every tool call so you can watch the loop.

> The sample corpus is a **fictional** Hong Kong group, *Harbour Lane Group Limited* (HK head office + Shenzhen subsidiary), with bilingual 繁體/廣東話 + English documents under `entity/department/`. Policies are invented for teaching; nothing here is legal or HR advice.

## What you will learn

1. **Where "R" and "G" live in an agent:** R = the `retrieve` tool call, G = the LLM writing from tool results.
2. **Pydantic AI does not ingest, chunk or embed.** Those happen in the index builder before the agent runs.
3. **When tools beat chains:** the model can search several times, filter by department, skip retrieval for small talk, or ask what documents exist.
4. How to design a retrieval tool an LLM can use safely (empty-safe, clamped, cite-ready).

## The part you can lift out

[`src/agentrag/tools.py`](src/agentrag/tools.py): the retrieve tool's contract (`Snippet`, `Searcher` protocol, `retrieve_snippets`, `list_sources`). It needs only `pydantic`; wrap it in any agent framework, back it with any search.

```mermaid
flowchart LR
    subgraph Offline["Index builder (outside the agent) - agentrag index build"]
        D["PDF / Markdown<br/>entity/department/"] --> C["chunk.py"] --> E["embed.py"] --> DB[("SQLite + sqlite-vec")]
    end
    U["User"] --> A["Pydantic AI Agent<br/>system prompt: must cite"]
    A -- "tool call: retrieve(query, k, department)" --> T["retrieve tool<br/>tools.py"]
    T --> DB
    DB -- "snippets + sources" --> T
    T -- "tool result" --> A
    A --> ANS["Answer with [source] citations"]
```

### Contrast: agent-first vs chain-first

| | This repo (Pydantic AI agent) | LangChain LCEL RAG chain | LlamaIndex QueryEngine |
|---|---|---|---|
| Who decides to retrieve? | **The model**, per question | You: retriever is a fixed step in the pipeline | You: the engine always retrieves then synthesises |
| Retrieval count | 0, 1 or many calls, each with its own query | Exactly once (unless you build a loop) | Once per query (sub-question engines add more) |
| Ingest / chunk / embed | **Not provided**, you build the index | Loaders + splitters + vector stores in the framework | Readers + node parsers + indices in the framework |
| Strength | Flexible, multi-step, easy to add other tools | Predictable, cheap, easy to reason about | Batteries-included indexing and query tuning |
| Weakness | Less predictable, more LLM calls, needs a capable model | Can't adapt (bad query = bad answer) | Heavier abstractions |

Neither is "better": a chain is right when every question needs the same one lookup; an agent is right when questions vary or need several lookups or other tools.

## Quickstart

```bash
uv sync
uv run agentrag index build examples/harbour-lane       # offline index, hash embedder fallback (warns)
uv run agentrag agent ask "年假申請要提前幾多日？" --offline   # scripted model, no API key
```

With a real model:

```bash
export OPENAI_API_KEY=sk-...
uv run agentrag agent ask "Who must approve a taxi fare over HKD 300?"
uv run agentrag agent ask "What documents does the Shenzhen subsidiary have?"   # should call list_sources
```

The output shows each step:

```
Agent run (openai:gpt-4o-mini):
  -> tool call   retrieve({'query': 'taxi fare approval', 'k': 4, 'department': 'finance'})
  <- tool result retrieve: 2 item(s) ['hk-hq/finance/expense-claims.md', ...]
Answer: ... [hk-hq/finance/expense-claims.md]
```

For real semantic retrieval install the local embedding model (`uv sync --extra local`) and rebuild the index; switching embedders needs a rebuild and the store refuses mismatches.

### Models

`AGENT_MODEL` or `--model` takes any [Pydantic AI model string](https://ai.pydantic.dev/models/). Default `openai:gpt-4o-mini`.

- **OpenAI-compatible APIs** (Ollama, vLLM, gateways): set `OPENAI_BASE_URL` (e.g. `http://localhost:11434/v1`) and `OPENAI_API_KEY` (any value for Ollama), then `--model openai:<model-name>`. The agent needs a model with **tool calling** support.
- Other providers (Anthropic, Gemini, …) need their extra: `uv add "pydantic-ai-slim[anthropic]"`, then e.g. `--model anthropic:<model>`.
- `--offline` (or no `OPENAI_API_KEY`) uses a scripted `FunctionModel` that calls `retrieve` once and quotes the top passage. It shows the loop, not intelligence.

## Commands

| Command | What it does |
|---|---|
| `agentrag index build <folder>` | Ingest → chunk → embed → store (the index builder; the agent is not involved) |
| `agentrag index reset` | Delete the index |
| `agentrag agent ask "<q>" [--model M] [--offline]` | Run the agent; prints tool calls, tool results, final answer |

Config in `agentrag.toml` (env: `RAG_DB`, `EMBED_PROVIDER`, `EMBED_MODEL`, `AGENT_MODEL`).

## Layout

```
src/agentrag/  tools.py (standalone contract) searcher.py (SQLite adapter) agent.py (prompt + tools)
               ingest.py chunk.py embed.py store.py (index builder, same ideas as local-rag-sqlite)
               cli.py config.py
examples/harbour-lane/   fictional HK group
tests/         tool safety (empty / blank / clamped), citation prompt, agent calls the tool (no API key needed)
LEARNING.md    繁體中文 (zh-HK) + English
```

## Run with OpenRouter (or any OpenAI-compatible API)

Keep the key in your shell (e.g. `~/.zshrc`), never in the repo. `.env` files are git-ignored.

```bash
export OPENROUTER_API_KEY="..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

```bash
uv run agentrag agent ask "報銷同年假各有咩規定？" --model openai:openai/gpt-4o-mini
```

Model names use OpenRouter's `vendor/model` form. Embeddings stay local (`local` or `hash`); this demo does not send document embeddings to OpenRouter.

### Live tests (opt-in, cost a few cents)

```bash
RUN_LIVE_LLM=1 LIVE_MODEL=openai/gpt-4o-mini uv run pytest tests/test_live_llm.py -v
```

They are skipped in a normal `uv run pytest`. They check grounded answers, citations, filters, tool choice (`retrieve` vs `list_sources`) and parallel tool calls.

## Contributing

Good first issues: a `search_by_entity` tool, streaming output, an eval set (question → expected source) scored against the trace, a re-ranking tool. Keep `tools.py` free of agent-framework imports. MIT licensed.

## License

MIT
