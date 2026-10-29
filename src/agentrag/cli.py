"""agentrag index build <folder> | index reset | agent ask "<question>" [--offline] [--model M]"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from .agent import Deps, build_agent, offline_model  # noqa: E402
from .config import load_config  # noqa: E402
from .embed import get_embedder  # noqa: E402
from .ingest import build_index  # noqa: E402
from .searcher import StoreSearcher  # noqa: E402
from .store import Store, reset_db  # noqa: E402


def _open(cfg):
    embedder = get_embedder(cfg.embed_provider, cfg.embed_model)
    return embedder, Store(cfg.db_path, embedder.name, embedder.dim)


def cmd_index_build(args, cfg) -> int:
    embedder, store = _open(cfg)
    n = build_index(args.folder, cfg, embedder, store, log=print)
    print(f"Done: {n} chunks added, {store.count()} total in {cfg.db_path}")
    return 0


def cmd_index_reset(_args, cfg) -> int:
    print(f"Deleted {cfg.db_path}" if reset_db(cfg.db_path) else "Nothing to delete")
    return 0


def print_trace(messages) -> None:
    """Show the agent loop: every tool call the model made and what came back."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    for m in messages:
        for p in m.parts:
            if isinstance(p, ToolCallPart):
                print(f"  -> tool call   {p.tool_name}({p.args_as_dict()})")
            elif isinstance(p, ToolReturnPart):
                items = p.content if isinstance(p.content, list) else [p.content]
                srcs = [getattr(i, "source", None) or (i.get("source") if isinstance(i, dict) else None) for i in items]
                print(f"  <- tool result {p.tool_name}: {len(items)} item(s) {[s for s in srcs if s]}")


def cmd_ask(args, cfg) -> int:
    embedder, store = _open(cfg)
    model_name = args.model or cfg.agent_model
    if args.offline or (not args.model and "openai" in model_name and not os.environ.get("OPENAI_API_KEY")):
        print("note: no OPENAI_API_KEY (or --offline) -> using the scripted offline model, not a real LLM.\n")
        model, label = offline_model(), "offline-scripted"
    else:
        model, label = model_name, model_name
    agent = build_agent(model)
    result = agent.run_sync(args.question, deps=Deps(StoreSearcher(store, embedder)))
    print(f"Agent run ({label}):")
    print_trace(result.all_messages())
    print(f"\nAnswer:\n{result.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentrag", description=__doc__)
    p.add_argument("--config")
    top = p.add_subparsers(dest="group", required=True)

    idx = top.add_parser("index", help="the index builder (outside the agent)").add_subparsers(dest="cmd", required=True)
    s = idx.add_parser("build", help="ingest -> chunk -> embed -> SQLite")
    s.add_argument("folder")
    s.set_defaults(fn=cmd_index_build)
    idx.add_parser("reset", help="delete the index").set_defaults(fn=cmd_index_reset)

    ag = top.add_parser("agent", help="the agent").add_subparsers(dest="cmd", required=True)
    s = ag.add_parser("ask", help="ask the agent; prints tool calls and the final answer")
    s.add_argument("question")
    s.add_argument("--model", help="Pydantic AI model string, e.g. openai:gpt-4o-mini")
    s.add_argument("--offline", action="store_true", help="use the scripted model (no API key)")
    s.set_defaults(fn=cmd_ask)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    try:
        return args.fn(args, cfg)
    except Exception as e:  # surface model/auth errors as one line, keep traceback with --debug-style env
        if os.environ.get("AGENTRAG_DEBUG"):
            raise
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
