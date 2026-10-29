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


def cmd_eval(args, cfg) -> int:
    import tempfile
    from pathlib import Path

    from .evaluate import load_qa, run_case, summarize, to_markdown

    import json
    from dataclasses import replace

    qa = load_qa(args.qa)
    if args.rescore:
        from .evaluate import rescore

        table = to_markdown(rescore(args.rescore, qa))
        print(table)
        if args.out:
            Path(args.out).write_text(table, encoding="utf-8")
        return 0
    model = offline_model() if args.offline else (args.model or cfg.agent_model)
    if args.chunk_size:
        cfg = replace(cfg, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap or int(args.chunk_size * 0.15))
    dump = open(args.dump, "a", encoding="utf-8") if args.dump else None
    summaries = []
    for name in args.embed.split(","):
        embedder = get_embedder(name.strip())
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "eval.db", embedder.name, embedder.dim)
            n_chunks = build_index(args.folder, cfg, embedder, store)
            deps, agent = Deps(StoreSearcher(store, embedder)), build_agent(model)
            label = (
                f"{'offline-scripted' if args.offline else model} + `{embedder.name}`, "
                f"chunk {cfg.chunk_size}/{cfg.chunk_overlap} ({n_chunks} chunks)"
            )
            cases = []
            for item in qa:
                case = run_case(agent, deps, item)
                cases.append(case)
                if dump:
                    dump.write(json.dumps({"run": label, "id": item.id, "tools": case.tool_calls, "tool_args": case.tool_args,
                                           "checks": case.checks, "error": case.error, "output": case.output},
                                          ensure_ascii=False) + "\n")
                mark = "PASS" if case.passed else "FAIL " + (case.error or ",".join(k for k, v in case.checks.items() if not v))
                print(f"  [{name.strip():>5}] {item.id:<18} tools={case.tool_calls}  {mark}")
            store.close()
        summaries.append(summarize(label, cases))
    if dump:
        dump.close()
    table = to_markdown(summaries)
    print("\n" + table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"wrote {args.out}")
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

    s = top.add_parser("eval", help="score the agent: tool choice, citations, facts, refusals")
    s.add_argument("--qa", default="eval/agent_qa.jsonl")
    s.add_argument("--folder", default="examples/harbour-lane")
    s.add_argument("--embed", default="hash", help="comma list of embedders to compare, e.g. hash,local")
    s.add_argument("--model", help="Pydantic AI model string")
    s.add_argument("--offline", action="store_true", help="scripted model (tests the harness, not an LLM)")
    s.add_argument("--chunk-size", type=int, help="override chunk_size for the eval index")
    s.add_argument("--chunk-overlap", type=int, help="override chunk_overlap (default 15%% of size)")
    s.add_argument("--dump", help="append every case (answer, tool calls, checks) to this JSONL file")
    s.add_argument("--rescore", metavar="DUMP", help="re-score saved answers with the current scorer (no LLM calls)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_eval)

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
