"""Agent evaluation: not "is retrieval good?" but "does the AGENT behave?"

For each question we run the full agent loop and score four separate things, because an
agent can fail in ways a retrieval metric never sees:

  tool      did it call the expected tool? (retrieve for policy questions, list_sources for
            "what documents exist" questions)
  cite      does the answer cite every expected source? (the system prompt demands citations)
  fact      does the answer contain the key fact? (numbers/names, language-neutral)
  refuse    for out-of-scope questions: does it say it couldn't find it? (citing the documents
            it searched is fine; a confident answer without a refusal phrase fails)
  lang      is the answer in the language of the question? (the documents are mostly
            Chinese, so models drift into Chinese for English questions)

Categories: lookup (one doc), multi (needs two docs), inventory, out_of_scope.

Limitation: `refuse` matches refusal phrases, so new phrasings need adding (it has missed
"couldn't locate" and Simplified "没有找到" before). An LLM judge would be more robust.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.messages import ToolCallPart

from .lang import is_chinese

REFUSAL_MARKERS = (
    "could not find", "couldn't find", "can't find", "cannot find", "unable to find", "not find",
    "no information", "don't know", "do not know", "not available", "not mentioned", "no relevant",
    "does not mention", "doesn't mention", "no specific", "not specified",
    "couldn't locate", "could not locate", "not able to find", "been able to find", "not explicitly",
    "搵唔到", "找不到", "找唔到", "冇資料", "唔知", "不知道", "未能找到", "無法找到",
    "沒有相關", "沒有提及", "沒有提到", "並沒有", "沒有具體", "未有提及", "冇提及", "冇提到", "並無",
    "沒有找到", "没有找到", "未找到", "沒有關於", "没有关于",  # Traditional and Simplified
    "沒有明確", "没有明确", "未有明確", "冇明確",
    "無法確定", "无法确定", "沒有直接", "没有直接", "cannot determine", "can't determine", "unable to determine",
)

@dataclass(frozen=True)
class AgentQA:
    id: str
    lang: str
    category: str  # lookup | multi | inventory | out_of_scope
    q: str
    expect_tool: str
    sources: list[str]
    facts_any: list[str]


@dataclass
class CaseResult:
    id: str
    category: str
    checks: dict[str, bool]  # only the checks that apply to this case
    tool_calls: list[str]
    seconds: float
    output: str
    error: str = ""
    tool_args: list[dict] = field(default_factory=list)  # what the model actually asked for

    @property
    def passed(self) -> bool:
        return not self.error and all(self.checks.values())


@dataclass
class Summary:
    label: str
    n: int
    passed: float
    rates: dict[str, float] = field(default_factory=dict)  # per check
    by_category: dict[str, float] = field(default_factory=dict)  # pass rate
    avg_tool_calls: float = 0.0
    avg_seconds: float = 0.0
    failures: list[str] = field(default_factory=list)


def load_qa(path: str | Path) -> list[AgentQA]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [AgentQA(**json.loads(line)) for line in lines if line.strip()]


def score(qa: AgentQA, output: str, tool_calls: list[str]) -> dict[str, bool]:
    checks = {"tool": qa.expect_tool in tool_calls, "lang": is_chinese(output) == (qa.lang == "zh")}
    if qa.category == "out_of_scope":
        low = output.lower()
        checks["refuse"] = any(m in low for m in REFUSAL_MARKERS)
        return checks
    if qa.sources:
        checks["cite"] = all(src in output for src in qa.sources)
    if qa.facts_any:
        checks["fact"] = any(f in output for f in qa.facts_any)
    return checks


def run_case(agent, deps, qa: AgentQA) -> CaseResult:
    start = time.perf_counter()
    try:
        result = agent.run_sync(qa.q, deps=deps)
    except Exception as e:  # a crash is a failed case, not a crashed eval
        return CaseResult(qa.id, qa.category, {}, [], time.perf_counter() - start, "", f"{type(e).__name__}: {e}")
    parts = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)]
    calls = [p.tool_name for p in parts]
    return CaseResult(qa.id, qa.category, score(qa, result.output, calls), calls, time.perf_counter() - start,
                      result.output, tool_args=[{"tool": p.tool_name, **p.args_as_dict()} for p in parts])


def summarize(label: str, cases: list[CaseResult]) -> Summary:
    n = len(cases)
    names = sorted({c for case in cases for c in case.checks})
    rates = {}
    for name in names:
        applicable = [case for case in cases if name in case.checks]
        rates[name] = sum(case.checks[name] for case in applicable) / len(applicable)
    cats = sorted({c.category for c in cases})
    return Summary(
        label=label,
        n=n,
        passed=sum(c.passed for c in cases) / n,
        rates=rates,
        by_category={cat: sum(c.passed for c in cases if c.category == cat) / sum(c.category == cat for c in cases) for cat in cats},
        avg_tool_calls=sum(len(c.tool_calls) for c in cases) / n,
        avg_seconds=sum(c.seconds for c in cases) / n,
        failures=[
            f"{c.id}: " + (c.error or ", ".join(k for k, v in c.checks.items() if not v)) for c in cases if not c.passed
        ],
    )


def to_markdown(summaries: list[Summary]) -> str:
    checks = sorted({k for s in summaries for k in s.rates})
    cats = sorted({k for s in summaries for k in s.by_category})
    head = (
        "| run | pass | " + " | ".join(checks) + " | " + " | ".join(f"pass: {c}" for c in cats)
        + " | tool calls / q | s / q |"
    )
    sep = "|" + "---|" * (4 + len(checks) + len(cats))
    rows = [
        f"| {s.label} | {s.passed:.2f} | "
        + " | ".join(f"{s.rates.get(c, 0):.2f}" for c in checks) + " | "
        + " | ".join(f"{s.by_category.get(c, 0):.2f}" for c in cats)
        + f" | {s.avg_tool_calls:.1f} | {s.avg_seconds:.1f} |"
        for s in summaries
    ]
    out = f"{summaries[0].n} questions per run.\n\n" + "\n".join([head, sep, *rows]) + "\n"
    for s in summaries:
        if s.failures:
            out += f"\n**Failures - {s.label}:** " + "; ".join(s.failures) + "\n"
    return out


def rescore(dump_path: str | Path, qa: list[AgentQA], prefix: str = "") -> list[Summary]:
    """Re-apply the CURRENT scorer to answers saved with --dump: fix the ruler, not re-buy the run."""
    by_id = {q.id: q for q in qa}
    runs: dict[str, list[CaseResult]] = {}
    for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        item = by_id[r["id"]]
        case = CaseResult(r["id"], item.category, score(item, r["output"], r["tools"]) if not r["error"] else {},
                          r["tools"], 0.0, r["output"], r["error"])
        runs.setdefault(r["run"], []).append(case)
    return [summarize(prefix + label, cases) for label, cases in runs.items()]
