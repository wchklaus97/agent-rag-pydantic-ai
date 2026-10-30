from pathlib import Path

from agentrag.agent import Deps, build_agent, offline_model
from agentrag.config import Config
from agentrag.embed import HashEmbedder
from agentrag.evaluate import AgentQA, load_qa, run_case, score, summarize, to_markdown
from agentrag.ingest import build_index
from agentrag.searcher import StoreSearcher
from agentrag.store import Store

ROOT = Path(__file__).parent.parent
QA = load_qa(ROOT / "eval" / "agent_qa.jsonl")


def _qa(**kw):
    base = dict(id="x", lang="en", category="lookup", q="q", expect_tool="retrieve", sources=["a.md"], facts_any=["15"])
    return AgentQA(**{**base, **kw})


def test_qa_set_covers_every_category_and_real_sources():
    assert {q.category for q in QA} == {"lookup", "multi", "inventory", "out_of_scope"}
    files = {p.relative_to(ROOT / "examples" / "harbour-lane").as_posix() for p in (ROOT / "examples" / "harbour-lane").rglob("*.md")}
    for q in QA:
        for src in q.sources:
            assert any(src in f for f in files), f"{q.id}: {src}"


def test_score_lookup_needs_tool_citation_and_fact():
    qa = _qa()
    assert score(qa, "15 days [a.md]", ["retrieve"]) == {"tool": True, "lang": True, "cite": True, "fact": True}
    assert score(qa, "15 days", ["retrieve"])["cite"] is False
    assert score(qa, "15 days [a.md]", [])["tool"] is False


def test_score_out_of_scope_rejects_confident_citation():
    qa = _qa(category="out_of_scope", sources=[], facts_any=[])
    assert score(qa, "I could not find this in the documents.", ["retrieve"])["refuse"] is True
    assert score(qa, "搵唔到相關資料。", ["retrieve"])["refuse"] is True
    assert score(qa, "Crypto is banned [hk-hq/compliance/x.md]", ["retrieve"])["refuse"] is False


def test_offline_run_scores_and_catches_wrong_tool(tmp_path):
    emb = HashEmbedder()
    store = Store(tmp_path / "t.db", emb.name, emb.dim)
    build_index(ROOT / "examples" / "harbour-lane", Config(), emb, store)
    agent, deps = build_agent(offline_model()), Deps(StoreSearcher(store, emb))
    cases = [run_case(agent, deps, q) for q in QA]
    by_id = {c.id: c for c in cases}
    assert by_id["sz-docs-en"].checks["tool"] is False  # scripted model never calls list_sources
    s = summarize("offline", cases)
    assert 0 <= s.passed <= 1 and set(s.by_category) == {"lookup", "multi", "inventory", "out_of_scope"}
    assert "| offline |" in to_markdown([s]) and s.failures


def test_cli_eval_offline_with_chunk_override_and_dump(tmp_path, monkeypatch):
    import json

    from agentrag.cli import main

    monkeypatch.chdir(ROOT)
    dump, out = tmp_path / "cases.jsonl", tmp_path / "results.md"
    assert main(["eval", "--offline", "--chunk-size", "200", "--dump", str(dump), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(QA) and all("chunk 200/30" in r["run"] for r in rows)
    assert {"id", "tools", "tool_args", "checks", "output"} <= set(rows[0])
    assert rows[0]["tool_args"][0]["tool"] == "retrieve" and "query" in rows[0]["tool_args"][0]
    assert "chunk 200/30 (" in out.read_text(encoding="utf-8")


def test_language_check_ignores_citations_and_catches_drift():
    en, zh = _qa(), _qa(lang="zh")
    assert score(en, "You get 15 days [hk-hq/hr/staff-handbook.md]", ["retrieve"])["lang"] is True
    assert score(en, "每年有 15 日年假 [a.md]", ["retrieve"])["lang"] is False  # English question, Chinese answer
    assert score(zh, "每年有 15 日年假 [hk-hq/hr/staff-handbook.md]", ["retrieve"])["lang"] is True


def test_refusal_allows_citing_searched_docs_but_not_confident_claims():
    qa = _qa(category="out_of_scope", sources=[], facts_any=[])
    assert score(qa, "似乎並沒有具體提到泊車津貼。[hk-hq/hr/staff-handbook.md]", ["retrieve"])["refuse"] is True
    assert score(qa, "加密貨幣交易被禁止【hk-hq/compliance/x.md】", ["retrieve"])["refuse"] is False


def test_rescore_reapplies_current_scorer_to_saved_answers(tmp_path):
    import json

    from agentrag.evaluate import rescore

    dump = tmp_path / "d.jsonl"
    rows = [
        {"run": "r", "id": "leave-days-en", "tools": ["retrieve"], "checks": {}, "error": "",
         "output": "Full-time staff get 15 days. [hk-hq/hr/staff-handbook.md]"},
        {"run": "r", "id": "crypto-en", "tools": ["retrieve"], "checks": {}, "error": "", "output": "我找不到相關資料。"},
    ]
    dump.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (s,) = rescore(dump, QA)
    assert s.n == 2 and s.rates["lang"] == 0.5 and s.rates["refuse"] == 1.0


def test_system_prompt_requires_retry_and_same_language():
    from agentrag.agent import SYSTEM_PROMPT

    assert "search AGAIN" in SYSTEM_PROMPT and "same language" in SYSTEM_PROMPT


def test_language_check_ignores_markdown_link_paths():
    zh = _qa(lang="zh")
    answer = "1. **合規部** (Compliance) - 文件: [個人資料政策](hk-hq/compliance/personal-data-and-aml.md)"
    assert score(zh, answer, ["list_sources"])["lang"] is True


def test_refusal_markers_cover_phrasings_seen_in_real_runs():
    qa = _qa(category="out_of_scope", sources=[], facts_any=[])
    for answer in ["I couldn't locate specific information regarding this year's bonus.",
                   "目前在檔案中没有找到关于员工泊车位津贴的具体信息。",
                   "根據檔案，沒有明確提到有關員工泊車位津貼的資訊。",
                   # hybrid-search-rag agent check, 2026-10-30
                   "根據目前所找到的資料，無法確定公司是否提供員工泊車位津貼。所有相關文檔似乎沒有直接提到泊車位津貼的政策或規定。"]:
        assert score(qa, answer, ["retrieve"])["refuse"] is True, answer
