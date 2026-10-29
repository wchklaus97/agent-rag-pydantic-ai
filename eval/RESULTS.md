# Agent evaluation - Harbour Lane

17 questions (`eval/agent_qa.jsonl`): 10 lookups, 2 two-topic questions, 2 inventory questions ("what documents exist?") and 3 out-of-scope questions, in zh-HK and English. Model: `openai/gpt-4o-mini` via OpenRouter. Each run scores five separate checks:

- **tool:** right tool (`retrieve` for policy questions, `list_sources` for inventory questions)
- **cite:** every expected source is cited
- **fact:** the key fact is in the answer
- **lang:** the answer is in the question's language
- **refuse:** out-of-scope questions get "couldn't find"

`pass` means every check passed. Every answer is saved in `eval/runs/*.jsonl`, and **all rows below are re-scored with the final scorer** (`agentrag eval --rescore FILE`), so the stages are compared on one ruler.

## Results by stage

| stage | chunk | embedder | runs | pass | lang | cite | fact | refuse | tool calls / q |
|---|---|---|---|---|---|---|---|---|---|
| A. Original prompt | 200 | hash | 1 | 0.47 | 0.47 | 0.92 | 0.92 | 1.00 | 1.1 |
| A. Original prompt | 200 | multilingual | 1 | 0.35 | 0.47 | 0.85 | 0.85 | 1.00 | 1.1 |
| B. + retry & language rules in the prompt | 200 | hash | 2 | 0.59 (0.53–0.65) | 0.62 (0.59–0.65) | 0.96 (0.92–1.00) | 1.00 | 1.00 | 1.5 |
| B. + retry & language rules in the prompt | 200 | multilingual | 2 | 0.56 (0.47–0.65) | 0.65 (0.59–0.71) | 0.85 (0.77–0.92) | 0.85 (0.77–0.92) | 1.00 | 1.5 |
| C. + language output validator (ModelRetry) | 200 | hash | 3 | 0.94 (0.88–1.00) | 1.00 | 0.92 (0.85–1.00) | 0.92 (0.85–1.00) | 1.00 | 1.5 |
| C. + language output validator (ModelRetry) | 200 | multilingual | 3 | 0.92 (0.82–1.00) | 1.00 | 0.90 (0.77–1.00) | 0.92 (0.85–1.00) | 1.00 | 1.5 |
| D. + tool docstring: no guessed department, one search per topic | 200 | hash | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.5 |
| D. + tool docstring: no guessed department, one search per topic | 200 | multilingual | 2 | 0.94 | 1.00 | 0.92 | 0.92 | 1.00 | 1.5 |
| B at chunk 800 (each search returns the whole corpus) | 800 | hash | 1 | 0.65 | 0.65 | 1.00 | 1.00 | 1.00 | 1.5 |
| B at chunk 800 (each search returns the whole corpus) | 800 | multilingual | 1 | 0.71 | 0.76 | 0.92 | 0.92 | 1.00 | 1.5 |

Ranges are min–max across repeated identical runs. Each question is worth 0.06, and identical reruns moved by up to 0.18, so read single-run differences with care.

## What we learned, in the order we learned it

1. **The first headline was wrong.** The first run (chunk 800) scored `hash` 1.00 and multilingual 0.94. That looked perfect, but with 800-character chunks each of the 4 documents is a single chunk, and `retrieve` defaults to k=4. Every search returned the **whole corpus**, so retrieval could not matter. The run also had no language check. **An agent eval on a tiny corpus cannot measure retrieval.** We used chunk 200 (15 chunks) from then on.
2. **Read the answers before trusting a metric.** Reading the failures turned up five scorer bugs:
   - a missing Chinese alias (財務經理 for Finance Manager)
   - full-width 【】 citations
   - `.md` paths counted as English text
   - refusal phrasings the checker didn't recognise, such as "couldn't locate" and Simplified 没有找到

   `--rescore` re-applies a fixed scorer to saved answers without new LLM calls.
3. **Language drift was the biggest real failure.** The documents are mostly Chinese, and gpt-4o-mini answered most English questions in Chinese: lang was 0.47 with the original prompt. An explicit rule in the prompt only reached 0.62–0.65 (B). A Pydantic AI **output validator** that raises `ModelRetry` with feedback reached **1.00 in every run** (C, D). **A prompt rule is a request; a validator is a check.** On the last allowed retry the validator accepts the answer rather than crashing the run.
4. **A retry rule makes the agent search again** (tool calls per question rose from 1.1 to 1.5), but it retries **in the question's language**. It never once searched in the other language, even though the prompt suggests it.
5. **Optional tool filters are a trap.** In stage C the model set `department="hr"` on an *expense* query, which hid the answer. 9 of 46 searches used a filter. Two sentences in the tool docstring ("only set department when the user names one", "one search per topic") cut that to **0 of 94**, and the two-topic question passed every time. **In an agent, the tool docstring is a prompt.**
6. **Fixes have trade-offs.** The one stage-C pass of `leak-en` with the multilingual embedder came from a lucky `department="compliance"` guess. Once guessing stopped, that case fails consistently: the answer is in a Chinese paragraph, and the agent only searches in English.
7. **Inside the agent, the embedder hardly mattered** here (stage C: `hash` 0.94, multilingual 0.92). Hong Kong business text is code-mixed: Chinese paragraphs carry English terms such as "Personal data (PDPO)", "Compliance Officer" and "Finance Manager". The agent writes keyword-style queries, so a keyword embedder still matches. Contrast this with `local-rag-sqlite`'s raw retrieval eval, where the user's full question goes straight in and the multilingual model was clearly better for English questions.

## Still open

- **`leak-en` with the multilingual embedder.** This needs cross-lingual or keyword (hybrid) search inside the tool. Follow-up in [hybrid-search-rag](https://github.com/wchklaus97/hybrid-search-rag/blob/main/eval/RESULTS.md): replaying this agent's recorded queries, hybrid search plus reranking finds the answer for 4 of 10 queries (vector search: 0 of 10). That is a partial fix; the rest needs cross-lingual query expansion.
- **Simplified vs Traditional Chinese.** One answer drifted to Simplified, and the validator only checks Chinese vs English.
- **The refusal check is phrase-based.** An LLM judge would be more robust.
- **Scale.** This is one model and 17 questions; grow the set before tuning further.

## Reproduce

```bash
uv sync --extra local
# stage D (current code), chunk 200, both embedders; costs a few US cents on gpt-4o-mini
uv run agentrag eval --model openai:openai/gpt-4o-mini --embed hash,local --chunk-size 200 --dump my-run.jsonl
uv run agentrag eval --rescore eval/runs/D-tool-docstring-chunk200-run1.jsonl   # re-score saved answers, no API calls
```
