# LEARNING — Agent-first RAG（Pydantic AI）

情境：虛構嘅 Harbour Lane 集團（香港總部 + 深圳子公司）。
Scenario: a fictional Hong Kong group with a Shenzhen subsidiary.

---

## 繁體中文（zh-HK）

### 1. Pydantic AI 唔負責 ingest／chunk／embed
呢個係最重要嘅一點。Pydantic AI 係 **agent 框架**：佢管 LLM 迴圈、tool 呼叫、型別驗證、重試。佢**唔會**讀 PDF、切塊、做 embedding、建向量庫。
所以 repo 分兩邊：
- **Index builder**（agent 之外）：`ingest.py → chunk.py → embed.py → store.py`，用 `agentrag index build` 預先跑好。
- **Agent**（`agent.py`）：只知道有個 `retrieve` tool。佢唔知 chunk size，亦唔知用咩 embedding。

### 2. RAG 嘅 R 同 G 喺邊？
- **R（Retrieval）** = 模型決定呼叫 `retrieve(query, k, department)`，tool 去 SQLite 搵段落，連來源一齊傳返。
- **G（Generation）** = 模型根據 tool 結果寫答案，並按 system prompt 引用來源 `[hk-hq/hr/staff-handbook.md]`。

### 3. 點解有時 tool 勝過 chain？
Chain 係「**一定**檢索一次，再生成」。Agent 係「模型**自己決定**」：
- 問「Shenzhen 有咩文件？」→ 呼叫 `list_sources`，唔使向量搜尋。
- 問「報銷同埋年假嘅分別」→ 可能分兩次 `retrieve`，每次不同 query。
- 第一次搵唔到 → 改 query 再搵。
- 打招呼 → 完全唔檢索。
代價：多啲 LLM 呼叫、行為冇咁可預測、要用支援 tool calling 嘅模型。如果每條問題都係同一個查詢，chain 更簡單更平。

### 4. 設計一個「LLM 友好」嘅 retrieve tool（`tools.py`）
- **搵唔到就回傳 `[]`，唔好拋錯**：模型先可以老實講「搵唔到」。
- **限制 k**：避免模型要求一萬段。
- **每段都帶 source**：冇來源就冇得引用。
- **docstring 就係 tool 說明書**：模型靠佢同參數 schema 決定幾時、點樣用 tool，所以要寫清楚（見 `agent.py`）。
- 用 Pydantic model（`Snippet`）做回傳型別，結構清晰。

### 5. 評估 agent（`agentrag eval`）—— 實測學到嘅嘢
17 條中英問題，逐項評分：揀啱 tool、引用來源、事實正確、用問題嘅語言作答、範圍外要拒答。所有答案存喺 `eval/runs/`，可以用 `--rescore` 重新評分而唔使再叫 LLM。實測過程（見 `eval/RESULTS.md`）：
1. **第一個結果係錯嘅**：chunk 800 時每份文件得一個 chunk，`retrieve` 預設 k=4 = 成個語料庫，檢索根本冇影響。**語料細嘅 agent 評估量度唔到檢索。**
2. **先睇答案，再信指標**：睇失敗個案時發現 5 個評分器 bug（中文別名、全形【】引用、檔案路徑、拒答講法）。
3. **最大嘅真問題係語言漂移**：英文問題 gpt-4o-mini 大多用中文答（0.47）。prompt 加規則只得 0.62–0.65；用 Pydantic AI **output validator + `ModelRetry`** 之後每次都係 1.00。**Prompt 規則係請求，validator 係檢查。**
4. **可選參數係陷阱**：模型對「報銷」查詢設咗 `department="hr"`，答案被隱藏（46 次有 9 次用 filter）。tool docstring 加兩句之後變 0/94。**喺 agent 入面，tool docstring 就係 prompt。**
5. **修正有取捨**：之前 `leak-en` 啱，原來係靠估中 `compliance` filter；唔再估之後，答案喺中文段落而 agent 只用英文搜尋，就搵唔到。
6. **喺 agent 入面 embedding 模型差唔多**：香港文件中英夾雜（"Compliance Officer"、"PDPO"），agent 又寫關鍵字式 query，所以 hash 都搵到。

### 常見錯誤
- 以為 Pydantic AI 會幫你切塊、建索引（唔會）。
- System prompt 冇要求引用，或者冇話「搵唔到就唔好估」→ 模型憑記憶作答。
- Tool 搵唔到時拋 exception 令整個 run 失敗。
- 用唔支援 tool calling 嘅模型。
- Sync tool 喺 worker thread 行，而且模型可以**同時**發出幾個 tool call（parallel tool calls）。兩個 thread 同時用同一條 SQLite 連線會令結果錯亂（亂碼、None、API misuse）。只加 `check_same_thread=False` 唔夠，`StoreSearcher` 要用 lock 串行化存取。呢個 bug 係用真模型測試先發現嘅。
- 部門過濾係向量搜尋後先做，冷門部門可能少過 k 個結果。

### 練習
1. 用真模型問「Shenzhen 有咩文件？」，睇 trace 有冇呼叫 `list_sources`。
2. 問「報銷同年假各有咩規定？」，睇模型 call 幾多次 `retrieve`、query 係咩。
3. 將 system prompt 第 3 點刪走，問一條文件冇答案嘅問題，比較行為。
4. 用 `--offline` 對比真模型，理解 tool 迴圈同「智能」係兩回事。

---

## English

### 1. Pydantic AI does not ingest, chunk or embed
This is the key point. Pydantic AI is an **agent framework**: it runs the LLM loop, tool calls, type validation and retries. It does **not** read PDFs, split text, compute embeddings or build a vector store. So the repo has two sides:
- **Index builder** (outside the agent): `ingest.py → chunk.py → embed.py → store.py`, run ahead of time with `agentrag index build`.
- **Agent** (`agent.py`): knows only that a `retrieve` tool exists. It has no idea about chunk size or the embedding model.

### 2. Where are R and G?
- **R (Retrieval)** = the model decides to call `retrieve(query, k, department)`; the tool searches SQLite and returns passages with sources.
- **G (Generation)** = the model writes the answer from tool results and, per the system prompt, cites `[hk-hq/hr/staff-handbook.md]`.

### 3. When do tools beat chains?
A chain says "**always** retrieve once, then generate." An agent lets the **model decide**:
- "What documents does Shenzhen have?" → `list_sources`, no vector search.
- "Compare expense and leave rules" → possibly two `retrieve` calls with different queries.
- First search finds nothing → reformulate and retry.
- A greeting → no retrieval at all.
Cost: more LLM calls, less predictable behaviour, and a model that supports tool calling. If every question needs the same single lookup, a chain is simpler and cheaper.

### 4. Designing an LLM-friendly retrieve tool (`tools.py`)
- **Return `[]` on no results, never raise:** so the model can honestly say "not found".
- **Clamp `k`:** a model shouldn't be able to ask for 10,000 passages.
- **Every passage carries its source:** no source, no citation.
- **The docstring is the tool's manual:** the model uses it plus the argument schema to decide when and how to call the tool (see `agent.py`).
- Return a Pydantic model (`Snippet`) for a clear, typed structure.

### 5. Evaluating the agent (`agentrag eval`): what the runs taught us
17 zh/en questions scored separately for tool choice, citations, facts, answer language and refusals. Every answer is saved in `eval/runs/`, and `--rescore` re-scores saved answers without calling the LLM. What happened (see `eval/RESULTS.md`):
1. **The first headline was wrong:** at chunk 800 each document is one chunk and `retrieve`'s default k=4 returns the whole corpus, so retrieval couldn't matter. **An agent eval on a tiny corpus can't measure retrieval.**
2. **Read answers before trusting metrics:** failures revealed five scorer bugs (Chinese alias, full-width 【】 citations, file paths, refusal phrasings).
3. **The biggest real failure was language drift:** gpt-4o-mini answered most English questions in Chinese (0.47). A prompt rule reached 0.62–0.65; a Pydantic AI **output validator raising `ModelRetry`** reached 1.00 in every run. **A prompt rule is a request; a validator is a check.**
4. **Optional parameters are a trap:** the model set `department="hr"` on an *expense* query and hid the answer (9 of 46 searches used a filter). Two sentences in the tool docstring brought it to 0 of 94. **In an agent, the tool docstring is a prompt.**
5. **Fixes have trade-offs:** an earlier `leak-en` pass came from a lucky `compliance` filter guess; once guessing stopped, the answer (in a Chinese paragraph) is missed because the agent only searches in English.
6. **Inside the agent the embedder barely mattered:** Hong Kong text is code-mixed ("Compliance Officer", "PDPO") and the agent writes keyword-style queries, so even `hash` matches.

### Common mistakes
- Assuming Pydantic AI chunks or indexes for you (it doesn't).
- A system prompt that doesn't demand citations or say "don't guess when nothing is found" → answers from memory.
- A tool that raises on empty results and crashes the run.
- Using a model without tool-calling support.
- Sync tools run in worker threads, and a model may issue **parallel** tool calls. Two threads using one SQLite connection at once corrupt results (garbled text, `None` rows, API misuse). `check_same_thread=False` alone is not enough; `StoreSearcher` serialises access with a lock. This bug only showed up with a real model.
- The department filter runs after vector search, so a rare department can return fewer than k results.

### Exercises
1. With a real model ask "What documents does Shenzhen have?" and check the trace for `list_sources`.
2. Ask "What are the expense and leave rules?" and count the `retrieve` calls and their queries.
3. Delete rule 3 from the system prompt, ask something the documents don't cover, and compare.
4. Compare `--offline` with a real model: a working tool loop is not the same as intelligence.
