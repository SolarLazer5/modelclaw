# config.json 字段说明

> 项目：python-model-cli · 配置文件字段速查
> 配置文件由七个分组构成，分别对应代码中的不同模块。

## 一、api —— API 连接配置

所有和「怎么连上 LLM 服务」相关的参数。

| 字段 | 作用 | 说明 |
|---|---|---|
| `base_url` | API 服务器地址 | 硅基流动（SiliconFlow）的 OpenAI 兼容接口。想换服务商时改这里 + `model` 即可，代码不用动 |
| `api_key` | 身份凭证 | 调用 API 的钥匙，服务器靠它识别身份和扣额度。留空是因为真实密钥走 `.env` 注入，不提交到 Git |
| `model` | 模型名 | 指定用哪个模型。`Qwen/Qwen2.5-7B-Instruct` 是 Qwen2.5 的 7B 指令版，有免费额度。换成 `deepseek-ai/DeepSeek-V3` 等就换模型 |
| `temperature` | 随机性 | 0~1 之间。越接近 0 回答越确定、死板；越接近 1 越发散、有创意。日常问答 0.7 合适，写代码建议降到 0.2 |
| `max_tokens` | 回复长度上限 | 单次回答最多生成多少个 token（约等于「字」）。超过会被强制截断，回答戛然而止。2048 够用，长文场景调大 |
| `timeout` | 请求超时 | 单次请求最长等多少秒，超时就抛异常。这是触发重试的前置条件——免费 API 高峰期经常慢 |

## 二、retry —— 重试策略配置

免费 API 几乎必然遇到限流（HTTP 429）和超时，这组参数控制「失败了怎么办」。

| 字段 | 作用 | 说明 |
|---|---|---|
| `max_attempts` | 最大尝试次数 | 含首次请求。5 = 第一次失败后再重试 4 次，全失败才放弃 |
| `initial_wait` | 首次等待秒数 | 失败后第一次重试前等 1 秒，给服务器喘息时间 |
| `backoff_multiplier` | 退避倍数 | 指数退避的基数。等待序列是：1s → 2s → 4s → 8s → 16s（每次乘 2） |
| `max_wait` | 单次等待上限 | 防止退避时间失控（比如乘到 64s 还在等）。封顶 30 秒，到顶不再增长 |

## 三、storage —— 结果存储配置

控制 AI 生成的结果怎么落地成文件。

| 字段 | 作用 | 说明 |
|---|---|---|
| `output_dir` | 输出目录 | 结果文件保存位置，相对项目根目录。已在 `.gitignore` 中忽略 |
| `default_format` | 默认格式 | `json` / `md` / `txt`。JSON 会连 prompt、模型名、时间戳一起存，方便回溯 |
| `filename_prefix` | 文件名前缀 | 生成文件名如 `result_20260915_194530.json`，避免重名覆盖 |
| `save_metadata` | 是否存元信息 | true 时文件里附带「你问了什么、用的哪个模型、几点调的」，事后排查问题很有用 |

## 四、logging —— 日志配置

程序运行过程的记录。

| 字段 | 作用 | 说明 |
|---|---|---|
| `level` | 日志级别 | `DEBUG`（最详细）→ `INFO`（常规）→ `WARNING` → `ERROR`。开发期用 `INFO`，排查问题临时改 `DEBUG` |
| `log_file` | 日志文件路径 | 运行记录写进这个文件。API 调用失败、重试了几次、最终成功与否，都能在这里查到 |

## 五、memory —— 会话记忆配置

多轮对话的持久化与上下文工程（M1 新增，对应 `session_store.py` + `context_engine.py`；M1.5 起支持双后端）。

| 字段 | 作用 | 说明 |
|---|---|---|
| `backend` | 存储后端 | `sqlite`（默认，零配置本地文件）或 `postgres`（服务端数据库，适合多用户/服务化部署）。切换后所有会话功能不变——上层代码只依赖接口，不感知底层 |
| `db_path` | 会话数据库路径 | 仅 sqlite 后端使用。默认 `output/sessions.db`，已随 `output/` 被 gitignore |
| `postgres` | PG 连接参数 | 仅 postgres 后端使用：`host` / `port` / `user` / `database`（库不存在会自动创建）。**密码不在这里**，走 `.env` 的 `MODELCLAW_PG_PASSWORD` |
| `max_context_tokens` | 上下文硬上限 | 每次请求前用 `trim_messages` 把历史裁剪到这个 token 数以内（tiktoken cl100k_base 近似计数）。超过模型上下文窗口会报 400，这个值是安全阀 |
| `summary_trigger_tokens` | 摘要触发阈值 | 历史超过这个 token 数时，自动把最老的几轮交给 LLM 压成「滚动摘要」存进库，原始消息删除。调小可观察摘要触发，调大则少用摘要省 token |
| `keep_recent_turns` | 摘要保留轮数 | 触发摘要时，最近 N 轮对话保留原文不压缩（保证近期上下文精确），更早的才进摘要 |

## 六、agent —— Agent 工具调用配置

Agent 模式（`modelclaw agent`）的行为边界，对应 `agent.py` + `tools.py`。

| 字段 | 作用 | 说明 |
|---|---|---|
| `max_iterations` | 迭代上限 | Agent 是「模型驱动的循环」，这是唯一的止损线：超过这个轮数还没给出最终答案就报错退出，防模型陷入死循环烧光额度。8 轮对绝大多数任务够用 |
| `enabled_tools` | 启用的工具集 | 数组里是 `tools.py` 注册表中的工具名。删掉某项即禁用（比如不想开放联网就去掉 `web_search`）；在 `tools.py` 里新增工具后，要在这里登记才会生效 |

## 七、rag —— 知识库问答配置

RAG（检索增强生成）管线，对应 `embeddings.py` / `rag_store.py` / `rag.py`。

| 字段 | 作用 | 说明 |
|---|---|---|
| `backend` | 向量库后端 | `pgvector`（默认，存在 PG 的 modelclaw 库）或 `sqlite-vec`（零配置本地文件）。切换后 `docs`/`ask` 行为一致 |
| `db_path` | sqlite-vec 库文件 | 仅 sqlite-vec 后端使用。向量是「派生数据」，删了重跑 `ingest` 即可重建 |
| `embedding_model` | 嵌入模型 | 默认 `Qwen/Qwen3-Embedding-0.6B`（1024 维）。**维度不用手配**——建库时从模型输出自动学习，换模型后若与已有库维度不符会提示 `--clear` 重建 |
| `embedding_source` | 模型来源 | `dashscope`（云端 API，零本地算力，默认）/ `modelscope`（本地缓存）/ `huggingface`（本地镜像） |
| `dashscope_url` | 云端嵌入端点 | 仅 dashscope 源使用；密钥走 `.env` 的 `DASHSCOPE_API_KEY` |
| `query_instruction` | 查询指令前缀 | 部分嵌入模型（如 Qwen3 系列）支持给查询加指令提升检索效果；实测提升不大则留空 |
| `chunk_size` / `chunk_overlap` | 切分参数 | 块大小与重叠。块太大检索不精确，太小语义不完整；500/50 是通用起点 |
| `top_k` | 回答用的块数 | 重排后取前 k 块拼进提示词。不是越大越好——无关块会稀释答案 |
| `reranker_model` | 重排模型 | cross-encoder，把问题和候选块拼起来精读打分。检索质量的关键保险丝 |

## 八、配置在代码中的流转

```
load_config() 读取
    ↓
api_client.py      用 api + retry 组 → 发请求、失败重试
storage.py         用 storage 组    → 存结果文件
logger.py          用 logging 组    → 写日志
session_store.py   用 memory 组     → 会话存取（SQLite / PostgreSQL 双后端）
context_engine.py  用 memory 组     → 裁剪 + 滚动摘要
agent.py + tools.py 用 agent 组     → 工具集 + 迭代上限
embeddings/rag_store/rag.py 用 rag 组 → 嵌入 / 切分 / 检索 / 重排
```

七个配置组正好对应各模块，职责清晰。每个字段将来都可以在不改代码的情况下调整——这就是配置化的意义：调参是改文件的事，不是改逻辑的事。

## 九、使用备忘

- JSON 不支持注释，想给字段写备忘，可用 `"_说明_xxx": "内容"` 形式，代码读取时忽略
- JSON 不允许尾随逗号，最后一个字段后面不能多逗号
- `api_key` 永远留空，真实值放在 `.env`（已在 `.gitignore` 中）
