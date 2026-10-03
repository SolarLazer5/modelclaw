# modelclaw

一个基于 OpenAI 兼容客户端调用托管大模型推理 API 的轻量 Python 命令行项目。

## 功能特性

- **流式对话输出**：实时打印模型回复，思考过程（`reasoning_content`，`=== Thinking ===`）与最终答案（`=== Final Answer ===`）分开展示
- **会话记忆**：多轮对话自动持久化（SQLite / PostgreSQL 双后端），退出不丢；`chat --resume` 恢复历史会话
- **上下文工程**：每次请求前自动执行「滚动摘要压缩 → token 裁剪」流水线
- **Agent 工具调用**：`agent` 命令下模型自主调用工具（计算器 / 本地文件 / 联网搜索 / 当前时间），手写 tool_calls 循环，调用过程全程可视；带 eval 白名单、路径沙箱、迭代上限三层安全边界
- **可插拔存储层**：修改memory.backend配置切换 SQLite ↔ PostgreSQL
- **异常重试**：基于 [tenacity](https://github.com/jd/tenacity) 的指数退避重试，覆盖超时、限流（429）、连接错误等可恢复异常
- **结果存文件**：对话结果自动保存到 `output/` 目录，文件名带时间戳防覆盖（如 `result_20260916_171114.json`）；支持 `json` / `md` / `txt` 三种格式，附带 prompt、模型名、时间戳、思考过程等元信息
- **运行日志**：按配置级别同时输出到控制台和 `output/app.log`，API 请求、重试过程、文件保存路径全程可查
- **凭证隔离**：API Token、数据库密码只放 `.env`，代码与配置文件中不存放任何密钥
- **CLI**：`typer + Rich` 实现，配置向导、单轮/多轮对话、会话管理、模型列表、连通性测试、历史结果管理全套入口

## CLI 命令

Windows 下通过 `modelclaw.bat` 启动（自动使用 `.venv` 中的 Python），也可直接 `python modelclaw.py <命令>`。

```bash
modelclaw configure                              # 交互式配置向导（回车保留当前值）
modelclaw configure --api-key ms-xxx \
    --base-url https://api-inference.modelscope.cn/v1 \
    --model deepseek-ai/DeepSeek-V4.1-Flash      # 非交互式，一步到位

modelclaw chat "用一句话介绍你自己"              # 单轮提问，流式输出并保存结果
modelclaw chat                                    # 进入多轮对话 REPL（会话自动持久化）
modelclaw chat --resume 1757                      # 恢复历史会话（ID 片段模糊匹配），模型记得之前聊过什么
modelclaw chat "写代码" --temperature 0.2 --no-save   # 临时覆盖参数 / 不保存

modelclaw sessions     # 列出所有历史会话（ID / 标题 / 轮数 / 更新时间），--delete <片段> 删除

modelclaw agent "计算 300 的 25% 再加 17"        # Agent：模型自主调用计算器
modelclaw agent "读 config.json 告诉我模型名"      # Agent：模型自主读文件
modelclaw agent                                    # Agent 多轮模式（/tools 查看可用工具）

modelclaw config      # 查看当前生效配置（API 密钥打码显示）
modelclaw models      # 列出当前 API 可用的模型
modelclaw ping        # 连通性 + 认证测试，报告延迟
modelclaw history     # 列出 output/ 下所有已保存结果（别名 ls）
modelclaw show 171114 # 查看某次结果，支持时间戳片段模糊匹配
modelclaw clean -y    # 清理结果文件；--logs 连日志一起删；--sessions 删会话库；--all 清空 output/
```

多轮对话 REPL 内可用命令：`/save`（保存上一轮结果）、`/clear`（清空上下文）、`/session`（当前会话信息）、`/help`、`/exit`。
会话持久化默认用 SQLite（`output/sessions.db`），在 `config.json` 里把 `memory.backend` 改成 `postgres` 即可切换到 PostgreSQL 后端（连接参数在 `memory.postgres`，密码走 `.env` 的 `MODELCLAW_PG_PASSWORD`）——上层功能完全一致，存储层可插拔。

| 命令 | 说明 |
|---|---|
| `configure` | 设置 `base_url` / `model`（写入 `config.json`）和 `api_key`（写入 `.env`），交互式或用参数非交互 |
| `chat`（别名 `send`） | 发送消息：带消息=单轮；不带=多轮对话。支持 `--system` / `--model` / `--temperature` / `--no-save` / `--session` / `--resume` |
| `agent` | Agent 模式：模型自主调用工具完成任务，支持 `--session` / `--resume`；REPL 内 `/tools` 查看工具 |
| `sessions` | 列出历史会话；`--delete <片段>` 删除指定会话 |
| `config` | 打印当前生效配置，密钥打码 |
| `models` | 调用 API 列出可用模型 ID |
| `ping` | GET `/models` 测连通性与延迟，验证密钥有效性 |
| `history`（别名 `ls`） | 按时间列出已保存的结果文件及大小 |
| `show` | 查看结果文件内容，接受完整文件名或时间戳片段 |
| `clean` | 删除结果文件，`--logs` 含日志、`--sessions` 含会话库、`--all` 清空目录、`-y` 跳过确认 |

## 项目架构

以 CLI 多轮对话为例，一次提问的完整链路：

```
modelclaw chat
    │
    ├─ load_dotenv()           从 .env 注入密钥（API Token / PG 密码）
    ├─ load_config()           读取 config.json（五个配置组）
    ├─ setup_logging()         logger.py        ← logging 组
    ├─ create_session_store()  session_store.py ← memory 组：Repository 工厂，按 backend 选 SQLite / PostgreSQL
    │
    │  每轮对话：
    ├─ store.add_message()     用户消息先落库
    ├─ build_context()         context_engine.py ← memory 组：滚动摘要 + token 裁剪，控制上下文成本
    ├─ ask()                   api_client.py    ← api + retry 组：流式请求 + 指数退避重试
    ├─ store.add_message()     助手回复落库（请求失败则回滚刚写入的用户消息）
    └─ save_result()           storage.py       ← storage 组：/save 时结果存文件
```

### 模块说明

| 模块 | 职责 | 对应配置组 |
|---|---|---|
| `modelclaw.py` | CLI 入口（typer + Rich）：12 个命令 + 多轮对话 REPL，只做参数解析与流程编排 | 全部 |
| `api_client.py` | 构造 OpenAI 兼容客户端；`ask()` 流式请求实时打印、`complete()` 非流式（供摘要调用）；共用 tenacity 重试工厂 | `api` + `retry` |
| `session_store.py` | `SessionStore` 抽象基类（12 个方法即接口契约）+ SQLite 实现 + 后端工厂。契约在此，方言在下层 | `memory` |
| `postgres_store.py` | PostgreSQL 后端（psycopg 3）：自动建库建表，密码只从 `.env` 读 | `memory` |
| `context_engine.py` | 上下文工程：tiktoken 计数、滚动摘要压缩、`trim_messages` 裁剪、最新用户消息兜底 | `memory` |
| `agent.py` | Agent 循环（手写 tool_calls 协议）：模型要行动就执行工具回灌结果，给出最终答案则落库返回；迭代上限防死循环 | `agent` + `memory` |
| `tools.py` | 工具注册表：JSON Schema（模型读的说明书）+ 执行函数；eval 白名单 / 路径沙箱 / 超时截断 | `agent` |
| `storage.py` | 结果落盘：按 `{前缀}_YYYYMMDD_HHMMSS.{格式}` 命名，JSON 格式可附元信息 | `storage` |
| `logger.py` | `setup_logging()`：按配置初始化 root logger，同时写文件（UTF-8）和控制台 | `logging` |
| `main.py` | 最简单的入口示例：一次完整调用流程，供学习对照 | 全部 |

### 配置流转

```
load_config() 读取 config.json
    ↓
api_client.py      用 api + retry 组 → 发请求、失败重试
storage.py         用 storage 组     → 存结果文件
logger.py          用 logging 组     → 写日志
session_store.py   用 memory 组      → 会话存取（后端可插拔）
context_engine.py  用 memory 组      → 裁剪 + 滚动摘要
agent.py + tools.py 用 agent 组      → 工具集 + 迭代上限
```

六个配置组与模块一一对应，职责清晰。字段的详细说明见 [doc/config字段说明.md](doc/config字段说明.md)。

## 配置说明（速览）

```jsonc
{
    "api": {
        "base_url": "https://api-inference.modelscope.cn/v1",  // API 地址，换服务商改这里
        "api_key": "",          // 永远留空，真实密钥走 .env
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",            // 模型名
        "temperature": 0.7,     // 随机性：0 稳定 ~ 1 发散
        "max_tokens": 2048,     // 单次回复长度上限
        "timeout": 30           // 单次请求超时秒数（触发重试的前置条件）
    },
    "retry": {
        "max_attempts": 5,        // 最大尝试次数（含首次）
        "initial_wait": 1,        // 首次重试前等待秒数
        "backoff_multiplier": 2,  // 退避倍数 → 1s→2s→4s→8s→16s
        "max_wait": 30            // 单次等待上限
    },
    "storage": {
        "output_dir": "output",   // 结果保存目录（已 gitignore）
        "default_format": "json", // json / md / txt
        "filename_prefix": "result",
        "save_metadata": true     // 是否附带 prompt/模型/时间戳等元信息
    },
    "logging": {
        "level": "INFO",          // DEBUG / INFO / WARNING / ERROR
        "log_file": "output/app.log"
    },
    "memory": {
        "backend": "sqlite",      // sqlite / postgres，存储后端一键切换
        "db_path": "output/sessions.db",           // sqlite 后端：数据库文件
        "postgres": {                               // postgres 后端：连接参数（密码在 .env）
            "host": "127.0.0.1", "port": 5432,
            "user": "postgres", "database": "modelclaw"
        },
        "max_context_tokens": 4000,     // 上下文 token 硬上限（裁剪）
        "summary_trigger_tokens": 3000, // 历史超过此值触发滚动摘要
        "keep_recent_turns": 4          // 摘要时保留最近 N 轮原文
    },
    "agent": {
        "max_iterations": 8,            // Agent 循环迭代上限（防死循环）
        "enabled_tools": ["calculator", "get_current_time", "read_local_file", "web_search"]
    }
}
```

## 快速开始

```bash
# 1. 创建并激活虚拟环境（Windows / Git Bash）
python -m venv .venv
source .venv/Scripts/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥与模型（交互式向导）
modelclaw configure

# 4. 开始对话
modelclaw chat "你好"
```

## 运行效果

终端实时流式输出：

```
 === Thinking ===

The user greeted me in Chinese with "你好" (hello). I should respond warmly...

 === Final Answer ===

很高兴见到你，有什么我可以帮忙的吗？

Result saved to: output/result_20260916_171114.json
```

生成的结果文件 `output/result_20260916_171114.json`：

```json
{
  "prompt": "你好",
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "timestamp": "20260916_171114",
  "reasoning": "The user greeted me in Chinese with \"你好\" (hello). ...",
  "answer": "很高兴见到你，有什么我可以帮忙的吗？"
}
```

## 目录结构

```
modelclaw/
├── modelclaw.py           # CLI 入口：configure / chat(--resume) / sessions / config / models / ping / history / show / clean
├── modelclaw.bat          # Windows 启动器（自动使用 .venv 的 Python）
├── session_store.py       # 会话持久化：SessionStore 抽象基类 + SQLite 实现 + 工厂（Repository 模式）
├── postgres_store.py      # PostgreSQL 后端：memory.backend 切换，自动建库建表，密码走 .env
├── context_engine.py      # 上下文工程：tiktoken 计数 / trim_messages 裁剪 / 滚动摘要压缩
├── agent.py               # Agent 循环：tool_calls → 工具执行 → 结果回灌，直至最终答案
├── tools.py               # 工具注册表：calculator / get_current_time / read_local_file / web_search
├── api_client.py          # API 请求（流式 ask / 非流式 complete / 带工具 ask_with_tools）+ 共用重试工厂
├── storage.py             # 结果存文件
├── logger.py              # 日志初始化
├── main.py                # 简单入口示例：一次完整调用流程
├── config.json            # 六组运行配置
├── .env                   # 密钥（gitignore，不提交）
├── requirements.txt       # 依赖清单（UTF-16 编码）
├── doc/
│   ├── config字段说明.md   # 配置字段详细文档
│   ├── M1学习计划.md       # M1 阶段学习笔记与验收清单
│   └── M2学习计划.md       # M2 阶段（Agent）学习笔记与验收清单
└── output/                # 运行产物：结果文件 + app.log + sessions.db（gitignore）
```

## 技术栈

- Python 3.12
- `typer` + `rich` — CLI 框架与终端渲染
- `openai` — OpenAI 兼容客户端
- `tenacity` — 重试与指数退避
- `langchain-core` + `tiktoken` — 上下文裁剪（trim_messages）与 token 计数
- `psycopg` — PostgreSQL 存储后端（可选）
- `ddgs` — DuckDuckGo 联网搜索（Agent 的 web_search 工具，免 API key）
- `python-dotenv` — `.env` 环境变量注入
