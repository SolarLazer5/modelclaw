# modelclaw

一个基于 OpenAI 兼容客户端调用托管大模型推理 API 的轻量 Python 命令行项目。当前接入 **ModelScope** 推理服务，支持**流式输出**、**异常自动重试**、**结果落盘存档**和**运行日志**，所有行为均由 `config.json` 驱动，调参不改代码。

## 功能特性

- **流式对话输出**：实时打印模型回复，并将思考过程（`reasoning_content`，`=== Thinking ===`）与最终答案（`=== Final Answer ===`）分开展示
- **异常重试**：基于 [tenacity](https://github.com/jd/tenacity) 的指数退避重试，覆盖超时、限流（429）、连接错误等可恢复异常；重试次数、等待序列、封顶时间全部可配置（默认 1s→2s→4s→8s→16s，封顶 30s，共 5 次），每次重试前写 WARNING 日志
- **结果存文件**：每次调用的结果自动保存到 `output/` 目录，文件名带时间戳防覆盖（如 `result_20260916_171114.json`）；支持 `json` / `md` / `txt` 三种格式，可附带 prompt、模型名、时间戳、思考过程等元信息，便于回溯排查
- **运行日志**：按配置级别同时输出到控制台和 `output/app.log`，API 请求、重试过程、文件保存路径全程可查
- **配置化**：API 连接、重试策略、存储、日志四组参数集中在 `config.json`，换服务商/换模型/调重试参数只需改配置文件
- **密钥隔离**：真实 API Token 放在 `.env`（已 gitignore），代码与配置文件中不存放任何密钥
- **完整 CLI**：`modelclaw` 命令提供配置向导、单轮/多轮对话、模型列表、连通性测试、历史结果管理等全套入口（见下文「CLI 命令」）

## CLI 命令

Windows 下通过 `modelclaw.bat` 启动（自动使用 `.venv` 中的 Python），也可直接 `python modelclaw.py <命令>`。

```bash
modelclaw configure                              # 交互式配置向导（回车保留当前值）
modelclaw configure --api-key ms-xxx \
    --base-url https://api-inference.modelscope.cn/v1 \
    --model deepseek-ai/DeepSeek-V4.1-Flash      # 非交互式，一步到位

modelclaw chat "用一句话介绍你自己"              # 单轮提问，流式输出并保存结果
modelclaw chat                                    # 进入多轮对话 REPL
modelclaw chat "写代码" --temperature 0.2 --no-save   # 临时覆盖参数 / 不保存

modelclaw config      # 查看当前生效配置（API 密钥打码显示）
modelclaw models      # 列出当前 API 可用的模型
modelclaw ping        # 连通性 + 认证测试，报告延迟
modelclaw history     # 列出 output/ 下所有已保存结果（别名 ls）
modelclaw show 171114 # 查看某次结果，支持时间戳片段模糊匹配
modelclaw clean -y    # 清理结果文件；--logs 连日志一起删；--all 清空 output/
```

多轮对话 REPL 内可用命令：`/save`（保存上一轮结果）、`/clear`（清空上下文）、`/help`、`/exit`。

| 命令 | 说明 |
|---|---|
| `configure` | 设置 `base_url` / `model`（写入 `config.json`）和 `api_key`（写入 `.env`），交互式或用参数非交互 |
| `chat`（别名 `send`） | 发送消息：带消息=单轮；不带=多轮对话。支持 `--system` / `--model` / `--temperature` / `--no-save` |
| `config` | 打印当前生效配置，密钥打码 |
| `models` | 调用 API 列出可用模型 ID |
| `ping` | GET `/models` 测连通性与延迟，验证密钥有效性 |
| `history`（别名 `ls`） | 按时间列出已保存的结果文件及大小 |
| `show` | 查看结果文件内容，接受完整文件名或时间戳片段 |
| `clean` | 删除结果文件，`--logs` 含日志、`--all` 清空目录、`-y` 跳过确认 |

## 项目架构

```
main.py            入口：串起整个流程
    │
    ├─ load_dotenv()        从 .env 注入 MODELSCOPE_API_KEY
    ├─ load_config()        读取 config.json（四个配置组）
    ├─ setup_logging()      logger.py   ← 使用 logging 组：日志级别 + 日志文件
    ├─ create_client()      api_client.py ← 使用 api 组：base_url / api_key / timeout
    ├─ ask()                api_client.py ← 使用 api + retry 组：流式请求 + 指数退避重试
    └─ save_result()        storage.py  ← 使用 storage 组：输出目录 / 格式 / 文件名前缀 / 元信息
```

### 模块说明

| 模块 | 职责 | 对应配置组 |
|---|---|---|
| `main.py` | 程序入口：加载环境变量与配置，初始化日志，发起请求，保存结果，异常时记录 ERROR 并以非零码退出 | 全部 |
| `api_client.py` | 构造 OpenAI 兼容客户端；发送流式 chat 请求并实时打印；用 tenacity 装饰器实现失败重试（`APIError` / `APITimeoutError` / `RateLimitError` / `APIConnectionError`），返回完整的 reasoning + answer 文本 | `api` + `retry` |
| `storage.py` | 自动创建输出目录，按 `{前缀}_YYYYMMDD_HHMMSS.{格式}` 命名保存结果；JSON 格式包含元信息，`save_metadata=false` 时只存答案 | `storage` |
| `logger.py` | 一个 `setup_logging()` 函数：按配置初始化 root logger，同时写文件（UTF-8）和控制台 | `logging` |

### 配置流转

```
load_config() 读取 config.json
    ↓
api_client.py   用 api + retry 组 → 发请求、失败重试
storage.py      用 storage 组     → 存结果文件
logger.py       用 logging 组     → 写日志
```

四个配置组与三个模块一一对应，职责清晰。字段的详细说明见 [doc/config字段说明.md](doc/config字段说明.md)。

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

> 注意：`requirements.txt` 为 UTF-16 编码，编辑时请保留原编码或统一转为 UTF-8。

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
├── modelclaw.py           # CLI 入口：configure / chat / config / models / ping / history / show / clean
├── modelclaw.bat          # Windows 启动器（自动使用 .venv 的 Python）
├── main.py                # 简单入口示例：一次完整调用流程
├── api_client.py          # API 请求 + 重试
├── storage.py             # 结果存文件
├── logger.py              # 日志初始化
├── config.json            # 四组运行配置
├── .env                   # API 密钥（gitignore，不提交）
├── requirements.txt       # 依赖清单（UTF-16 编码）
├── doc/
│   └── config字段说明.md   # 配置字段详细文档
└── output/                # 运行产物：结果文件 + app.log（gitignore）
```

## 安全说明

- **API 密钥只放 `.env`**，`config.json` 的 `api_key` 字段保持为空；`.env` 与 `output/` 均已在 `.gitignore` 中
- 项目早期版本曾将 ModelScope Token 硬编码提交进 git 历史，该 Token 已视为泄露，**如仍在使用请先到 ModelScope 控制台轮换**

## 技术栈

- Python 3.12
- `openai` — OpenAI 兼容客户端（ModelScope / SiliconFlow 等服务通用）
- `tenacity` — 重试与指数退避
- `python-dotenv` — `.env` 环境变量注入

## 后续规划

- 支持指定保存格式（`--format md`）
- 多轮对话历史持久化与恢复
- 引入 pytest 测试套件
