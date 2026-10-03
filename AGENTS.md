# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

`modelclaw` is a small, early-stage Python learning/experiment project for calling hosted LLM inference APIs through the OpenAI-compatible client. It loads `config.json`, calls the ModelScope API with tenacity-based retry, and saves results to `output/`. There is no package structure, no build system, and no test suite yet.

- **Remote**: `git@github.com:SolarLazer5/modelclaw.git` (branch `main`)
- **Platform**: Windows, developed with VS Code
- **Python**: 3.12.5, with a local virtual environment in `.venv/`

### File inventory

| File | Purpose |
|------|---------|
| `modelclaw.py` | CLI entry point (**typer + Rich**): `configure` (set api_key/base_url/model, password-style key prompt), `chat`/`send` (one-shot or multi-turn REPL with `/save` `/clear` `/session` `/exit`, `--session`/`--resume` for persisted conversations), `agent` (tool-calling agent with `/tools`, shares the session system with chat), `sessions` (list/delete chat sessions), `config` (show, key masked), `models`, `ping`, `history`/`ls`, `show` (fuzzy timestamp match), `clean` (`--sessions` also drops the session DB). Rich renders tables/panels/syntax highlighting; `--version` is an eager option; stdout/stderr/stdin are reconfigured with `errors="replace"` for Windows GBK consoles. |
| `modelclaw.bat` | Windows launcher that runs `modelclaw.py` with the `.venv` interpreter, so `modelclaw configure` works without activating the venv. |
| `main.py` | Simple entry example: loads `.env` + `config.json`, sets up logging, calls the model via `api_client.ask()`, saves the result via `storage.save_result()`. |
| `api_client.py` | Builds the OpenAI client from the `api` config group (key from `MODELSCOPE_API_KEY` env var). `ask()` sends streaming chat requests; `complete()` is the non-streaming variant (summarization); `ask_with_tools()` is the non-streaming function-calling variant (agent loop). All share one tenacity retry factory driven by the `retry` config group (max_attempts=5, exponential backoff 1s→2s→4s→8s→16s capped at 30s). |
| `storage.py` | Saves results to `output/{prefix}_YYYYMMDD_HHMMSS.{json|md|txt}` per the `storage` config group, optionally with metadata (prompt/model/timestamp/reasoning). |
| `logger.py` | `setup_logging()`: configures root logger from the `logging` config group (level + `output/app.log`, also echoed to console). |
| `session_store.py` | Pluggable session persistence (Repository pattern): `SessionStore` ABC (12-method contract), `SqliteSessionStore` (default, one local file), and `create_session_store(cfg)` factory driven by `memory.backend`. Connections are opened per-operation and explicitly closed (`with sqlite3.connect()` alone never closes, which locks the file on Windows). |
| `postgres_store.py` | `PostgresSessionStore`: psycopg 3 backend selected via `memory.backend = "postgres"`. Auto-creates the database (via the maintenance db — PG has no CREATE DATABASE IF NOT EXISTS) and tables. Password comes only from `MODELCLAW_PG_PASSWORD` in `.env`. |
| `agent.py` | Hand-rolled agent loop (native function calling): `run_agent()` feeds `build_context()` output + tool protocol messages to `ask_with_tools()`, executes `tool_calls`, appends tool results, loops until a final answer or `agent.max_iterations`. Only user/final-answer turns are persisted; tool traces render live via Rich and go to the log. |
| `tools.py` | Tool registry: each tool = OpenAI JSON Schema + executor function, gated by `agent.enabled_tools`. Ships `calculator` (eval with empty `__builtins__` + whitelist), `get_current_time`, `read_local_file` (path sandboxed to the project root), `web_search` (ddgs, fails soft). `run_tool()` never raises — errors return as text so the model can self-correct. |
| `context_engine.py` | Context engineering before every model call: tiktoken `cl100k_base` token counting (ModelScope ids are unknown to tiktoken's registry), rolling LLM summarization of old turns when over `memory.summary_trigger_tokens` (summary persisted in DB, summarized raw messages deleted), then `langchain_core.trim_messages` hard cap at `memory.max_context_tokens`. The latest user message is always kept as a safety net. |
| `config.json` | Runtime configuration, now actually loaded by the code. Points to ModelScope (`https://api-inference.modelscope.cn/v1`, `deepseek-ai/DeepSeek-V4.1-Flash`); `api_key` stays empty — the real key is injected from `.env`. Groups: `api` / `retry` / `storage` / `logging` / `memory` (incl. `backend` switch + `postgres` connection). See `doc/config字段说明.md` for the field reference. |
| `README.md` | Project introduction, architecture, features, config reference, and quick-start guide (Chinese). |
| `doc/config字段说明.md` | Chinese field-by-field documentation of `config.json` and the module layout (api_client / storage / logger). |
| `requirements.txt` | Pinned dependencies. **Note: this file is UTF-16 encoded**, so plain UTF-8 readers may show garbled text. Key direct dependencies: `openai==3.14.0`, `typer==0.27.2`, `rich==15.0.0`, `langchain-core` (trim_messages only), `tiktoken`, `psycopg[binary]` (PostgreSQL backend), `ddgs` (web_search tool), `requests`, `python-dotenv`, `PyYAML`, `tenacity`, `pydantic`. |
| `.env` | Holds `MODELSCOPE_API_KEY` (the real ModelScope token); gitignored. |
| `.gitignore` | Excludes `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `output/`, `.vscode/`. |

## Setup and Run

```bash
# Create/activate the virtual environment (Windows, Git Bash)
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

There is no build step, no lint configuration, and no CI pipeline.

## Testing

There is **no test framework or test directory** in this project. If you add logic beyond the current modules, prefer adding `pytest` tests in a `tests/` directory; run them with `python -m pytest`. Verification today means running `python main.py` and checking the streamed output plus the generated file in `output/`.

## Code Style

- Standard Python 3.12, 4-space indentation (per `.vscode/settings.json`: format on save/paste enabled, tab size 4).
- The CLI (`modelclaw.py`) uses typer command functions and Rich for output; the core modules (`api_client.py`, `storage.py`, `logger.py`, `main.py`) stay simple top-level script style. Keep it simple; do not introduce frameworks or abstractions that the project does not need.
- Comments and documentation are in **English** (the only non-English content is a Chinese greeting string `'你好'` used as a demo prompt). Write new comments/docs in English.

## Security Considerations

- The ModelScope token lives in `.env` as `MODELSCOPE_API_KEY` (loaded via `python-dotenv`). **An earlier version of `main.py` had this token hardcoded and committed to git — the token should be rotated in the ModelScope console.**
- `.env` and `output/` are gitignored — keep secrets and generated artifacts out of version control.
- `config.json`'s `api_key` field must stay empty — never commit a real key there; use `.env`.

## Known Quirks

- `requirements.txt` is UTF-16 encoded — preserve or normalize the encoding deliberately when editing.
- The git index contains a file tracked as `main,py` (comma typo) from an earlier commit; the working tree file is `main.py`.
- Git Bash console may display Chinese stream output garbled (console codepage), but files written to `output/` are correct UTF-8.
