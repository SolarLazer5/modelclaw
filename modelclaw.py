"""modelclaw CLI: command-line interface for configuration, chat, and result management.

Built with typer + rich.

Usage:
    modelclaw configure              # interactively set api_key / base_url / model
    modelclaw chat "你好"            # one-shot message
    modelclaw chat                   # interactive multi-turn session
    modelclaw config                 # show current effective config
    modelclaw models                 # list available models
    modelclaw ping                   # connectivity test
    modelclaw history                # list saved results
    modelclaw show <file>            # view a saved result
    modelclaw clean                  # remove saved results
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from api_client import ask, create_client
from logger import setup_logging
from main import load_config
from storage import save_result

__version__ = "0.3.0"

CONFIG_PATH = "config.json"
ENV_PATH = ".env"
ENV_KEY_NAME = "MODELSCOPE_API_KEY"

logger = logging.getLogger("modelclaw")
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    help="modelclaw — 配置化的大模型 CLI 客户端",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def mask_key(key: str) -> str:
    if not key:
        return "(not set)"
    if len(key) <= 10:
        return key[:2] + "***"
    return f"{key[:6]}...{key[-4:]}"


def apply_overrides(cfg: dict, model: Optional[str], temperature: Optional[float]) -> dict:
    """Return a config copy with CLI --model / --temperature overrides applied."""
    cfg = json.loads(json.dumps(cfg))  # cheap deep copy
    if model:
        cfg["api"]["model"] = model
    if temperature is not None:
        cfg["api"]["temperature"] = temperature
    return cfg


def find_result_file(cfg: dict, name: str) -> Path:
    """Resolve a result file by full path, file name, or timestamp fragment."""
    candidates = [Path(name)]
    output_dir = Path(cfg.get("storage", {}).get("output_dir", "output"))
    candidates.append(output_dir / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # fuzzy match: timestamp fragment, e.g. `show 171114`
    matches = sorted(output_dir.glob(f"*{name}*")) if output_dir.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"找不到结果文件: {name}"
        + (f"（有 {len(matches)} 个模糊匹配，请写完整文件名）" if matches else "")
    )


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"modelclaw {__version__}")
        raise typer.Exit()


@app.callback()
def _app_callback(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="显示版本号")
    ] = False,
) -> None:
    """modelclaw — 配置化的大模型 CLI 客户端"""


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

@app.command()
def configure(
    api_key: Annotated[Optional[str], typer.Option("--api-key", help="API 密钥（写入 .env，不进入 git）")] = None,
    base_url: Annotated[Optional[str], typer.Option("--base-url", help="API 服务器地址（写入 config.json）")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="模型名（写入 config.json）")] = None,
) -> None:
    """设置 api_key / base_url / model（交互式向导，或用参数一步到位）"""
    cfg = load_config(CONFIG_PATH)
    api_cfg = cfg.setdefault("api", {})

    if not (api_key and base_url and model):
        console.print(Panel.fit("modelclaw 配置向导（直接回车保留当前值）", border_style="cyan"))

    if not base_url:
        base_url = Prompt.ask("Base URL", default=api_cfg.get("base_url", ""))
    if not model:
        model = Prompt.ask("Model", default=api_cfg.get("model", ""))
    if not api_key:
        existing = os.environ.get(ENV_KEY_NAME, "")
        console.print(f"当前 API Key: [dim]{mask_key(existing)}[/dim]")
        entered = Prompt.ask("API Key（密码式输入，回车保持不变）", password=True, default="")
        api_key = entered.strip() or existing

    api_cfg["base_url"] = base_url
    api_cfg["model"] = model
    Path(CONFIG_PATH).write_text(
        json.dumps(cfg, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )
    if api_key:
        set_key(ENV_PATH, ENV_KEY_NAME, api_key)

    console.print("\n[green]√ 配置已保存[/green]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("base_url", base_url)
    table.add_row("model", model)
    table.add_row("api_key", f"{mask_key(api_key)}  [dim](写入 {ENV_PATH})[/dim]")
    console.print(table)


@app.command(name="config")
def show_config() -> None:
    """查看当前生效配置（API 密钥打码显示）"""
    cfg = load_config(CONFIG_PATH)
    display = json.loads(json.dumps(cfg))
    display["api"]["api_key"] = mask_key(os.environ.get(ENV_KEY_NAME, ""))
    console.print(Syntax(json.dumps(display, ensure_ascii=False, indent=4), "json", theme="ansi_dark"))
    console.print(f"[dim]api_key 来源: {ENV_PATH} 的 {ENV_KEY_NAME}[/dim]")


@app.command()
def chat(
    message: Annotated[Optional[str], typer.Argument(help="要发送的消息；不提供则进入多轮对话")] = None,
    system: Annotated[Optional[str], typer.Option("--system", "-s", help="system prompt")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="临时覆盖模型名")] = None,
    temperature: Annotated[Optional[float], typer.Option("--temperature", "-t", help="临时覆盖 temperature")] = None,
    no_save: Annotated[bool, typer.Option("--no-save", help="不保存结果文件")] = False,
) -> None:
    """发送消息；不带消息则进入多轮对话 REPL"""
    cfg = apply_overrides(load_config(CONFIG_PATH), model, temperature)
    setup_logging(cfg)
    client = create_client(cfg)

    if message:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        try:
            result = ask(client, cfg, messages)
        except Exception as exc:
            logger.error("Request failed after all retries: %s", exc)
            err_console.print(f"\n[red]× 请求失败（已重试至上限）: {exc}[/red]")
            raise typer.Exit(1)
        if not no_save:
            path = save_result(cfg, message, result)
            console.print(f"\n[dim]Result saved to: {path}[/dim]")
        return

    _chat_repl(client, cfg, system)


def _chat_repl(client, cfg: dict, system: Optional[str]) -> None:
    console.print(Panel.fit(
        f"多轮对话 · 模型: [bold]{cfg['api']['model']}[/bold]\n"
        "[dim]/save 保存上一轮结果 · /clear 清空上下文 · /help 帮助 · /exit 退出[/dim]",
        border_style="cyan",
    ))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    last = None  # (prompt, result) of the latest exchange

    while True:
        try:
            user_input = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            return
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            console.print("[dim]bye.[/dim]")
            return
        if user_input == "/clear":
            messages = [m for m in messages if m["role"] == "system"]
            last = None
            console.print("[dim](上下文已清空)[/dim]")
            continue
        if user_input == "/save":
            if last:
                path = save_result(cfg, last[0], last[1])
                console.print(f"[dim](已保存到 {path})[/dim]")
            else:
                console.print("[dim](还没有可保存的对话)[/dim]")
            continue
        if user_input == "/help":
            console.print("[dim]/save 保存上一轮结果 · /clear 清空上下文 · /exit 退出[/dim]")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            result = ask(client, cfg, messages)
        except Exception as exc:
            logger.error("Request failed after all retries: %s", exc)
            err_console.print(f"[red]× 请求失败（已重试至上限）: {exc}[/red]")
            messages.pop()  # don't keep the failed user turn
            continue
        messages.append({"role": "assistant", "content": result["answer"]})
        last = (user_input, result)
        print()


# chat 的别名：modelclaw send
app.command("send", help="chat 的别名")(chat)


@app.command()
def models() -> None:
    """列出当前 API 可用的模型"""
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg)
    client = create_client(cfg)
    try:
        with console.status(f"正在获取模型列表 ({cfg['api']['base_url']}) ..."):
            model_list = client.models.list()
    except Exception as exc:
        logger.error("Failed to list models: %s", exc)
        err_console.print(f"[red]× 获取模型列表失败: {exc}[/red]")
        raise typer.Exit(1)
    for m in model_list.data:
        console.print(f"  {m.id}")


@app.command()
def ping() -> None:
    """API 连通性 + 认证测试"""
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg)
    base_url = cfg["api"]["base_url"]
    try:
        client = create_client(cfg)
        with console.status(f"Pinging {base_url} ..."):
            start = time.perf_counter()
            client.models.list()
            latency = (time.perf_counter() - start) * 1000
    except Exception as exc:
        err_console.print(f"[red]× FAILED: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]√ OK ({latency:.0f} ms)[/green] — 认证有效，服务可用")


@app.command()
def history() -> None:
    """列出 output/ 下所有已保存的结果文件"""
    cfg = load_config(CONFIG_PATH)
    storage_cfg = cfg.get("storage", {})
    output_dir = Path(storage_cfg.get("output_dir", "output"))
    prefix = storage_cfg.get("filename_prefix", "result")

    files = sorted(output_dir.glob(f"{prefix}_*.*"), key=lambda p: p.stat().st_mtime)
    if not files:
        console.print(f"[dim]( {output_dir}/ 下还没有保存的结果 )[/dim]")
        return

    table = Table(title=f"已保存的结果（{output_dir}/）", header_style="bold cyan")
    table.add_column("文件", style="bold")
    table.add_column("大小", justify="right")
    table.add_column("修改时间")
    for f in files:
        stat = f.stat()
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        table.add_row(f.name, f"{stat.st_size}B", mtime)
    console.print(table)


# history 的别名：modelclaw ls
app.command("ls", help="history 的别名")(history)


@app.command()
def show(
    file: Annotated[str, typer.Argument(help="文件名或时间戳片段，如 result_20260916_171114.json 或 171114")],
) -> None:
    """查看某个保存的结果（支持时间戳片段模糊匹配）"""
    cfg = load_config(CONFIG_PATH)
    try:
        path = find_result_file(cfg, file)
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        meta = "  ".join(
            f"[bold]{k}[/bold]: {data[k]}" for k in ("prompt", "model", "timestamp") if k in data
        )
        console.print(Panel(meta, title=path.name, border_style="cyan"))
        if data.get("reasoning"):
            console.rule("[dim]Thinking[/dim]")
            console.print(f"[dim]{data['reasoning']}[/dim]")
        console.rule("[bold]Final Answer[/bold]")
        console.print(data.get("answer", ""))
    else:
        console.print(text)


@app.command()
def clean(
    logs: Annotated[bool, typer.Option("--logs", help="同时删除日志文件")] = False,
    all_files: Annotated[bool, typer.Option("--all", help="删除 output 下所有文件")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="跳过确认")] = False,
) -> None:
    """清理 output 目录中的结果文件"""
    cfg = load_config(CONFIG_PATH)
    storage_cfg = cfg.get("storage", {})
    output_dir = Path(storage_cfg.get("output_dir", "output"))
    prefix = storage_cfg.get("filename_prefix", "result")

    targets = list(output_dir.glob(f"{prefix}_*.*")) if output_dir.is_dir() else []
    if logs or all_files:
        log_file = Path(cfg.get("logging", {}).get("log_file", "output/app.log"))
        if log_file.is_file():
            targets.append(log_file)
    if all_files:
        targets = [p for p in output_dir.glob("*") if p.is_file()] if output_dir.is_dir() else []

    if not targets:
        console.print("[dim](没有需要清理的文件)[/dim]")
        return
    if not yes:
        for t in targets:
            console.print(f"  {t}")
        if not Confirm.ask(f"确认删除以上 {len(targets)} 个文件？", default=False):
            console.print("[dim]已取消[/dim]")
            return
    for t in targets:
        t.unlink()
    console.print(f"[green]√ 已删除 {len(targets)} 个文件[/green]")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    # Windows GBK console can't encode emoji etc. — replace instead of crashing mid-stream
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        app(prog_name="modelclaw")
    except BrokenPipeError:
        # stdout closed early by the consumer (e.g. `modelclaw models | head`), exit quietly
        os._exit(0)
    except OSError as exc:
        if exc.errno in (22, 32):  # same situation on Windows legacy console renderer
            os._exit(0)
        raise


if __name__ == "__main__":
    main()
