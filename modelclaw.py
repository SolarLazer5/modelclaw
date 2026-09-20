"""modelclaw CLI: command-line interface for configuration, chat, and result management.

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

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv, set_key

from api_client import ask, create_client
from logger import setup_logging
from main import load_config
from storage import save_result

__version__ = "0.2.0"

CONFIG_PATH = "config.json"
ENV_PATH = ".env"
ENV_KEY_NAME = "MODELSCOPE_API_KEY"

logger = logging.getLogger("modelclaw")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def mask_key(key: str) -> str:
    if not key:
        return "(not set)"
    if len(key) <= 10:
        return key[:2] + "***"
    return f"{key[:6]}...{key[-4:]}"


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Return a config copy with CLI --model / --temperature overrides applied."""
    cfg = json.loads(json.dumps(cfg))  # cheap deep copy
    if getattr(args, "model", None):
        cfg["api"]["model"] = args.model
    if getattr(args, "temperature", None) is not None:
        cfg["api"]["temperature"] = args.temperature
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


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_configure(args: argparse.Namespace) -> int:
    """Set api_key (.env), base_url and model (config.json)."""
    cfg = load_config(CONFIG_PATH)
    api_cfg = cfg.setdefault("api", {})

    base_url = args.base_url
    model = args.model
    api_key = args.api_key

    if not (base_url and model and api_key):
        print("modelclaw 配置向导（直接回车保留当前值）\n")
    if not base_url:
        base_url = input(f"Base URL [{api_cfg.get('base_url', '')}]: ").strip() \
            or api_cfg.get("base_url", "")
    if not model:
        model = input(f"Model [{api_cfg.get('model', '')}]: ").strip() \
            or api_cfg.get("model", "")
    if not api_key:
        existing = os.environ.get(ENV_KEY_NAME, "")
        entered = input(f"API Key [{mask_key(existing)}, 回车保持不变]: ").strip()
        api_key = entered or existing

    api_cfg["base_url"] = base_url
    api_cfg["model"] = model
    Path(CONFIG_PATH).write_text(
        json.dumps(cfg, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )
    if api_key:
        set_key(ENV_PATH, ENV_KEY_NAME, api_key)

    print(f"\n配置已保存：")
    print(f"  base_url = {base_url}")
    print(f"  model    = {model}")
    print(f"  api_key  = {mask_key(api_key)}  (写入 {ENV_PATH})")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show the current effective configuration (API key masked)."""
    cfg = load_config(CONFIG_PATH)
    display = json.loads(json.dumps(cfg))
    display["api"]["api_key"] = mask_key(os.environ.get(ENV_KEY_NAME, "")) \
        + f"  (from {ENV_PATH}:{ENV_KEY_NAME})"
    print(json.dumps(display, ensure_ascii=False, indent=4))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """One-shot message or interactive multi-turn session."""
    cfg = apply_overrides(load_config(CONFIG_PATH), args)
    setup_logging(cfg)
    client = create_client(cfg)

    if args.message:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.message})
        try:
            result = ask(client, cfg, messages)
        except Exception as exc:
            logger.error("Request failed after all retries: %s", exc)
            return 1
        if not args.no_save:
            path = save_result(cfg, args.message, result)
            print(f"\nResult saved to: {path}")
        return 0

    return _chat_repl(client, cfg, args)


def _chat_repl(client, cfg: dict, args: argparse.Namespace) -> int:
    print(f"进入多轮对话（模型: {cfg['api']['model']}）")
    print("命令：/save 保存上一轮结果 · /clear 清空上下文 · /exit 退出\n")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    last = None  # (prompt, result) of the latest exchange

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("bye.")
            return 0
        if user_input == "/clear":
            messages = [m for m in messages if m["role"] == "system"]
            last = None
            print("(上下文已清空)")
            continue
        if user_input == "/save":
            if last:
                path = save_result(cfg, last[0], last[1])
                print(f"(已保存到 {path})")
            else:
                print("(还没有可保存的对话)")
            continue
        if user_input == "/help":
            print("命令：/save 保存上一轮结果 · /clear 清空上下文 · /exit 退出")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            result = ask(client, cfg, messages)
        except Exception as exc:
            logger.error("Request failed after all retries: %s", exc)
            messages.pop()  # don't keep the failed user turn
            continue
        messages.append({"role": "assistant", "content": result["answer"]})
        last = (user_input, result)
        print()


def cmd_models(args: argparse.Namespace) -> int:
    """List models available on the configured API."""
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg)
    client = create_client(cfg)
    try:
        models = client.models.list()
    except Exception as exc:
        logger.error("Failed to list models: %s", exc)
        return 1
    for m in models.data:
        print(m.id)
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    """Connectivity test against the configured API."""
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg)
    api_cfg = cfg["api"]
    print(f"Pinging {api_cfg['base_url']} ...")
    try:
        client = create_client(cfg)
        start = time.perf_counter()
        client.models.list()
        latency = (time.perf_counter() - start) * 1000
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK ({latency:.0f} ms) — 认证有效，服务可用")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """List saved result files."""
    cfg = load_config(CONFIG_PATH)
    storage_cfg = cfg.get("storage", {})
    output_dir = Path(storage_cfg.get("output_dir", "output"))
    prefix = storage_cfg.get("filename_prefix", "result")

    files = sorted(output_dir.glob(f"{prefix}_*.*"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"( {output_dir}/ 下还没有保存的结果 )")
        return 0
    print(f"{'文件':<40} {'大小':>8}  修改时间")
    for f in files:
        stat = f.stat()
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"{f.name:<40} {stat.st_size:>7}B  {mtime}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print the content of a saved result file."""
    cfg = load_config(CONFIG_PATH)
    try:
        path = find_result_file(cfg, args.file)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        for key in ("prompt", "model", "timestamp"):
            if key in data:
                print(f"{key}: {data[key]}")
        if data.get("reasoning"):
            print(f"\n === Thinking ===\n\n{data['reasoning']}")
        print(f"\n === Final Answer ===\n\n{data.get('answer', '')}")
    else:
        print(text)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Delete saved result files (and optionally logs)."""
    cfg = load_config(CONFIG_PATH)
    storage_cfg = cfg.get("storage", {})
    output_dir = Path(storage_cfg.get("output_dir", "output"))
    prefix = storage_cfg.get("filename_prefix", "result")

    targets = list(output_dir.glob(f"{prefix}_*.*")) if output_dir.is_dir() else []
    if args.logs or args.all:
        log_file = Path(cfg.get("logging", {}).get("log_file", "output/app.log"))
        if log_file.is_file():
            targets.append(log_file)
    if args.all:
        targets = [p for p in output_dir.glob("*") if p.is_file()] if output_dir.is_dir() else []

    if not targets:
        print("(没有需要清理的文件)")
        return 0
    if not args.yes:
        for t in targets:
            print(f"  {t}")
        confirm = input(f"确认删除以上 {len(targets)} 个文件？[y/N] ").strip().lower()
        if confirm != "y":
            print("已取消")
            return 0
    for t in targets:
        t.unlink()
    print(f"已删除 {len(targets)} 个文件")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelclaw",
        description="modelclaw — 配置化的大模型 CLI 客户端",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("configure", help="设置 api_key / base_url / model")
    p.add_argument("--api-key", help="API 密钥（写入 .env，不进入 git）")
    p.add_argument("--base-url", help="API 服务器地址（写入 config.json）")
    p.add_argument("--model", help="模型名（写入 config.json）")
    p.set_defaults(func=cmd_configure)

    p = sub.add_parser("config", help="查看当前配置（密钥打码）")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("chat", aliases=["send"], help="发送消息；不带消息则进入多轮对话")
    p.add_argument("message", nargs="?", help="要发送的消息")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--model", help="临时覆盖模型名")
    p.add_argument("--temperature", type=float, help="临时覆盖 temperature")
    p.add_argument("--no-save", action="store_true", help="不保存结果文件")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("models", help="列出 API 可用模型")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("ping", help="API 连通性测试")
    p.set_defaults(func=cmd_ping)

    p = sub.add_parser("history", aliases=["ls"], help="列出已保存的结果文件")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("show", help="查看某个保存的结果（支持时间戳片段匹配）")
    p.add_argument("file", help="文件名或时间戳片段，如 result_20260916_171114.json 或 171114")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("clean", help="清理 output 目录中的结果文件")
    p.add_argument("--logs", action="store_true", help="同时删除日志文件")
    p.add_argument("--all", action="store_true", help="删除 output 下所有文件")
    p.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    p.set_defaults(func=cmd_clean)

    return parser


def main(argv=None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
