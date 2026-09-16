"""Entry point: load config, call the model with retry, save the result to a file."""

import json
import logging
import sys

from dotenv import load_dotenv

from api_client import ask, create_client
from logger import setup_logging
from storage import save_result

def load_config(path: str = "config.json") -> dict:
    """读取config.json，返回一个字典"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main() -> int:
    load_dotenv() #读取环境变量，存储到os.environ环境变量字典中      
    cfg = load_config()
    setup_logging(cfg)

    logger = logging.getLogger(__name__)
    prompt = "你好"

    client = create_client(cfg)
    try:
        result = ask(client, cfg, prompt)
    except Exception as exc:
        logger.error("Request failed after all retries: %s", exc)
        return 1

    path = save_result(cfg, prompt, result)
    print(f"\nResult saved to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
