"""API client: send chat requests with retry, using the `api` and `retry` config groups."""

import logging
import os

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (APIError, APITimeoutError, RateLimitError, APIConnectionError)


def create_client(cfg: dict) -> OpenAI:
    """Build an OpenAI-compatible client from the `api` section of config.json."""
    api_cfg = cfg["api"]
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "未找到 MODELSCOPE_API_KEY，请在 .env 文件中配置 ModelScope Token"
        )
    return OpenAI(
        base_url=api_cfg["base_url"],
        api_key=api_key,
        timeout=api_cfg.get("timeout", 30),
    )


def ask(client: OpenAI, cfg: dict, messages: list) -> dict:
    """Send a streaming chat request with exponential-backoff retry.

    Prints reasoning and answer chunks as they arrive, and returns the
    collected texts as {"reasoning": str, "answer": str}.
    """
    api_cfg = cfg["api"]
    retry_cfg = cfg.get("retry", {})

    @retry(
        stop=stop_after_attempt(retry_cfg.get("max_attempts", 5)),
        wait=wait_exponential(
            multiplier=retry_cfg.get("initial_wait", 1),
            exp_base=retry_cfg.get("backoff_multiplier", 2),
            max=retry_cfg.get("max_wait", 30),
        ),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _ask_once() -> dict:
        response = client.chat.completions.create(
            model=api_cfg["model"],
            messages=messages,
            temperature=api_cfg.get("temperature", 0.7),
            max_tokens=api_cfg.get("max_tokens", 2048),
            stream=True,
        )

        reasoning_parts = []
        answer_parts = []
        done_thinking = False
        done_thinking_print = False

        for chunk in response:
            if not chunk.choices:
                continue
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            answer_chunk = chunk.choices[0].delta.content
            if reasoning_chunk:
                if not done_thinking_print:
                    print("\n\n === Thinking ===\n")
                    done_thinking_print = True
                print(reasoning_chunk, end="", flush=True)
                reasoning_parts.append(reasoning_chunk)
            elif answer_chunk:
                if not done_thinking:
                    print("\n\n === Final Answer ===\n")
                    done_thinking = True
                print(answer_chunk, end="", flush=True)
                answer_parts.append(answer_chunk)

        print()
        return {"reasoning": "".join(reasoning_parts), "answer": "".join(answer_parts)}

    logger.info("Sending request: model=%s", api_cfg["model"])
    return _ask_once()
