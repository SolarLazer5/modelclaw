"""Result storage: save model output to a file, using the `storage` config group."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def save_result(cfg: dict, prompt: str, result: dict) -> Path:
    """Save the model result to `storage.output_dir` and return the file path."""
    storage_cfg = cfg.get("storage", {})
    output_dir = storage_cfg.get("output_dir", "output")
    fmt = storage_cfg.get("default_format", "json")
    prefix = storage_cfg.get("filename_prefix", "result")
    save_metadata = storage_cfg.get("save_metadata", True)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"{prefix}_{timestamp}.{fmt}"

    metadata = {
        "prompt": prompt,
        "model": cfg["api"]["model"],
        "timestamp": timestamp,
    }

    if fmt == "json":
        payload = dict(metadata) if save_metadata else {}
        payload["reasoning"] = result["reasoning"]
        payload["answer"] = result["answer"]
        if not save_metadata:
            payload = {"answer": result["answer"]}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif fmt == "md":
        lines = []
        if save_metadata:
            lines += [
                f"# {prefix}_{timestamp}",
                "",
                f"- Prompt: {prompt}",
                f"- Model: {metadata['model']}",
                f"- Time: {timestamp}",
                "",
            ]
        lines.append(result["answer"])
        path.write_text("\n".join(lines), encoding="utf-8")
    else:  # txt
        path.write_text(result["answer"], encoding="utf-8")

    logger.info("Result saved to %s", path)
    return path
