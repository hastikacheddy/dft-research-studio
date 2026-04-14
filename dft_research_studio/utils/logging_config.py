"""
utils/logging_config.py
------------------------
Centralised logging configuration.
Call setup_logging() once at the entry point of each process.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(log_dir: str | None = None, run_tag: str = "run") -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(console)

    # File handlers
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    txt_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, f"{run_tag}_{date_str}.log"),
        maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    txt_handler.setLevel(level)
    txt_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(txt_handler)

    json_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, f"{run_tag}_{date_str}.jsonl"),
        maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    json_handler.setLevel(level)
    json_handler.setFormatter(_JSONFormatter())
    root.addHandler(json_handler)

    logging.getLogger(__name__).info(
        "Logging initialised — level=%s", level_name
    )
