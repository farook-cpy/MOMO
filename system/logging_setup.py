"""Logging setup with automatic credential and secret redaction."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gsk_[A-Za-z0-9_-]{12,}"), "gsk_[REDACTED]"),
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), "AIza[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "sk-[REDACTED]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._-]+", re.IGNORECASE), r"\1[REDACTED]"),
    (
        re.compile(
            r"((?:api[_-]?key|x-goog-api-key|authorization|token|password|secret)\s*[:=]\s*['\"]?)([^\s'\",;}]+)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]",
    ),
]


def redact_text(text: str) -> str:
    """Scrub API keys, bearer tokens, and secret assignments from log text."""
    result = text
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class RedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from formatted records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact_text(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact_text(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    console: bool = False,
) -> logging.Logger:
    """Configure rotating file logging (and optional stderr logging) with redaction."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "zia.log"

    logger = logging.getLogger("zia")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Clear existing handlers when reconfiguring
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = RedactingFilter()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redactor)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(redactor)
        logger.addHandler(stream_handler)

    return logger
