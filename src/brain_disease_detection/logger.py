"""Centralised logging configuration.

Call :func:`configure_logging` once at process start-up (the web app and the
training CLI both do). Everywhere else, obtain a module-scoped logger with
:func:`get_logger` so log records carry a meaningful ``name``.
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger with a single stream handler.

    Idempotent: calling it more than once will not attach duplicate handlers.

    Args:
        level: Logging level as a name (``"INFO"``) or numeric value. Falls back
            to the ``LOG_LEVEL`` environment variable, then ``INFO``.
    """
    global _configured

    resolved_level = level if level is not None else os.getenv("LOG_LEVEL", "INFO")

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Replace any existing handlers so log output stays consistent even if a
    # dependency (e.g. TensorFlow) installed its own handler first.
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use.

    Args:
        name: Typically ``__name__`` of the calling module.
    """
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
