"""Lightweight, consistent logging for scripts and notebooks.

A single ``get_logger`` factory returns a configured :class:`logging.Logger`
that writes to the console and (optionally) to a rotating file under
``outputs/logs``. Re-calling it for the same name is safe — handlers are not
duplicated.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from pathsnap.config import CONFIG

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "pathsnap",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name (usually the module ``__name__``).
    level:
        Logging level, e.g. ``logging.INFO`` or ``logging.DEBUG``.
    log_file:
        Optional filename (not a full path). When given, logs are also written
        to ``outputs/logs/<log_file>`` using a rotating file handler.

    Notes
    -----
    Idempotent: if the logger already has handlers, it is returned unchanged so
    importing from multiple modules does not produce duplicate log lines.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Already configured.
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        CONFIG.paths.logs.mkdir(parents=True, exist_ok=True)
        path: Path = CONFIG.paths.logs / log_file
        file_handler = RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
