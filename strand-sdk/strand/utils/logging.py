"""Centralized logging helpers."""

from __future__ import annotations

import logging
from typing import Final

_LOGGER_NAME: Final = "strand"


def get_logger(component: str | None = None) -> logging.Logger:
    name = f"{_LOGGER_NAME}.{component}" if component else _LOGGER_NAME
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
