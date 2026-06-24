# -*- coding: utf-8 -*-
"""
config/logging_config.py
Centralized logging setup. Call setup_logging() once at app startup.
"""

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure root logger with console + optional file handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("traffic_eye")

    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt,
                        handlers=handlers)
    # Quiet noisy third-party loggers
    for noisy in ("ultralytics", "PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger("traffic_eye")


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced child logger."""
    return logging.getLogger(f"traffic_eye.{name}")
