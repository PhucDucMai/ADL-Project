"""Logging utilities for the fighting detection system."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, only console logging is used.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
