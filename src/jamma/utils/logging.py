"""Loguru configuration for JAMMA."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
) -> None:
    """Configure loguru for JAMMA.

    Sets up console logging with INFO level (or DEBUG if verbose), and
    optional file logging with JSON serialization.

    Args:
        verbose: If True, set console logging to DEBUG level.
        log_file: Optional path to log file. If provided, DEBUG-level
            logs are written with JSON serialization.
    """
    # Remove default handler
    logger.remove()

    # Console handler — stdout for Databricks visibility (stderr may be buffered)
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stdout,
        level=level,
        format="{time:HH:mm:ss} | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    # File handler with JSON (DEBUG level)
    if log_file:
        logger.add(
            log_file,
            serialize=True,
            level="DEBUG",
        )
