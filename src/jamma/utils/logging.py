"""Logging utilities for JAMMA.

This module provides loguru-based logging configuration and GEMMA-compatible
log file output.
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

import jamma


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


def write_gemma_log(
    output_config: "jamma.core.config.OutputConfig",
    params: dict,
    timing: dict,
    command_line: str,
) -> Path:
    """Write GEMMA-compatible log file.

    Produces a .log.txt file matching GEMMA's format with ## prefixes
    for section headers.

    Args:
        output_config: Output configuration specifying directory and prefix.
        params: Dictionary of parameters to log (e.g., n_samples, n_snps).
        timing: Dictionary of timing information (expects 'total' key in seconds).
        command_line: The command line used to invoke the program.

    Returns:
        Path to the written log file.

    Example output format:
        ##
        ## JAMMA Version = 0.1.0
        ## Date = 2024-01-31T10:30:00
        ##
        ## Command Line Input = jamma gk -bfile data
        ##
        ## Summary Statistics:
        ## n_samples = 1940
        ## n_snps = 12226
        ##
        ## Computation Time:
        ## total time = 1.23 seconds
        ##
    """
    # Ensure output directory exists
    output_config.ensure_outdir()

    log_path = output_config.log_path

    with open(log_path, "w") as f:
        # Header
        f.write("##\n")
        f.write(f"## JAMMA Version = {jamma.__version__}\n")
        f.write(f"## Date = {datetime.now().isoformat()}\n")
        f.write("##\n")

        # Command line
        f.write(f"## Command Line Input = {command_line}\n")
        f.write("##\n")

        # Parameters
        f.write("## Summary Statistics:\n")
        for key, value in params.items():
            f.write(f"## {key} = {value}\n")
        f.write("##\n")

        # Timing
        f.write("## Computation Time:\n")
        for key, value in timing.items():
            if isinstance(value, float):
                f.write(f"## {key} time = {value:.2f} seconds\n")
            else:
                f.write(f"## {key} time = {value} seconds\n")
        f.write("##\n")

    return log_path


def log_rss_memory(phase: str, checkpoint: str) -> float:
    """Log current RSS memory usage with phase context.

    Uses loguru's bind() for structured logging so memory readings
    can be filtered/searched by phase and checkpoint.

    Args:
        phase: Workflow phase name (e.g., "eigendecomp", "lmm", "kinship")
        checkpoint: Checkpoint within phase (e.g., "start", "end", "chunk_1")

    Returns:
        Current RSS in GB (for chaining/testing).

    Example:
        >>> log_rss_memory("eigendecomp", "before")
        INFO     | RSS memory: 12.34GB (phase=eigendecomp, checkpoint=before)
    """
    import psutil

    rss_gb = psutil.Process().memory_info().rss / 1e9
    logger.bind(phase=phase, checkpoint=checkpoint).info(
        f"RSS memory: {rss_gb:.2f}GB (phase={phase}, checkpoint={checkpoint})"
    )
    return rss_gb
