"""Logging utilities for GEMMA-Next.

This module provides loguru-based logging configuration and GEMMA-compatible
log file output.
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

import gemma_next


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
) -> None:
    """Configure loguru for GEMMA-Next.

    Sets up console logging with INFO level (or DEBUG if verbose), and
    optional file logging with JSON serialization.

    Args:
        verbose: If True, set console logging to DEBUG level.
        log_file: Optional path to log file. If provided, DEBUG-level
            logs are written with JSON serialization.
    """
    # Remove default handler
    logger.remove()

    # Console handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | {message}",
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
    output_config: "gemma_next.core.config.OutputConfig",
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
        ## GEMMA-Next Version = 0.1.0
        ## Date = 2024-01-31T10:30:00
        ##
        ## Command Line Input = gemma-next gk -bfile data
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
        f.write(f"## GEMMA-Next Version = {gemma_next.__version__}\n")
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
