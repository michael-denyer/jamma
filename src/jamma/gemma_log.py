"""GEMMA-compatible .log.txt output for a finished pipeline run."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import jamma

if TYPE_CHECKING:
    from jamma.pipeline_config import PipelineConfig


def write_gemma_log(
    config: "PipelineConfig",
    params: dict,
    timing: dict,
    command_line: str,
) -> Path:
    """Write GEMMA-compatible log file.

    Produces a .log.txt file matching GEMMA's format with ## prefixes
    for section headers.

    Args:
        config: Pipeline configuration specifying output directory and prefix.
        params: Dictionary of parameters to log (e.g., n_samples, n_snps).
        timing: Dictionary of timing information (expects 'total' key in seconds).
        command_line: The command line used to invoke the program.

    Returns:
        Path to the written log file.

    Example output format:
        ##
        ## JAMMA Version = 2.1.0
        ## Date = 2024-01-31T10:30:00
        ##
        ## Command Line Input = jamma -gk 1 -bfile data
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
    config.ensure_outdir()

    log_path = config.log_path

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
