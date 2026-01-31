"""GEMMA-Next command-line interface.

This module provides a Typer-based CLI matching GEMMA's command-line interface,
including -bfile, -o, -outdir flags for data loading and output configuration.
"""

import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

import gemma_next
from gemma_next.core import OutputConfig
from gemma_next.io import load_plink_binary
from gemma_next.utils import setup_logging, write_gemma_log

# Create Typer app
app = typer.Typer(
    name="gemma-next",
    help="GEMMA-Next: Mixed Model Association for genome-wide association studies.",
    add_completion=False,
)

# Store global options set by callback
_global_config: Optional[OutputConfig] = None


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"GEMMA-Next version {gemma_next.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    outdir: Annotated[
        Path,
        typer.Option("-outdir", help="Output directory"),
    ] = Path("output"),
    output: Annotated[
        str,
        typer.Option("-o", help="Output file prefix"),
    ] = "result",
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Verbose output"),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """GEMMA-Next: Mixed Model Association.

    A modern Python reimplementation of GEMMA targeting exact numerical
    compatibility with the original C++ implementation.
    """
    global _global_config
    _global_config = OutputConfig(outdir=outdir, prefix=output, verbose=verbose)
    setup_logging(verbose=verbose)


@app.command("gk")
def gk_command(
    bfile: Annotated[
        Path,
        typer.Option("-bfile", help="PLINK binary file prefix (without .bed/.bim/.fam)"),
    ],
    mode: Annotated[
        int,
        typer.Option("-gk", help="Kinship matrix type (1=centered, 2=standardized)"),
    ] = 1,
) -> None:
    """Compute kinship matrix from genotype data.

    Reads PLINK binary files and computes a genetic relatedness matrix.
    Currently a placeholder - full implementation in Phase 2.
    """
    start_time = time.time()

    # Get global config
    global _global_config
    if _global_config is None:
        _global_config = OutputConfig()

    # Ensure output directory exists
    _global_config.ensure_outdir()

    # Construct command line for logging
    command_line = " ".join(sys.argv)

    # Validate bfile exists
    bed_path = Path(f"{bfile}.bed")
    if not bed_path.exists():
        typer.echo(f"Error: PLINK file not found: {bed_path}", err=True)
        raise typer.Exit(code=1)

    # Load PLINK data
    typer.echo(f"Loading PLINK data from {bfile}...")
    try:
        plink_data = load_plink_binary(bfile)
    except Exception as e:
        typer.echo(f"Error loading PLINK data: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loaded {plink_data.n_samples} samples, {plink_data.n_snps} SNPs")

    # Placeholder for kinship computation
    typer.echo("Kinship computation not yet implemented (Phase 2)")

    # Calculate timing
    end_time = time.time()
    elapsed = end_time - start_time

    # Write log file
    params = {
        "n_samples": plink_data.n_samples,
        "n_snps": plink_data.n_snps,
        "kinship_mode": mode,
    }
    timing = {"total": elapsed}

    log_path = write_gemma_log(_global_config, params, timing, command_line)
    typer.echo(f"Log written to {log_path}")


@app.command("lmm")
def lmm_command(
    bfile: Annotated[
        Path,
        typer.Option("-bfile", help="PLINK binary file prefix (without .bed/.bim/.fam)"),
    ],
    lmm_mode: Annotated[
        int,
        typer.Option("-lmm", help="LMM analysis type (1-4)"),
    ] = 1,
) -> None:
    """Perform linear mixed model association testing.

    Currently a placeholder - full implementation in Phase 3.
    """
    typer.echo("LMM not yet implemented (Phase 3)")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
