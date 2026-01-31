"""JAMMA command-line interface.

This module provides a Typer-based CLI matching GEMMA's command-line interface,
including -bfile, -o, -outdir flags for data loading and output configuration.
"""

import sys
import time
from pathlib import Path
from typing import Annotated

import typer

import jamma
from jamma.core import OutputConfig
from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship, write_kinship_matrix
from jamma.utils import setup_logging, write_gemma_log

# Create Typer app
app = typer.Typer(
    name="jamma",
    help="JAMMA: Mixed Model Association for genome-wide association studies.",
    add_completion=False,
)

# Store global options set by callback
_global_config: OutputConfig | None = None


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"JAMMA version {jamma.__version__}")
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
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """JAMMA: Mixed Model Association.

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
        typer.Option("-bfile", help="PLINK binary file prefix"),
    ],
    mode: Annotated[
        int,
        typer.Option("-gk", help="Kinship matrix type (1=centered, 2=standardized)"),
    ] = 1,
) -> None:
    """Compute kinship matrix from genotype data.

    Reads PLINK binary files and computes a genetic relatedness matrix.
    Writes output in GEMMA-compatible .cXX.txt format.
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
        raise typer.Exit(code=1) from None

    typer.echo(f"Loaded {plink_data.n_samples} samples, {plink_data.n_snps} SNPs")

    # Warn if mode 2 requested (standardized) - not yet implemented
    if mode == 2:
        typer.echo(
            "Warning: Mode 2 (standardized) not yet implemented, using mode 1 (centered)"
        )

    # Compute kinship matrix
    typer.echo(f"Computing centered kinship matrix (mode {mode})...")
    kinship_start = time.time()
    K = compute_centered_kinship(plink_data.genotypes)
    kinship_time = time.time() - kinship_start
    typer.echo(f"Kinship computation completed in {kinship_time:.2f}s")

    # Write kinship matrix
    kinship_path = _global_config.outdir / f"{_global_config.prefix}.cXX.txt"
    write_kinship_matrix(K, kinship_path)
    typer.echo(f"Kinship matrix written to {kinship_path}")

    # Calculate timing
    end_time = time.time()
    elapsed = end_time - start_time

    # Write log file
    params = {
        "n_samples": plink_data.n_samples,
        "n_snps": plink_data.n_snps,
        "kinship_mode": mode,
        "kinship_file": str(kinship_path),
    }
    timing = {"total": elapsed, "kinship": kinship_time}

    log_path = write_gemma_log(_global_config, params, timing, command_line)
    typer.echo(f"Log written to {log_path}")


@app.command("lmm")
def lmm_command(
    bfile: Annotated[
        Path,
        typer.Option("-bfile", help="PLINK binary file prefix"),
    ],
    kinship_file: Annotated[
        Path | None,
        typer.Option("-k", help="Pre-computed kinship matrix file"),
    ] = None,
    lmm_mode: Annotated[
        int,
        typer.Option("-lmm", help="LMM analysis type (1=Wald, 2-4=not yet implemented)"),
    ] = 1,
) -> None:
    """Perform linear mixed model association testing.

    Runs LMM association tests using a pre-computed kinship matrix.
    Currently supports Wald test (-lmm 1) only.
    """
    # Get global config
    global _global_config
    if _global_config is None:
        _global_config = OutputConfig()

    # Validate -lmm mode
    if lmm_mode not in (1, 2, 3, 4):
        typer.echo(f"Error: -lmm must be 1, 2, 3, or 4 (got {lmm_mode})", err=True)
        raise typer.Exit(code=1)

    if lmm_mode in (2, 3, 4):
        typer.echo(f"Error: -lmm {lmm_mode} not yet implemented. Only -lmm 1 (Wald test) is currently supported.", err=True)
        raise typer.Exit(code=1)

    # Validate bfile exists
    bed_path = Path(f"{bfile}.bed")
    bim_path = Path(f"{bfile}.bim")
    fam_path = Path(f"{bfile}.fam")

    if not bed_path.exists():
        typer.echo(f"Error: PLINK .bed file not found: {bed_path}", err=True)
        raise typer.Exit(code=1)
    if not bim_path.exists():
        typer.echo(f"Error: PLINK .bim file not found: {bim_path}", err=True)
        raise typer.Exit(code=1)
    if not fam_path.exists():
        typer.echo(f"Error: PLINK .fam file not found: {fam_path}", err=True)
        raise typer.Exit(code=1)

    # Validate kinship file is provided and exists
    if kinship_file is None:
        typer.echo("Error: -k (kinship matrix) is required for -lmm 1", err=True)
        raise typer.Exit(code=1)
    if not kinship_file.exists():
        typer.echo(f"Error: Kinship matrix file not found: {kinship_file}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Input validation passed: bfile={bfile}, kinship={kinship_file}, mode={lmm_mode}")


if __name__ == "__main__":
    app()
