"""JAMMA command-line interface.

This module provides a Typer-based CLI matching GEMMA's command-line interface,
including -bfile, -o, -outdir flags for data loading and output configuration.
"""

import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

import jamma
from jamma.core import OutputConfig
from jamma.io import load_plink_binary
from jamma.kinship import (
    compute_centered_kinship,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm import run_lmm_association, write_assoc_results
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
            "Warning: Mode 2 (standardized) not yet implemented, "
            "using mode 1 (centered)"
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
        typer.Option(
            "-lmm", help="LMM analysis type (1=Wald, 2-4=not yet implemented)"
        ),
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
        typer.echo(
            f"Error: -lmm {lmm_mode} not yet implemented. "
            "Only -lmm 1 (Wald test) is currently supported.",
            err=True,
        )
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

    # Ensure output directory exists
    _global_config.ensure_outdir()

    # Record start time
    t_start = time.perf_counter()
    command_line = " ".join(sys.argv)

    # Load PLINK data
    typer.echo(f"Loading PLINK data from {bfile}...")
    try:
        plink_data = load_plink_binary(bfile)
    except Exception as e:
        typer.echo(f"Error loading PLINK data: {e}", err=True)
        raise typer.Exit(code=1) from None

    n_samples_raw = plink_data.n_samples
    n_snps = plink_data.n_snps
    typer.echo(f"Loaded {n_samples_raw} samples, {n_snps} SNPs")

    # Load kinship matrix
    typer.echo(f"Loading kinship matrix from {kinship_file}...")
    try:
        K = read_kinship_matrix(kinship_file, n_samples=n_samples_raw)
    except Exception as e:
        typer.echo(f"Error loading kinship matrix: {e}", err=True)
        raise typer.Exit(code=1) from None

    # Extract phenotypes from .fam file (6th column)
    # We load fam file directly to get phenotypes as bed-reader stores separately
    fam_data = np.loadtxt(fam_path, dtype=str, usecols=(5,))
    phenotypes = np.array(
        [float(x) if x not in ("-9", "NA") else np.nan for x in fam_data]
    )

    # Create valid sample mask (filter missing phenotypes: -9 or NaN)
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9)
    n_analyzed = int(valid_mask.sum())

    if n_analyzed == 0:
        typer.echo("Error: No samples with valid phenotypes", err=True)
        raise typer.Exit(code=1)

    n_filtered = n_samples_raw - n_analyzed
    typer.echo(
        f"Analyzing {n_analyzed} samples with valid phenotypes "
        f"(filtered {n_filtered})"
    )

    # Apply mask consistently to genotypes, phenotypes, AND kinship
    genotypes = plink_data.genotypes
    genotypes_filtered = genotypes[valid_mask, :]
    phenotypes_filtered = phenotypes[valid_mask]
    K_filtered = K[np.ix_(valid_mask, valid_mask)]

    # Validate dimensions after filtering
    assert (
        genotypes_filtered.shape[0]
        == phenotypes_filtered.shape[0]
        == K_filtered.shape[0]
    ), "Dimension mismatch after filtering"

    # Build snp_info list from PLINK metadata
    snp_info = []
    for i in range(n_snps):
        x = genotypes_filtered[:, i]
        n_miss = int(np.isnan(x).sum())
        # Minor allele frequency (using non-missing samples)
        valid_geno = x[~np.isnan(x)]
        maf = float(np.mean(valid_geno) / 2.0) if len(valid_geno) > 0 else 0.0

        snp_info.append(
            {
                "chr": str(plink_data.chromosome[i]),
                "rs": str(plink_data.sid[i]),
                "pos": int(plink_data.bp_position[i]),
                "a1": str(plink_data.allele_1[i]),
                "a0": str(plink_data.allele_2[i]),
                "maf": maf,
                "n_miss": n_miss,
            }
        )

    t_load = time.perf_counter()
    typer.echo(f"Data loading completed in {t_load - t_start:.2f}s")

    # Run LMM association
    typer.echo(f"Running LMM Wald test on {n_snps} SNPs...")

    # Progress callback for long runs
    last_progress = [0]  # Use list to allow modification in closure

    def progress_callback(i: int, n_total: int) -> None:
        """Print progress every 1000 SNPs or at 10% increments."""
        percent = int(100 * i / n_total)
        if i % 1000 == 0 or percent >= last_progress[0] + 10:
            typer.echo(f"  Processing SNP {i}/{n_total} ({percent}%)")
            last_progress[0] = percent

    # Run association testing
    results = run_lmm_association(
        genotypes_filtered,
        phenotypes_filtered,
        K_filtered,
        snp_info,
    )

    t_lmm = time.perf_counter()
    lmm_time = t_lmm - t_load
    typer.echo(f"LMM analysis completed in {lmm_time:.2f}s")

    # Write results
    assoc_path = _global_config.outdir / f"{_global_config.prefix}.assoc.txt"
    write_assoc_results(results, assoc_path)
    typer.echo(f"Association results written to {assoc_path}")

    # Calculate total time
    total_time = t_lmm - t_start
    load_time = t_load - t_start

    # Write log file with LMM-specific parameters
    params = {
        "n_samples": n_samples_raw,
        "n_analyzed": n_analyzed,
        "n_snps": n_snps,
        "lmm_mode": lmm_mode,
        "kinship_file": str(kinship_file),
        "output_file": str(assoc_path),
    }
    timing = {
        "total": total_time,
        "load": load_time,
        "lmm": lmm_time,
    }

    log_path = write_gemma_log(_global_config, params, timing, command_line)
    typer.echo(f"Log written to {log_path}")

    # Final summary
    typer.echo(f"\nAnalyzed {n_snps} SNPs in {total_time:.2f} seconds")


if __name__ == "__main__":
    app()
