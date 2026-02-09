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
from jamma.core.memory import estimate_streaming_memory
from jamma.io import get_plink_metadata, load_plink_binary, read_covariate_file
from jamma.kinship import (
    compute_centered_kinship,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm import run_lmm_association_streaming
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
        from jamma.core import get_backend_info

        typer.echo(f"JAMMA version {jamma.__version__}")

        # Show backend information for debugging
        info = get_backend_info()
        typer.echo(f"Backend: {info['selected']}")
        typer.echo("  (JAX pipeline + numpy/LAPACK eigendecomp)")
        typer.echo(f"GPU available: {info['gpu_available']}")
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
    maf: Annotated[
        float,
        typer.Option("-maf", help="MAF threshold for SNP filtering (default: 0.0)"),
    ] = 0.0,
    miss: Annotated[
        float,
        typer.Option("-miss", help="Missing rate threshold (default: 1.0)"),
    ] = 1.0,
    check_memory: Annotated[
        bool,
        typer.Option(
            "--check-memory/--no-check-memory",
            help="Enable/disable pre-flight memory check (default: enabled)",
        ),
    ] = True,
) -> None:
    """Compute kinship matrix from genotype data.

    Reads PLINK binary files and computes a genetic relatedness matrix.
    Writes output in GEMMA-compatible .cXX.txt format.
    """
    start_time = time.perf_counter()

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

    # Mode 2 (standardized) is not yet implemented - fail loudly
    if mode == 2:
        raise NotImplementedError(
            "Kinship mode 2 (standardized) is not yet implemented. "
            "Use -gk 1 for centered relatedness matrix."
        )

    # Compute kinship matrix
    typer.echo("Computing centered kinship matrix...")
    if maf > 0.0 or miss < 1.0:
        typer.echo(f"Filtering: MAF >= {maf}, missing rate <= {miss}")
    kinship_start = time.perf_counter()
    K = compute_centered_kinship(
        plink_data.genotypes,
        maf_threshold=maf,
        miss_threshold=miss,
        check_memory=check_memory,
    )
    kinship_time = time.perf_counter() - kinship_start
    typer.echo(f"Kinship computation completed in {kinship_time:.2f}s")

    # Write kinship matrix
    kinship_path = _global_config.outdir / f"{_global_config.prefix}.cXX.txt"
    write_kinship_matrix(K, kinship_path)
    typer.echo(f"Kinship matrix written to {kinship_path}")

    # Calculate timing
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Write log file
    params = {
        "n_samples": plink_data.n_samples,
        "n_snps": plink_data.n_snps,
        "kinship_mode": mode,
        "kinship_file": str(kinship_path),
        "maf_threshold": maf,
        "miss_threshold": miss,
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
    covariate_file: Annotated[
        Path | None,
        typer.Option("-c", help="Covariate file (whitespace-delimited, no header)"),
    ] = None,
    lmm_mode: Annotated[
        int,
        typer.Option("-lmm", help="LMM analysis type (1=Wald, 2=LRT, 3=Score, 4=All)"),
    ] = 1,
    maf: Annotated[
        float,
        typer.Option("-maf", help="MAF threshold for SNP filtering (default: 0.01)"),
    ] = 0.01,
    miss: Annotated[
        float,
        typer.Option("-miss", help="Missing rate threshold (default: 0.05)"),
    ] = 0.05,
    check_memory: Annotated[
        bool,
        typer.Option(
            "--check-memory/--no-check-memory",
            help="Enable/disable pre-flight memory check (default: enabled)",
        ),
    ] = True,
    mem_budget: Annotated[
        float | None,
        typer.Option(
            "--mem-budget",
            help="Hard memory budget in GB. Fail if estimate exceeds this.",
        ),
    ] = None,
) -> None:
    """Perform linear mixed model association testing.

    Runs LMM association tests using a pre-computed kinship matrix.
    Supports Wald test (-lmm 1), LRT (-lmm 2), Score test (-lmm 3),
    and all tests combined (-lmm 4).
    """
    # Get global config
    global _global_config
    if _global_config is None:
        _global_config = OutputConfig()

    # Validate -lmm mode
    if lmm_mode not in (1, 2, 3, 4):
        typer.echo(f"Error: -lmm must be 1, 2, 3, or 4 (got {lmm_mode})", err=True)
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

    # Validate covariate file exists if provided
    if covariate_file is not None and not covariate_file.exists():
        typer.echo(f"Error: Covariate file not found: {covariate_file}", err=True)
        raise typer.Exit(code=1)

    # Always fetch PLINK metadata (needed for memory check AND progress message)
    meta = get_plink_metadata(bfile)
    n_samples_meta = meta["n_samples"]
    n_snps_meta = meta["n_snps"]

    # Pre-flight memory check (before any large allocations)
    if check_memory:
        typer.echo("Checking memory requirements...")

        est = estimate_streaming_memory(
            n_samples=n_samples_meta,
            n_snps=n_snps_meta,
            chunk_size=10_000,
        )

        typer.echo(
            f"Memory estimate: {est.total_peak_gb:.1f}GB required, "
            f"{est.available_gb:.1f}GB available"
        )

        # Check hard budget if specified
        if mem_budget is not None and est.total_peak_gb > mem_budget:
            typer.echo(
                f"Error: Estimated memory ({est.total_peak_gb:.1f}GB) exceeds "
                f"budget ({mem_budget}GB). Use --no-check-memory to override.",
                err=True,
            )
            raise typer.Exit(code=1)

        # Check available memory
        if not est.sufficient:
            typer.echo(
                f"Error: Insufficient memory. "
                f"Need {est.total_peak_gb:.1f}GB (with 10% margin), "
                f"have {est.available_gb:.1f}GB. "
                f"Use --no-check-memory to override.",
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo("Memory check passed.")

    # Ensure output directory exists
    _global_config.ensure_outdir()

    # Record start time
    t_start = time.perf_counter()
    command_line = " ".join(sys.argv)

    # === Data loading ===

    # Parse phenotypes from .fam file
    fam_data = np.loadtxt(fam_path, dtype=str, usecols=(5,))
    phenotypes = np.array(
        [float(x) if x not in ("-9", "NA") else np.nan for x in fam_data]
    )
    n_samples_raw = len(phenotypes)

    # Count valid samples for logging (before any filtering)
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

    # Load kinship matrix
    typer.echo(f"Loading kinship matrix from {kinship_file}...")
    try:
        K = read_kinship_matrix(kinship_file, n_samples=n_samples_raw)
    except Exception as e:
        typer.echo(f"Error loading kinship matrix: {e}", err=True)
        raise typer.Exit(code=1) from None

    # Load covariate file if provided
    covariates = None
    if covariate_file is not None:
        typer.echo(f"Loading covariates from {covariate_file}...")
        try:
            covariates, indicator_cvt = read_covariate_file(covariate_file)
        except Exception as e:
            typer.echo(f"Error loading covariate file: {e}", err=True)
            raise typer.Exit(code=1) from None

        if covariates.shape[0] != n_samples_raw:
            typer.echo(
                f"Error: Covariate file has {covariates.shape[0]} rows "
                f"but PLINK data has {n_samples_raw} samples. "
                f"Covariate rows must match sample count exactly.",
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo(f"Loaded {covariates.shape[1]} covariates")

        first_col = covariates[:, 0]
        valid_first = first_col[~np.isnan(first_col)]
        if not np.allclose(valid_first, 1.0):
            typer.echo(
                "Warning: Covariate file does not have intercept column "
                "(first column is not all 1s). Model will NOT include intercept.",
                err=True,
            )

    t_load = time.perf_counter()
    typer.echo(f"Data loading completed in {t_load - t_start:.2f}s")

    # === Run LMM association (JAX streaming) ===
    test_name = {1: "Wald", 2: "LRT", 3: "Score", 4: "All tests"}[lmm_mode]
    typer.echo(f"Running LMM {test_name} test on {n_snps_meta} SNPs...")

    assoc_path = _global_config.outdir / f"{_global_config.prefix}.assoc.txt"

    run_lmm_association_streaming(
        bed_path=bfile,
        phenotypes=phenotypes,
        kinship=K,
        snp_info=None,
        covariates=covariates,
        maf_threshold=maf,
        miss_threshold=miss,
        output_path=assoc_path,
        lmm_mode=lmm_mode,
        check_memory=False,  # Already checked above
        show_progress=True,
    )

    t_lmm = time.perf_counter()
    lmm_time = t_lmm - t_load
    typer.echo(f"LMM analysis completed in {lmm_time:.2f}s")

    typer.echo(f"Association results written to {assoc_path}")

    # Calculate total time
    total_time = t_lmm - t_start
    load_time = t_load - t_start

    # Write log file with LMM-specific parameters
    params = {
        "n_samples": n_samples_raw,
        "n_analyzed": n_analyzed,
        "n_snps": n_snps_meta,
        "backend": "jax",
        "lmm_mode": lmm_mode,
        "kinship_file": str(kinship_file),
        "covariate_file": str(covariate_file) if covariate_file else None,
        "n_covariates": covariates.shape[1] if covariates is not None else 1,
        "output_file": str(assoc_path),
        "maf_threshold": maf,
        "miss_threshold": miss,
        "check_memory": check_memory,
        "mem_budget": mem_budget,
    }
    timing = {
        "total": total_time,
        "load": load_time,
        "lmm": lmm_time,
    }

    log_path = write_gemma_log(_global_config, params, timing, command_line)
    typer.echo(f"Log written to {log_path}")

    # Final summary
    typer.echo(f"\nAnalyzed {n_snps_meta} SNPs in {total_time:.2f} seconds")


if __name__ == "__main__":
    app()
