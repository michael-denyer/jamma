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
from jamma.kinship import (
    compute_centered_kinship,
    compute_loco_kinship_streaming,
    write_kinship_matrix,
    write_loco_kinship_matrices,
)
from jamma.pipeline import PipelineConfig, PipelineRunner
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
    loco: Annotated[
        bool,
        typer.Option(
            "-loco/--no-loco",
            help="Compute LOCO kinship matrices (one per chromosome)",
        ),
    ] = False,
    write_eigen: Annotated[
        bool,
        typer.Option(
            "-eigen",
            help="Write eigendecomposition files (.eigenD.txt, .eigenU.txt)",
        ),
    ] = False,
) -> None:
    """Compute kinship matrix from genotype data.

    Reads PLINK binary files and computes a genetic relatedness matrix.
    Writes output in GEMMA-compatible .cXX.txt format.

    With -loco, computes one LOCO kinship matrix per chromosome, writing
    each to {prefix}.loco.cXX.chr{N}.txt.

    With -eigen, also eigendecomposes the kinship matrix and writes
    .eigenD.txt and .eigenU.txt files.
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

    if write_eigen and loco:
        typer.echo(
            "Error: -eigen not supported with -loco mode",
            err=True,
        )
        raise typer.Exit(code=1)

    if loco:
        # LOCO kinship mode: compute per-chromosome LOCO kinship matrices
        typer.echo(f"Computing LOCO kinship matrices from {bfile}...")
        kinship_start = time.perf_counter()

        loco_iter = compute_loco_kinship_streaming(
            bfile,
            maf_threshold=maf,
            miss_threshold=miss,
            check_memory=check_memory,
            show_progress=True,
        )
        written_paths = write_loco_kinship_matrices(
            loco_iter,
            output_dir=_global_config.outdir,
            prefix=_global_config.prefix,
        )
        kinship_time = time.perf_counter() - kinship_start

        typer.echo(
            f"Wrote {len(written_paths)} LOCO kinship matrices in {kinship_time:.2f}s"
        )
        for p in written_paths:
            typer.echo(f"  {p}")

        # Write log file
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        params = {
            "kinship_mode": "loco",
            "n_chromosomes": len(written_paths),
            "maf_threshold": maf,
            "miss_threshold": miss,
        }
        timing = {"total": elapsed, "kinship": kinship_time}
        log_path = write_gemma_log(_global_config, params, timing, command_line)
        typer.echo(f"Log written to {log_path}")
        return

    # Standard kinship mode
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

    # Eigendecompose and write if -eigen flag
    if write_eigen:
        from jamma.lmm.eigen import eigendecompose_kinship
        from jamma.lmm.eigen_io import write_eigen_files

        eigenvalues, eigenvectors = eigendecompose_kinship(K)
        d_path, u_path = write_eigen_files(
            eigenvalues,
            eigenvectors,
            _global_config.outdir,
            _global_config.prefix,
        )
        typer.echo(f"Eigenvalues written to {d_path}")
        typer.echo(f"Eigenvectors written to {u_path}")

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
    loco: Annotated[
        bool,
        typer.Option(
            "-loco/--no-loco",
            help="Enable leave-one-chromosome-out analysis",
        ),
    ] = False,
    eigenvalue_file: Annotated[
        Path | None,
        typer.Option(
            "-d",
            help="Pre-computed eigenvalues file (.eigenD.txt)",
        ),
    ] = None,
    eigenvector_file: Annotated[
        Path | None,
        typer.Option(
            "-u",
            help="Pre-computed eigenvectors file (.eigenU.txt)",
        ),
    ] = None,
    write_eigen: Annotated[
        bool,
        typer.Option(
            "-eigen",
            help="Write eigendecomposition output files",
        ),
    ] = False,
) -> None:
    """Perform linear mixed model association testing.

    Runs LMM association tests using a pre-computed kinship matrix.
    Supports Wald test (-lmm 1), LRT (-lmm 2), Score test (-lmm 3),
    and all tests combined (-lmm 4).

    With -loco, computes per-chromosome LOCO kinship internally.
    The -k flag is not required (and is mutually exclusive with -loco).

    With -d and -u, loads pre-computed eigendecomposition and skips
    both kinship loading and eigendecomposition. Both flags required
    together.
    """
    global _global_config
    if _global_config is None:
        _global_config = OutputConfig()

    # Mutual exclusivity check
    if loco and kinship_file is not None:
        typer.echo(
            "Error: -k and -loco are mutually exclusive",
            err=True,
        )
        raise typer.Exit(code=1)

    # CLI requires kinship file unless LOCO mode or eigen files
    if kinship_file is None and not loco and eigenvalue_file is None:
        typer.echo(
            "Error: -k (kinship matrix) is required for -lmm "
            "(or use -d/-u for pre-computed eigen)",
            err=True,
        )
        raise typer.Exit(code=1)

    # Build pipeline config
    config = PipelineConfig(
        bfile=bfile,
        kinship_file=kinship_file,
        covariate_file=covariate_file,
        lmm_mode=lmm_mode,
        maf=maf,
        miss=miss,
        output_dir=_global_config.outdir,
        output_prefix=_global_config.prefix,
        check_memory=check_memory,
        show_progress=True,
        mem_budget=mem_budget,
        loco=loco,
        eigenvalue_file=eigenvalue_file,
        eigenvector_file=eigenvector_file,
        write_eigen=write_eigen,
    )

    # Run pipeline, converting exceptions to CLI-friendly errors
    try:
        if check_memory:
            typer.echo("Checking memory requirements...")
        result = PipelineRunner(config).run()
    except (FileNotFoundError, ValueError, MemoryError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    # Write GEMMA log file (CLI-only)
    command_line = " ".join(sys.argv)
    n_covariates = 1
    if covariate_file is not None:
        # Covariate count from result timing context
        n_covariates = result.timing.get("n_covariates", 1)

    params = {
        "n_samples": result.n_samples,
        "n_snps": result.n_snps_tested,
        "backend": "jax",
        "lmm_mode": lmm_mode,
        "kinship_file": str(kinship_file),
        "covariate_file": str(covariate_file) if covariate_file else None,
        "n_covariates": n_covariates,
        "output_file": str(result.assoc_path),
        "maf_threshold": maf,
        "miss_threshold": miss,
        "check_memory": check_memory,
        "mem_budget": mem_budget,
    }
    timing = {
        "total": result.timing["total_s"],
        "load": result.timing["load_s"],
        "lmm": result.timing["lmm_s"],
    }

    _global_config.ensure_outdir()
    log_path = write_gemma_log(_global_config, params, timing, command_line)
    typer.echo(f"Log written to {log_path}")

    # Final summary
    typer.echo(
        f"\nAnalyzed {result.n_snps_tested} SNPs "
        f"in {result.timing['total_s']:.2f} seconds"
    )


if __name__ == "__main__":
    app()
