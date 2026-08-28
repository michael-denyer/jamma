"""JAMMA command-line interface.

This module provides a Click-based CLI matching GEMMA's flat flag interface,
including -bfile, -gk, -lmm, -k, -o, -outdir flags for data loading,
mode selection, and output configuration.
"""

import os
import sys
import time
from pathlib import Path
from typing import Literal, NoReturn

import click
from loguru import logger

import jamma
from jamma.lmm.schema import DEFAULT_L_MAX, DEFAULT_L_MIN, DEFAULT_MAF, DEFAULT_MISS
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.pipeline_kinship import compute_kinship
from jamma.utils import setup_logging, write_gemma_log


def _cli_error(message: str) -> NoReturn:
    """Print error to stderr and exit with code 1."""
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


def _int_list(value: str, flag: str) -> list[int]:
    """Parse a space- or comma-separated list of integers from a CLI flag."""
    try:
        columns = [int(x) for x in value.replace(",", " ").split()]
    except ValueError as e:
        raise click.UsageError(
            f"{flag} must be integer column indices, got '{value}'"
        ) from e
    if not columns:
        raise click.UsageError(f"{flag} requires at least one column index")
    return columns


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print version and backend info, then exit."""
    if not value or ctx.resilient_parsing:
        return
    from jamma.core.backend import get_backend_info

    info = get_backend_info()
    click.echo(f"JAMMA version {jamma.__version__} ({jamma.__release_date__})")
    click.echo(f"Backend: {info['selected']}")
    ctx.exit()


@click.command()
@click.option(
    "-bfile",
    type=click.Path(path_type=Path),
    required=True,
    help="PLINK binary file prefix",
)
@click.option(
    "-gk",
    type=click.IntRange(1, 2),
    default=None,
    help="Kinship mode (1=centered, 2=standardized)",
)
@click.option(
    "-lmm", type=int, default=None, help="LMM mode (1=Wald, 2=LRT, 3=Score, 4=All)"
)
@click.option(
    "-k",
    type=click.Path(path_type=Path),
    default=None,
    help="Pre-computed kinship matrix file",
)
@click.option(
    "-c",
    type=click.Path(path_type=Path),
    default=None,
    help="Covariate file (whitespace-delimited, no header)",
)
@click.option("-o", type=str, default="result", help="Output file prefix")
@click.option(
    "-outdir",
    type=click.Path(path_type=Path),
    default="output",
    help="Output directory path",
)
@click.option(
    "-maf", type=float, default=DEFAULT_MAF, help="MAF threshold for SNP filtering"
)
@click.option("-miss", type=float, default=DEFAULT_MISS, help="Missing rate threshold")
@click.option(
    "-loco",
    is_flag=True,
    default=False,
    help="Enable leave-one-chromosome-out analysis",
)
@click.option(
    "-eigen", is_flag=True, default=False, help="Write eigendecomposition files"
)
@click.option(
    "--eigen-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory for LOCO per-chromosome eigen cache. When set with "
        "-lmm -loco, looks for cached eigen files to skip eigendecomp. "
        "Combined with -eigen, writes eigen files here."
    ),
)
@click.option(
    "-n",
    type=str,
    default="1",
    help=(
        "Phenotype column(s) in .fam file, 1-based. "
        "Single value or space/comma-separated: "
        "-n 1 or -n '1 2 3' or -n '1,2,3'"
    ),
)
@click.option(
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Eigenvalue file (.eigenD.npy or .txt)",
)
@click.option(
    "-u",
    type=click.Path(path_type=Path),
    default=None,
    help="Eigenvector file (.eigenU.npy or .txt)",
)
@click.option("-hwe", type=float, default=0.0, help="HWE p-value threshold")
@click.option(
    "-lmin",
    type=float,
    default=DEFAULT_L_MIN,
    help="Minimum lambda for optimization (default: 1e-5)",
)
@click.option(
    "-lmax",
    type=float,
    default=DEFAULT_L_MAX,
    help="Maximum lambda for optimization (default: 1e5)",
)
@click.option(
    "-snps",
    type=click.Path(path_type=Path),
    default=None,
    help="SNP list for association testing",
)
@click.option(
    "-ksnps",
    type=click.Path(path_type=Path),
    default=None,
    help="SNP list for kinship computation",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
@click.option(
    "--check-memory/--no-check-memory",
    default=True,
    help="Enable/disable pre-flight memory check",
)
@click.option("--mem-budget", type=float, default=None, help="Hard memory budget in GB")
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show version and exit",
)
@click.option(
    "--backend",
    type=click.Choice(
        ["auto", "numpy", "numpy-streaming"],
        case_sensitive=False,
    ),
    default="auto",
    help="Compute backend: auto, numpy, or numpy-streaming.",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    default=False,
    help="Disable benchmark telemetry for this run.",
)
@click.option(
    "--legacy-text",
    is_flag=True,
    default=False,
    help="Write kinship/eigen files in GEMMA text format instead of binary .npy",
)
@click.option(
    "-cat",
    type=str,
    default=None,
    help=(
        "Categorical covariate columns, 1-indexed. Single value or "
        "space/comma-separated: -cat '1 3' or -cat '1,3'. JAMMA-specific."
    ),
)
@click.option(
    "-widv",
    type=click.Path(path_type=Path),
    default=None,
    help="Individual weight file (one weight per line)",
)
@click.pass_context
def main(
    ctx,
    bfile,
    gk,
    lmm,
    k,
    c,
    o,
    outdir,
    maf,
    miss,
    loco,
    eigen,
    eigen_dir,
    n,
    d,
    u,
    hwe,
    lmin,
    lmax,
    snps,
    ksnps,
    verbose,
    check_memory,
    mem_budget,
    backend,
    no_telemetry,
    legacy_text,
    cat,
    widv,
):
    """JAMMA: Highly-Accelerated Multi-method Mixed-Model Association.

    A modern Python and C reimplementation of GEMMA for large-scale GWAS.
    """
    setup_logging(verbose=verbose)

    if no_telemetry:
        os.environ["JAMMA_NO_TELEMETRY"] = "1"

    # CLI policy: rules about flags PipelineConfig has no field for. Every
    # rule about a knob the config does carry lives in its __post_init__ (or
    # LmmConfig's) and surfaces below as a usage error when the config is built.
    if gk is not None and lmm is not None:
        raise click.UsageError("-gk and -lmm are mutually exclusive")
    # Not chained to `gk is lmm is None` (FURB124, silenced in pyproject):
    # the line above is its mirror image, and the pair should look alike.
    if gk is None and lmm is None:
        raise click.UsageError("One of -gk or -lmm is required")

    if mem_budget is not None and mem_budget <= 0:
        raise click.UsageError(f"--mem-budget must be positive, got {mem_budget}")

    phenotype_columns = _int_list(n, "-n")
    cat_columns = _int_list(cat, "-cat") if cat is not None else None

    if gk is not None:
        # gk mode: no filtering by default (GEMMA kinship behavior)
        # lmm mode: standard filtering (GEMMA association behavior)
        if ctx.get_parameter_source("maf") == click.core.ParameterSource.DEFAULT:
            maf = 0.0
        if ctx.get_parameter_source("miss") == click.core.ParameterSource.DEFAULT:
            miss = 1.0
        if len(phenotype_columns) > 1:
            raise click.UsageError(
                "-n with multiple columns is not supported in -gk mode. "
                "Kinship computation uses all samples regardless of phenotype."
            )

    try:
        pipeline_config = PipelineConfig(
            bfile=bfile,
            kinship_file=k,
            covariate_file=c,
            lmm_mode=1 if lmm is None else lmm,
            maf=maf,
            miss=miss,
            output_dir=outdir,
            output_prefix=o,
            check_memory=check_memory,
            show_progress=True,
            mem_budget=mem_budget,
            loco=loco,
            eigenvalue_file=d,
            eigenvector_file=u,
            write_eigen=eigen,
            eigen_dir=eigen_dir,
            phenotype_columns=phenotype_columns,
            snps_file=snps,
            ksnps_file=ksnps,
            hwe_threshold=hwe,
            l_min=lmin,
            l_max=lmax,
            weight_file=widv,
            cat_columns=cat_columns,
            backend=backend,
            legacy_text=legacy_text,
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    if gk is not None:
        _run_gk(pipeline_config, gk)
        return

    # The CLI requires a kinship source up front; the Python API computes one
    # when none is given.
    if k is None and not loco and d is None:
        _cli_error(
            "-k (kinship matrix) is required for -lmm "
            "(or use -d/-u for pre-computed eigen)"
        )
    if eigen_dir is not None and not loco:
        _cli_error("--eigen-dir is only supported with -loco mode")

    _run_lmm(pipeline_config)


def _run_gk(config: PipelineConfig, mode: Literal[1, 2]) -> None:
    """Run kinship matrix computation (thin shell over compute_kinship)."""
    start_time = time.perf_counter()
    config.ensure_outdir()
    command_line = " ".join(sys.argv)

    try:
        result = compute_kinship(config, mode)
    except (FileNotFoundError, ValueError, MemoryError, OSError) as e:
        logger.debug("Kinship computation failed with traceback:", exc_info=True)
        _cli_error(str(e))

    if result.is_loco:
        click.echo(f"Wrote {len(result.kinship_paths)} LOCO kinship matrices")
        for p in result.kinship_paths:
            click.echo(f"  {p}")
    else:
        click.echo(f"Kinship matrix written to {result.kinship_paths[0]}")
        if result.eigen_paths is not None:
            click.echo(f"Eigenvalues written to {result.eigen_paths[0]}")
            click.echo(f"Eigenvectors written to {result.eigen_paths[1]}")

    elapsed = time.perf_counter() - start_time
    if result.is_loco:
        params = {
            "kinship_mode": "loco",
            "n_chromosomes": len(result.kinship_paths),
            "maf_threshold": config.maf,
            "miss_threshold": config.miss,
        }
    else:
        params = {
            "n_samples": result.n_samples,
            "n_snps": result.n_snps,
            "kinship_mode": mode,
            "kinship_file": str(result.kinship_paths[0]),
            "maf_threshold": config.maf,
            "miss_threshold": config.miss,
        }
    timing = {"total": elapsed, "kinship": result.kinship_s}
    log_path = write_gemma_log(config, params, timing, command_line)
    click.echo(f"Log written to {log_path}")


def _run_lmm(config: PipelineConfig) -> None:
    """Run LMM association testing (thin shell over PipelineRunner)."""
    try:
        if config.check_memory:
            click.echo("Checking memory requirements...")
        result = PipelineRunner(config).run()
    except (FileNotFoundError, ValueError, MemoryError, OSError) as e:
        logger.debug("Pipeline failed with traceback:", exc_info=True)
        _cli_error(str(e))

    command_line = " ".join(sys.argv)
    params = {
        "n_samples": result.n_samples,
        "n_snps": result.n_snps_tested,
        "lmm_mode": config.lmm_mode,
        "kinship_file": str(config.kinship_file),
        "covariate_file": str(config.covariate_file) if config.covariate_file else None,
        "n_covariates": result.n_covariates,
        "output_file": str(result.assoc_path),
        "maf_threshold": config.maf,
        "miss_threshold": config.miss,
        "check_memory": config.check_memory,
        "mem_budget": config.mem_budget,
    }
    timing = {
        "total": result.timing.total_s,
        "load": result.timing.load_s,
        "lmm": result.timing.lmm_s,
    }

    config.ensure_outdir()
    log_path = write_gemma_log(config, params, timing, command_line)
    click.echo(f"Log written to {log_path}")

    click.echo(
        f"\nAnalyzed {result.n_snps_tested} SNPs in {result.timing.total_s:.2f} seconds"
    )


if __name__ == "__main__":
    main()
