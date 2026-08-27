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
from jamma.core import OutputConfig
from jamma.lmm.schema import DEFAULT_L_MAX, DEFAULT_L_MIN, DEFAULT_MAF, DEFAULT_MISS
from jamma.pipeline import BackendRequest, PipelineConfig, PipelineRunner
from jamma.pipeline_kinship import compute_kinship
from jamma.utils import setup_logging, write_gemma_log


def _cli_error(message: str) -> NoReturn:
    """Print error to stderr and exit with code 1."""
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


def _opt_path(value: str | None) -> Path | None:
    """Convert optional string to Path."""
    return Path(value) if value else None


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
    "-bfile", type=click.Path(), required=True, help="PLINK binary file prefix"
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
    "-k", type=click.Path(), default=None, help="Pre-computed kinship matrix file"
)
@click.option(
    "-c",
    type=click.Path(),
    default=None,
    help="Covariate file (whitespace-delimited, no header)",
)
@click.option("-o", type=str, default="result", help="Output file prefix")
@click.option(
    "-outdir", type=click.Path(), default="output", help="Output directory path"
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
    type=click.Path(),
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
    "-d", type=click.Path(), default=None, help="Eigenvalue file (.eigenD.npy or .txt)"
)
@click.option(
    "-u", type=click.Path(), default=None, help="Eigenvector file (.eigenU.npy or .txt)"
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
    "-snps", type=click.Path(), default=None, help="SNP list for association testing"
)
@click.option(
    "-ksnps", type=click.Path(), default=None, help="SNP list for kinship computation"
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
    type=click.Path(),
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

    try:
        config = OutputConfig(outdir=Path(outdir), prefix=o, verbose=verbose)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    # Mode validation: exactly one of -gk or -lmm required
    if gk is not None and lmm is not None:
        raise click.UsageError("-gk and -lmm are mutually exclusive")
    # Not chained to `gk is lmm is None` (FURB124, silenced in pyproject):
    # the line above is its mirror image, and the pair should look alike.
    if gk is None and lmm is None:
        raise click.UsageError("One of -gk or -lmm is required")

    # Validate memory budget
    if mem_budget is not None and mem_budget <= 0:
        raise click.UsageError(f"--mem-budget must be positive, got {mem_budget}")

    # Validate lambda bounds (before pipeline construction)
    if lmin <= 0:
        raise click.UsageError(f"-lmin must be > 0, got {lmin}")
    if lmax <= lmin:
        raise click.UsageError(f"-lmax must be > -lmin ({lmin}), got {lmax}")

    # Apply per-mode defaults for maf/miss.
    # gk mode: no filtering by default (GEMMA kinship behavior)
    # lmm mode: standard filtering (GEMMA association behavior)
    if gk is not None:
        if ctx.get_parameter_source("maf") == click.core.ParameterSource.DEFAULT:
            maf = 0.0
        if ctx.get_parameter_source("miss") == click.core.ParameterSource.DEFAULT:
            miss = 1.0

    phenotype_columns = _int_list(n, "-n")
    if len(phenotype_columns) != len(set(phenotype_columns)):
        raise click.UsageError(
            f"-n contains duplicate column indices: {phenotype_columns}"
        )
    cat_columns = _int_list(cat, "-cat") if cat is not None else None

    # Dispatch to handler. Both arms test their own mode flag rather than
    # relying on the mutually-exclusive/one-required checks far above, so each
    # handler's mode argument is known to be set at the call.
    if gk is not None:
        if len(phenotype_columns) > 1:
            raise click.UsageError(
                "-n with multiple columns is not supported in -gk mode. "
                "Kinship computation uses all samples regardless of phenotype."
            )
        _run_gk(
            bfile=Path(bfile),
            mode=gk,
            config=config,
            maf=maf,
            miss=miss,
            check_memory=check_memory,
            loco=loco,
            write_eigen=eigen,
            ksnps_file=_opt_path(ksnps),
            legacy_text=legacy_text,
        )
    elif lmm is not None:
        _run_lmm(
            bfile=Path(bfile),
            mode=lmm,
            config=config,
            kinship_file=_opt_path(k),
            covariate_file=_opt_path(c),
            maf=maf,
            miss=miss,
            check_memory=check_memory,
            mem_budget=mem_budget,
            loco=loco,
            eigenvalue_file=_opt_path(d),
            eigenvector_file=_opt_path(u),
            write_eigen=eigen,
            eigen_dir=_opt_path(eigen_dir),
            phenotype_columns=phenotype_columns,
            snps_file=_opt_path(snps),
            ksnps_file=_opt_path(ksnps),
            hwe_threshold=hwe,
            l_min=lmin,
            l_max=lmax,
            weight_file=_opt_path(widv),
            cat_columns=cat_columns,
            backend=backend,
            legacy_text=legacy_text,
        )
    else:  # pragma: no cover - the checks above already rejected this
        raise click.UsageError("One of -gk or -lmm is required")


def _run_gk(
    *,
    bfile: Path,
    mode: Literal[1, 2],
    config: OutputConfig,
    maf: float,
    miss: float,
    check_memory: bool,
    loco: bool,
    write_eigen: bool,
    ksnps_file: Path | None,
    legacy_text: bool = False,
) -> None:
    """Run kinship matrix computation (thin shell over compute_kinship)."""
    start_time = time.perf_counter()
    config.ensure_outdir()
    command_line = " ".join(sys.argv)

    # Construction is inside the try because PipelineConfig validates its
    # knobs in __post_init__, and compute_kinship validates the -gk flag
    # combinations at its top; either should read as a CLI error.
    try:
        pipeline_config = PipelineConfig(
            bfile=bfile,
            maf=maf,
            miss=miss,
            output_dir=config.outdir,
            output_prefix=config.prefix,
            check_memory=check_memory,
            show_progress=True,
            loco=loco,
            write_eigen=write_eigen,
            ksnps_file=ksnps_file,
            legacy_text=legacy_text,
        )
        result = compute_kinship(pipeline_config, mode)
    except (FileNotFoundError, ValueError, MemoryError, OSError) as e:
        logger.debug("Kinship computation failed with traceback:", exc_info=True)
        _cli_error(str(e))

    # Summary (CLI-facing)
    if result.is_loco:
        click.echo(f"Wrote {len(result.kinship_paths)} LOCO kinship matrices")
        for p in result.kinship_paths:
            click.echo(f"  {p}")
    else:
        click.echo(f"Kinship matrix written to {result.kinship_paths[0]}")
        if result.eigen_paths is not None:
            click.echo(f"Eigenvalues written to {result.eigen_paths[0]}")
            click.echo(f"Eigenvectors written to {result.eigen_paths[1]}")

    # Write GEMMA log file (CLI-only)
    elapsed = time.perf_counter() - start_time
    if result.is_loco:
        params = {
            "kinship_mode": "loco",
            "n_chromosomes": len(result.kinship_paths),
            "maf_threshold": maf,
            "miss_threshold": miss,
        }
    else:
        params = {
            "n_samples": result.n_samples,
            "n_snps": result.n_snps,
            "kinship_mode": mode,
            "kinship_file": str(result.kinship_paths[0]),
            "maf_threshold": maf,
            "miss_threshold": miss,
        }
    timing = {"total": elapsed, "kinship": result.kinship_s}
    log_path = write_gemma_log(config, params, timing, command_line)
    click.echo(f"Log written to {log_path}")


def _run_lmm(
    *,
    bfile: Path,
    mode: int,
    config: OutputConfig,
    kinship_file: Path | None,
    covariate_file: Path | None,
    maf: float,
    miss: float,
    check_memory: bool,
    mem_budget: float | None,
    loco: bool,
    eigenvalue_file: Path | None,
    eigenvector_file: Path | None,
    write_eigen: bool,
    eigen_dir: Path | None,
    phenotype_columns: list[int],
    snps_file: Path | None,
    ksnps_file: Path | None,
    hwe_threshold: float,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    weight_file: Path | None = None,
    cat_columns: list[int] | None = None,
    backend: BackendRequest = "auto",
    legacy_text: bool = False,
) -> None:
    """Run LMM association testing."""
    # Mutual exclusivity check
    if loco and kinship_file is not None:
        _cli_error("-k and -loco are mutually exclusive")

    # CLI requires kinship file unless LOCO mode or eigen files
    if kinship_file is None and not loco and eigenvalue_file is None:
        _cli_error(
            "-k (kinship matrix) is required for -lmm "
            "(or use -d/-u for pre-computed eigen)"
        )

    # --eigen-dir only makes sense with -loco
    if eigen_dir is not None and not loco:
        _cli_error("--eigen-dir is only supported with -loco mode")

    # Default eigen_dir to output_dir when -eigen is set with LOCO
    if write_eigen and loco and eigen_dir is None:
        eigen_dir = config.outdir

    # Build and run the pipeline, converting exceptions to CLI-friendly errors.
    # Construction is inside the try because PipelineConfig validates its knobs
    # in __post_init__, and a bad value (-lmm 99) should read as a CLI error
    # rather than a traceback.
    try:
        pipeline_config = PipelineConfig(
            bfile=bfile,
            kinship_file=kinship_file,
            covariate_file=covariate_file,
            lmm_mode=mode,
            maf=maf,
            miss=miss,
            output_dir=config.outdir,
            output_prefix=config.prefix,
            check_memory=check_memory,
            show_progress=True,
            mem_budget=mem_budget,
            loco=loco,
            eigenvalue_file=eigenvalue_file,
            eigenvector_file=eigenvector_file,
            write_eigen=write_eigen,
            eigen_dir=eigen_dir,
            phenotype_columns=phenotype_columns,
            snps_file=snps_file,
            ksnps_file=ksnps_file,
            hwe_threshold=hwe_threshold,
            l_min=l_min,
            l_max=l_max,
            weight_file=weight_file,
            cat_columns=cat_columns,
            backend=backend,
            legacy_text=legacy_text,
        )
        if check_memory:
            click.echo("Checking memory requirements...")
        result = PipelineRunner(pipeline_config).run()
    except (FileNotFoundError, ValueError, MemoryError, OSError) as e:
        logger.debug("Pipeline failed with traceback:", exc_info=True)
        _cli_error(str(e))

    # Write GEMMA log file (CLI-only)
    command_line = " ".join(sys.argv)
    n_covariates = result.n_covariates

    params = {
        "n_samples": result.n_samples,
        "n_snps": result.n_snps_tested,
        "backend": result.backend,
        "lmm_mode": mode,
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
        "total": result.timing.get("total_s", 0.0),
        "load": result.timing.get("load_s", 0.0),
        "lmm": result.timing.get("lmm_s", 0.0),
    }

    config.ensure_outdir()
    log_path = write_gemma_log(config, params, timing, command_line)
    click.echo(f"Log written to {log_path}")

    # Final summary
    click.echo(
        f"\nAnalyzed {result.n_snps_tested} SNPs "
        f"in {result.timing.get('total_s', 0.0):.2f} seconds"
    )


if __name__ == "__main__":
    main()
