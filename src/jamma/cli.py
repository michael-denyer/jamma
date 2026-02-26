"""JAMMA command-line interface.

This module provides a Click-based CLI matching GEMMA's flat flag interface,
including -bfile, -gk, -lmm, -k, -o, -outdir flags for data loading,
mode selection, and output configuration.
"""

import sys
import time
from pathlib import Path
from typing import NoReturn

import click
from loguru import logger

import jamma
from jamma.core import OutputConfig
from jamma.io import load_plink_binary
from jamma.kinship import (
    compute_kinship_streaming,
    compute_loco_kinship_streaming,
    compute_standardized_kinship,
    write_kinship_matrix,
    write_loco_kinship_matrices,
)
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.utils import setup_logging, write_gemma_log


def _cli_error(message: str) -> NoReturn:
    """Print error to stderr and exit with code 1."""
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


def _opt_path(value: str | None) -> Path | None:
    """Convert optional string to Path."""
    return Path(value) if value else None


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print version, backend info, and GPU status, then exit."""
    if not value or ctx.resilient_parsing:
        return
    from jamma.core import get_backend_info

    info = get_backend_info()
    click.echo(f"JAMMA version {jamma.__version__} ({jamma.__release_date__})")
    click.echo(f"Backend: {info['selected']}")
    click.echo("  (JAX pipeline + numpy/LAPACK eigendecomp)")
    click.echo(f"GPU available: {info['gpu_available']}")
    ctx.exit()


@click.command()
@click.option(
    "-bfile", type=click.Path(), required=True, help="PLINK binary file prefix"
)
@click.option(
    "-gk", type=int, default=None, help="Kinship mode (1=centered, 2=standardized)"
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
@click.option("-maf", type=float, default=0.01, help="MAF threshold for SNP filtering")
@click.option("-miss", type=float, default=0.05, help="Missing rate threshold")
@click.option(
    "-loco",
    is_flag=True,
    default=False,
    help="Enable leave-one-chromosome-out analysis",
)
@click.option(
    "-eigen", is_flag=True, default=False, help="Write eigendecomposition files"
)
@click.option("-n", type=int, default=1, help="Phenotype column in .fam file (1-based)")
@click.option(
    "-d", type=click.Path(), default=None, help="Eigenvalue file (.eigenD.txt)"
)
@click.option(
    "-u", type=click.Path(), default=None, help="Eigenvector file (.eigenU.txt)"
)
@click.option("-hwe", type=float, default=0.0, help="HWE p-value threshold")
@click.option(
    "-lmin",
    type=float,
    default=1e-5,
    help="Minimum lambda for optimization (default: 1e-5)",
)
@click.option(
    "-lmax",
    type=float,
    default=1e5,
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
    "--profile-dir",
    type=click.Path(),
    default=None,
    help="Directory for XLA profiling traces (view with TensorBoard)",
)
@click.option(
    "-cat",
    type=str,
    default=None,
    help="Categorical covariate columns, 1-indexed (e.g., -cat '1 3'). JAMMA-specific.",
)
@click.option(
    "-widv",
    type=click.Path(),
    default=None,
    help="Individual weight file (one weight per line)",
)
@click.option(
    "-wsnp",
    type=click.Path(),
    default=None,
    hidden=True,
    help="SNP weight file (not yet implemented)",
)
@click.option(
    "-gxe",
    type=click.Path(),
    default=None,
    hidden=True,
    help="G x E interaction file (not yet implemented)",
)
@click.option(
    "-vc",
    type=int,
    default=None,
    hidden=True,
    help="Variance component estimation (not yet implemented)",
)
@click.option(
    "-mk",
    type=click.Path(),
    default=None,
    hidden=True,
    help="Multiple kinship matrices (not yet implemented)",
)
@click.option(
    "-mvlmm",
    type=int,
    default=None,
    hidden=True,
    help="Multivariate LMM (not yet implemented)",
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
    profile_dir,
    cat,
    widv,
    wsnp,
    gxe,
    vc,
    mk,
    mvlmm,
):
    """JAMMA: Mixed Model Association for GWAS.

    A modern Python reimplementation of GEMMA targeting exact numerical
    compatibility with the original C++ implementation.
    """
    setup_logging(verbose=verbose)

    config = OutputConfig(outdir=Path(outdir), prefix=o, verbose=verbose)

    # Validate unimplemented flags
    _unimplemented = {"-wsnp": wsnp, "-gxe": gxe, "-vc": vc, "-mk": mk, "-mvlmm": mvlmm}
    for flag_name, flag_value in _unimplemented.items():
        if flag_value is not None:
            raise click.ClickException(f"Flag {flag_name} is not yet implemented")

    # Mode validation: exactly one of -gk or -lmm required
    if gk is not None and lmm is not None:
        raise click.UsageError("-gk and -lmm are mutually exclusive")
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

    # Parse -cat option
    cat_columns = None
    if cat is not None:
        try:
            cat_columns = [int(x) for x in cat.split()]
        except ValueError as e:
            raise click.UsageError(
                f"-cat must be space-separated integers, got '{cat}'"
            ) from e
        if not cat_columns:
            raise click.UsageError("-cat requires at least one column index")

    # Dispatch to handler
    if gk is not None:
        _run_gk(
            bfile=Path(bfile),
            mode=gk,
            config=config,
            maf=maf,
            miss=miss,
            check_memory=check_memory,
            loco=loco,
            write_eigen=eigen,
            phenotype_column=n,
            ksnps_file=_opt_path(ksnps),
        )
    else:
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
            phenotype_column=n,
            snps_file=_opt_path(snps),
            ksnps_file=_opt_path(ksnps),
            hwe_threshold=hwe,
            l_min=lmin,
            l_max=lmax,
            weight_file=_opt_path(widv),
            cat_columns=cat_columns,
            profile_dir=Path(profile_dir) if profile_dir else None,
        )


def _run_gk(
    *,
    bfile: Path,
    mode: int,
    config: OutputConfig,
    maf: float,
    miss: float,
    check_memory: bool,
    loco: bool,
    write_eigen: bool,
    phenotype_column: int,
    ksnps_file: Path | None,
) -> None:
    """Run kinship matrix computation."""
    if mode not in (1, 2):
        _cli_error(f"invalid kinship mode {mode}. Use -gk 1 or -gk 2.")

    if phenotype_column != 1:
        logger.info(
            f"Note: -n {phenotype_column} accepted but kinship computation "
            f"uses all samples"
        )

    start_time = time.perf_counter()

    # Ensure output directory exists
    config.ensure_outdir()

    # Construct command line for logging
    command_line = " ".join(sys.argv)

    # Validate bfile exists
    bed_path = Path(f"{bfile}.bed")
    if not bed_path.exists():
        _cli_error(f"PLINK file not found: {bed_path}")

    if write_eigen and loco:
        _cli_error("-eigen not supported with -loco mode")

    if mode == 2 and loco:
        _cli_error(
            "-gk 2 (standardized) is not supported with -loco. "
            "LOCO kinship uses centered mode (-gk 1). "
            "Use 'jamma -gk 2 -bfile X' without -loco for standardized kinship."
        )

    # Log startup banner (kinship uses all samples, so n_analyzed == n_total)
    from jamma.io.plink import get_plink_metadata as _get_meta

    _meta = _get_meta(bfile)
    PipelineRunner._log_banner(
        n_total=_meta["n_samples"],
        n_analyzed=_meta["n_samples"],
        n_snps=_meta["n_snps"],
    )

    # Resolve ksnps_file to indices if provided
    ksnps_indices = None
    if ksnps_file is not None:
        try:
            from jamma.io.plink import get_plink_metadata
            from jamma.io.snp_list import (
                read_snp_list_file,
                resolve_snp_list_to_indices,
            )

            meta = get_plink_metadata(bfile)
            ksnp_ids = read_snp_list_file(ksnps_file)
            ksnps_indices = resolve_snp_list_to_indices(ksnp_ids, meta["sid"])
        except (FileNotFoundError, ValueError) as e:
            logger.debug("CLI operation failed with traceback:", exc_info=True)
            _cli_error(str(e))
        click.echo(
            f"Kinship SNP list (-ksnps): {len(ksnps_indices)} of "
            f"{meta['n_snps']} SNPs selected"
        )

    if loco:
        # LOCO kinship mode: compute per-chromosome LOCO kinship matrices
        click.echo(f"Computing LOCO kinship matrices from {bfile}...")
        kinship_start = time.perf_counter()

        loco_iter = compute_loco_kinship_streaming(
            bfile,
            maf_threshold=maf,
            miss_threshold=miss,
            check_memory=check_memory,
            show_progress=True,
            ksnps_indices=ksnps_indices,
        )
        written_paths = write_loco_kinship_matrices(
            loco_iter,
            output_dir=config.outdir,
            prefix=config.prefix,
        )
        kinship_time = time.perf_counter() - kinship_start

        click.echo(
            f"Wrote {len(written_paths)} LOCO kinship matrices in {kinship_time:.2f}s"
        )
        for p in written_paths:
            click.echo(f"  {p}")

        # Write log file
        elapsed = time.perf_counter() - start_time
        params = {
            "kinship_mode": "loco",
            "n_chromosomes": len(written_paths),
            "maf_threshold": maf,
            "miss_threshold": miss,
        }
        timing = {"total": elapsed, "kinship": kinship_time}
        log_path = write_gemma_log(config, params, timing, command_line)
        click.echo(f"Log written to {log_path}")
        return

    # Standard kinship mode
    kinship_start = time.perf_counter()

    if mode == 1:
        # Centered kinship: use streaming to avoid loading full genotype matrix
        kinship_label = "centered"
        click.echo(f"Computing {kinship_label} kinship matrix (streaming)...")
        if maf > 0.0 or miss < 1.0:
            click.echo(f"Filtering: MAF >= {maf}, missing rate <= {miss}")

        K = compute_kinship_streaming(
            bfile,
            maf_threshold=maf,
            miss_threshold=miss,
            check_memory=check_memory,
            show_progress=True,
            ksnps_indices=ksnps_indices,
        )
    else:
        # Standardized kinship: requires full genotype matrix (no streaming variant)
        kinship_label = "standardized"
        click.echo(f"Loading PLINK data from {bfile}...")
        try:
            plink_data = load_plink_binary(bfile)
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.debug("Loading PLINK data failed with traceback:", exc_info=True)
            _cli_error(f"loading PLINK data: {e}")

        click.echo(f"Loaded {plink_data.n_samples} samples, {plink_data.n_snps} SNPs")
        click.echo(f"Computing {kinship_label} kinship matrix...")
        if maf > 0.0 or miss < 1.0:
            click.echo(f"Filtering: MAF >= {maf}, missing rate <= {miss}")

        genotypes = plink_data.genotypes
        if ksnps_indices is not None:
            genotypes = genotypes[:, ksnps_indices]
            click.echo(f"Using {genotypes.shape[1]} SNPs for kinship computation")

        K = compute_standardized_kinship(
            genotypes,
            maf_threshold=maf,
            miss_threshold=miss,
            check_memory=check_memory,
        )

    kinship_time = time.perf_counter() - kinship_start
    click.echo(f"{kinship_label.capitalize()} kinship computed in {kinship_time:.2f}s")

    # Write kinship matrix
    kinship_path = config.outdir / f"{config.prefix}.cXX.txt"
    write_kinship_matrix(K, kinship_path)
    click.echo(f"Kinship matrix written to {kinship_path}")

    n_samples = K.shape[0]

    # Eigendecompose and write if -eigen flag
    if write_eigen:
        from jamma.lmm.eigen import eigendecompose_kinship
        from jamma.lmm.eigen_io import write_eigen_files

        eigenvalues, eigenvectors = eigendecompose_kinship(K, check_memory=check_memory)
        del K  # K's buffer now contains eigenvectors; prevent accidental reuse
        d_path, u_path = write_eigen_files(
            eigenvalues,
            eigenvectors,
            config.outdir,
            config.prefix,
        )
        click.echo(f"Eigenvalues written to {d_path}")
        click.echo(f"Eigenvectors written to {u_path}")

    # Calculate timing
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    n_snps = _meta["n_snps"]
    params = {
        "n_samples": n_samples,
        "n_snps": n_snps,
        "kinship_mode": mode,
        "kinship_file": str(kinship_path),
        "maf_threshold": maf,
        "miss_threshold": miss,
    }
    timing = {"total": elapsed, "kinship": kinship_time}

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
    phenotype_column: int,
    snps_file: Path | None,
    ksnps_file: Path | None,
    hwe_threshold: float,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    weight_file: Path | None = None,
    cat_columns: list[int] | None = None,
    profile_dir: Path | None = None,
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

    # Build pipeline config
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
        phenotype_column=phenotype_column,
        snps_file=snps_file,
        ksnps_file=ksnps_file,
        hwe_threshold=hwe_threshold,
        l_min=l_min,
        l_max=l_max,
        weight_file=weight_file,
        cat_columns=cat_columns,
        profile_dir=profile_dir,
    )

    # Run pipeline, converting exceptions to CLI-friendly errors
    try:
        if check_memory:
            click.echo("Checking memory requirements...")
        result = PipelineRunner(pipeline_config).run()
    except (FileNotFoundError, ValueError, MemoryError) as e:
        logger.debug("Pipeline failed with traceback:", exc_info=True)
        _cli_error(str(e))

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
