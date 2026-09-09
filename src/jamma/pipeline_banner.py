"""GEMMA-style startup banners for a pipeline run.

Split out of ``pipeline.py`` because these are presentation, not
orchestration: both functions read their arguments, emit log lines, and
return nothing. Neither the ``-lmm`` path nor the ``-gk`` path is affected by
what they print, so keeping them here lets a reader skip them entirely when
following the compute flow.
"""

from __future__ import annotations

from loguru import logger

from jamma.core.threading import blas_display_name
from jamma.lmm.association_plan import ExecutionPlan

__all__ = ["log_dataset_banner", "log_pipeline_banner"]


def format_pipeline_banner(
    runner: str,
    blas: str,
    eigen_driver: str,
    c_ext: bool,
    jlinalg_backend: str | None = None,
) -> str:
    """Build a single-line pipeline startup banner.

    Thread counts are reported later by each resolved execution plan.

    Args:
        runner: Runner name (e.g. "numpy-batch", "numpy-streaming").
        blas: BLAS backend identifier (e.g. "mkl", "openblas",
            "accelerate").
        eigen_driver: Eigen driver name (e.g. "DSYEVD", "DSYEVR").
        c_ext: Whether the C extension is usable.
        jlinalg_backend: jlinalg's ``blas_backend`` (e.g. "MKL-ILP64",
            "numpy-fallback"). Omitted from the banner when None, since
            jlinalg can report "numpy-fallback" even with its C extension
            loaded (``JLINALG_NO_VENDOR_DGEMM``), a state the ``c_ext`` flag
            alone cannot show.

    Returns:
        Formatted banner string.

    Example:
        >>> format_pipeline_banner("numpy-batch", "mkl", "DSYEVD", True)
        'Pipeline: numpy-batch | MKL | DSYEVD | C-ext'
        >>> format_pipeline_banner(
        ...     "numpy-batch", "mkl", "DSYEVD", True, jlinalg_backend="MKL-ILP64"
        ... )
        'Pipeline: numpy-batch | MKL | DSYEVD | C-ext | jlinalg: MKL-ILP64'
    """
    blas_display = blas_display_name(blas)
    c_ext_str = "C-ext" if c_ext else "no C-ext"
    banner = f"Pipeline: {runner} | {blas_display} | {eigen_driver} | {c_ext_str}"
    if jlinalg_backend is not None:
        banner += f" | jlinalg: {jlinalg_backend}"
    return banner


def log_dataset_banner(
    n_total: int,
    n_analyzed: int,
    n_snps: int,
    n_covariates: int = 1,
    n_phenotypes: int = 1,
) -> None:
    """Log GEMMA-style startup banner with dataset summary.

    Prints version, release date, and dataset dimensions to match
    GEMMA's startup output format for user familiarity.

    Args:
        n_total: Total number of individuals in the PLINK file.
        n_analyzed: Number of individuals after phenotype/covariate filtering.
        n_snps: Total number of SNPs in the dataset.
        n_covariates: Number of covariate columns (1 = intercept-only).
        n_phenotypes: Number of phenotype columns being analyzed.
    """
    import jamma

    logger.info(f"JAMMA v{jamma.__version__} ({jamma.__release_date__})")
    logger.info("Reading Files ...")
    logger.info(f"## number of total individuals = {n_total:,}")
    logger.info(f"## number of analyzed individuals = {n_analyzed:,}")
    logger.info(f"## number of covariates = {n_covariates}")
    logger.info(f"## number of phenotypes = {n_phenotypes}")
    logger.info(f"## number of total SNPs/var = {n_snps:,}")


def log_pipeline_banner(plan: ExecutionPlan) -> None:
    """Emit the selected backend and available native implementation.

    The eigen driver and execution thread counts are reported when resolved.

    This function is purely diagnostic — failures are caught and logged
    as warnings to avoid aborting the GWAS pipeline.

    Args:
        plan: ExecutionPlan with backend and mode already decided.
    """
    try:
        import jamma.jlinalg as jlinalg
        from jamma.core.threading import get_blas_backend
        from jamma.lmm import accel

        blas = get_blas_backend()
        if blas == "unknown":
            blas = jlinalg.blas_backend
        banner = format_pipeline_banner(
            runner=plan.runner_name,
            blas=blas,
            eigen_driver="pending",
            c_ext=accel.available(),
            jlinalg_backend=jlinalg.blas_backend,
        )
        logger.info(banner)
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        logger.warning(f"Could not build pipeline banner: {exc}")
