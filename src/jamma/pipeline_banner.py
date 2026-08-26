"""GEMMA-style startup banners for a pipeline run.

Split out of ``pipeline.py`` because these are presentation, not
orchestration: both functions read their arguments, emit log lines, and
return nothing. Neither the ``-lmm`` path nor the ``-gk`` path is affected by
what they print, so keeping them here lets a reader skip them entirely when
following the compute flow.
"""

from __future__ import annotations

import os

from loguru import logger

from jamma.core.backend import format_pipeline_banner
from jamma.lmm.runner import ExecutionPlan

__all__ = ["log_dataset_banner", "log_pipeline_banner"]


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
    """Emit a consolidated one-line pipeline configuration banner.

    Gathers runner type, BLAS backend, C extension status, and
    thread count into a single log line. The banner shows "pending"
    for the eigen driver; the actual driver is logged separately by
    eigendecompose_kinship once the matrix size is known.

    This function is purely diagnostic — failures are caught and logged
    as warnings to avoid aborting the GWAS pipeline.

    Args:
        plan: ExecutionPlan with backend and mode already decided.
    """
    try:
        import jamma.jlinalg as jlinalg
        from jamma.core.threading import (
            get_blas_backend,
            get_blas_thread_count,
            get_c_extension_thread_count,
            get_physical_core_count,
            is_blas_controllable,
        )
        from jamma.lmm import compute_numpy

        c_ext = compute_numpy._accel is not None
        jlinalg_backend = jlinalg.blas_backend
        c_has_openmp = bool(compute_numpy._C_HAS_OPENMP)
        runner = plan.runner_name

        blas = get_blas_backend()

        # One parse of JAMMA_BLAS_THREADS: get_blas_thread_count owns it.
        if os.environ.get("JAMMA_BLAS_THREADS") or is_blas_controllable():
            threads = get_blas_thread_count()
        else:
            # Accelerate or no BLAS — use halved core count
            # (same fallback used by the NumPy LMM chunk runner).
            threads = max(1, get_physical_core_count() // 2)

        # A single-threaded _lmm_accel build should not be logged as a
        # multi-threaded compute kernel.
        if c_ext:
            threads = min(
                threads,
                get_c_extension_thread_count(
                    c_accel_available=c_ext,
                    c_has_openmp=c_has_openmp,
                ),
            )

        banner = format_pipeline_banner(
            runner=runner,
            blas=blas,
            eigen_driver="pending",
            c_ext=c_ext,
            threads=threads,
            jlinalg_backend=jlinalg_backend,
        )
        logger.info(banner)
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        logger.warning(f"Could not build pipeline banner: {exc}")
