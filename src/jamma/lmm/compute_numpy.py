"""NumPy mode dispatch for LMM chunk computation.

Dispatches to C extension (_lmm_accel) for Wald (batch/workspace),
Score (batch), and LRT (batch) when available and n_cvt=1; falls back
to NumPy Python path for n_cvt>1 or when C functions are unavailable.
Also exports split-workspace and general-workspace APIs for direct use
by runners. No JAX imports.

The caller is responsible for:
- Computing Uab_batch via batch_compute_uab_numpy (before calling this)
- There is no async dispatch in the NumPy backend — results are immediately
  available after the call returns.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    batch_calc_wald_stats_from_pab_numpy,
    batch_compute_iab_numpy,
    compute_iab_invariant_scalars_ncvt1,
    golden_section_optimize_lambda_mle_numpy,
    golden_section_optimize_lambda_numpy,
    golden_section_optimize_lambda_split_ncvt1_numpy,
)

_EXPECTED_ABI_VERSION = 6  # Must match ABI_VERSION in _lmm_accel.c


def _try_import_accel() -> tuple[
    bool,
    bool,
    bool,
    bool,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
]:
    """Attempt to import the C extension and validate ABI version.

    Returns:
        (accel_available, split_available, general_available, has_openmp,
         compute_batch_c, compute_batch_split_c,
         create_workspace_split_c, compute_lmm_chunk_split_c,
         create_workspace_general_c, compute_lmm_chunk_general_c,
         compute_score_batch_c, compute_lrt_batch_c)
    """
    _none12 = (
        False,
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    try:
        from jamma.lmm._lmm_accel import ABI_VERSION as abi
        from jamma.lmm._lmm_accel import HAS_OPENMP as has_omp
        from jamma.lmm._lmm_accel import compute_lmm_batch_c as batch_c
        from jamma.lmm._lmm_accel import (
            compute_lmm_batch_split_c as batch_split_c,
        )
        from jamma.lmm._lmm_accel import (
            compute_lmm_chunk_split_c as ws_chunk,
        )
        from jamma.lmm._lmm_accel import (
            create_workspace_split_c as ws_create,
        )
    except ImportError as e:
        from loguru import logger

        logger.debug(f"C extension import failed: {e}")
        return _none12
    except AttributeError as e:
        from loguru import logger

        logger.warning(
            f"C extension loaded but missing expected attribute: {e}. "
            "Stale .so may need recompilation."
        )
        return _none12

    if abi != _EXPECTED_ABI_VERSION:
        from loguru import logger

        logger.warning(
            "C extension ABI mismatch: "
            f"compiled={abi}, expected={_EXPECTED_ABI_VERSION}. "
            "Stale .so needs recompilation."
        )
        return _none12

    # General n_cvt support — expected since ABI v4
    try:
        from jamma.lmm._lmm_accel import (
            compute_lmm_chunk_general_c as ws_gen_chunk,
        )
        from jamma.lmm._lmm_accel import (
            create_workspace_general_c as ws_gen_create,
        )

        general_available = True
    except AttributeError:
        from loguru import logger

        logger.warning(
            f"C extension ABI v{abi} but "
            "create_workspace_general_c / compute_lmm_chunk_general_c "
            "not found. Extension may be partially compiled. "
            "Falling back to Python path for n_cvt > 1."
        )
        ws_gen_create = None
        ws_gen_chunk = None
        general_available = False

    # Score and LRT batch support — expected in ABI v5+
    # Import independently so a partial build (one present, one missing)
    # doesn't disable both.
    try:
        from jamma.lmm._lmm_accel import (
            compute_score_batch_c as score_batch_c,
        )
    except AttributeError:
        from loguru import logger

        logger.warning(
            "C extension missing compute_score_batch_c. Score will use Python path."
        )
        score_batch_c = None

    try:
        from jamma.lmm._lmm_accel import (
            compute_lrt_batch_c as lrt_batch_c,
        )
    except AttributeError:
        from loguru import logger

        logger.warning(
            "C extension missing compute_lrt_batch_c. LRT will use Python path."
        )
        lrt_batch_c = None

    return (
        True,
        True,
        general_available,
        has_omp,
        batch_c,
        batch_split_c,
        ws_create,
        ws_chunk,
        ws_gen_create,
        ws_gen_chunk,
        score_batch_c,
        lrt_batch_c,
    )


def _auto_recompile() -> bool:
    """Auto-recompile the LMM C extension and reimport into sys.modules."""
    from jamma.lmm._compile_utils import auto_recompile_c_extension

    return auto_recompile_c_extension(
        module_name="_lmm_accel",
        compiler_module="jamma.lmm._compile_accel",
        sys_module_key="jamma.lmm._lmm_accel",
        label="LMM",
    )


# First attempt
(
    _C_ACCEL_AVAILABLE,
    _C_SPLIT_AVAILABLE,
    _C_GENERAL_AVAILABLE,
    _C_HAS_OPENMP,
    _compute_lmm_batch_c,
    _compute_lmm_batch_split_c,
    _create_workspace_split_c,
    _compute_lmm_chunk_split_c,
    _create_workspace_general_c,
    _compute_lmm_chunk_general_c,
    _compute_score_batch_c,
    _compute_lrt_batch_c,
) = _try_import_accel()

if not _C_ACCEL_AVAILABLE:
    # Auto-recompile and retry once
    if _auto_recompile():
        (
            _C_ACCEL_AVAILABLE,
            _C_SPLIT_AVAILABLE,
            _C_GENERAL_AVAILABLE,
            _C_HAS_OPENMP,
            _compute_lmm_batch_c,
            _compute_lmm_batch_split_c,
            _create_workspace_split_c,
            _compute_lmm_chunk_split_c,
            _create_workspace_general_c,
            _compute_lmm_chunk_general_c,
            _compute_score_batch_c,
            _compute_lrt_batch_c,
        ) = _try_import_accel()

    if not _C_ACCEL_AVAILABLE:
        from loguru import logger as _logger

        _logger.warning(
            "C extension _lmm_accel not available — using pure-Python path "
            "(LMM may be slower without C extension; magnitude depends on "
            "dataset size and core count). To compile, run: "
            "python -m jamma.lmm._compile_accel"
        )
        del _logger
        _C_HAS_OPENMP = False
        _C_GENERAL_AVAILABLE = False


class WaldResult(TypedDict):
    """Result dict from REML Wald pipeline (both C and Python paths)."""

    lambdas: np.ndarray
    logls: np.ndarray
    betas: np.ndarray
    ses: np.ndarray
    pwalds: np.ndarray


LmmMode = Literal[1, 2, 3, 4]


def _compute_wald_c(
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    Iab_batch: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald via C extension (n_cvt=1 only).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, 6).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance).
        Iab_batch: Pre-computed identity-weighted Pab (n_snps, 3, 6).
        n_threads: OpenMP thread count for the C extension.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    return _compute_lmm_batch_c(
        eigenvalues,
        Uab_batch,
        Iab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
    )


def _compute_wald_split_c(
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    Iab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald via split-Uab C extension (n_cvt=1 only).

    Expects SoA layout arrays (no per-call transpose). Callers must pass
    arrays already in SoA layout from batch_compute_uab_split_soa_numpy or
    batch_compute_uab_varying_soa_numpy + compute_uab_invariant_soa.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_varying_soa: SNP-varying columns (n_snps, 3, n_samples) [wx, xx, xy].
        uab_invariant_soa: SNP-invariant columns (3, n_samples) [ww, wy, yy].
        Iab_batch: Pre-computed identity-weighted Pab (n_snps, 3, 6).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    return _compute_lmm_batch_split_c(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        Iab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
    )


def create_lmm_workspace(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> object:
    """Create a persistent C workspace for the split-Uab REML pipeline.

    The workspace holds precomputed lambda_grid, hi_eval_grid, logdet_h_grid,
    grid_inv, and invariant Iab column sums (iab_s_ww etc.). It is reused
    across all chunks — eliminating per-chunk C malloc and grid precomputation.

    Returns a PyCapsule that is freed automatically when it goes out of scope.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (3, n_samples) — rows [ww, wy, yy].
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        PyCapsule wrapping lmm_workspace_t (opaque; pass to compute_wald_split_c_ws).
    """
    return _create_workspace_split_c(
        eigenvalues,
        uab_invariant_soa,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
    )


def compute_wald_split_c_ws(
    workspace: object,
    uab_varying_soa: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald for one chunk using a pre-built workspace.

    Uses precomputed grids from create_lmm_workspace — no per-chunk malloc,
    no eigenvalue reprocessing, no invariant Iab computation. Iab logdet
    is computed inside C from the workspace's precomputed iab_s_ww.

    Args:
        workspace: PyCapsule from create_lmm_workspace.
        uab_varying_soa: SNP-varying Uab (n_snps, 3, n_samples) — SoA layout.
        n_threads: OpenMP thread count.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    return _compute_lmm_chunk_split_c(workspace, uab_varying_soa, n_threads)


def create_lmm_workspace_general(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    n_cvt: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> object:
    """Create a persistent C workspace for the general n_cvt REML pipeline.

    Builds the Pab recursion table via build_pab_table_for_c(), then passes
    all flat arrays to the C extension's create_workspace_general_c().

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (n_inv, n_samples) — SoA layout.
        n_samples: Number of samples.
        n_cvt: Number of covariates.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        PyCapsule wrapping lmm_workspace_general_t.
    """
    if _create_workspace_general_c is None:
        raise RuntimeError(
            "General n_cvt C workspace requires the _lmm_accel C extension "
            "with ABI version 4+. Recompile: python -m jamma.lmm._compile_accel"
        )

    from jamma.lmm.likelihood import build_pab_table_for_c

    table = build_pab_table_for_c(n_cvt)

    return _create_workspace_general_c(
        eigenvalues,
        uab_invariant_soa,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
        n_cvt,
        table["invariant_indices"],
        table["varying_indices"],
        table["logdet_diag_rows"],
        table["logdet_diag_cols"],
        table["level_offsets"],
        table["level_counts"],
        table["entries"],
        table["idx_xx"],
        table["idx_xy"],
        table["idx_yy"],
    )


def compute_wald_general_c_ws(
    workspace: object,
    uab_varying_soa: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald for one chunk using a general n_cvt workspace.

    Args:
        workspace: PyCapsule from create_lmm_workspace_general.
        uab_varying_soa: SNP-varying Uab (n_snps, n_var, n_samples) — SoA.
        n_threads: OpenMP thread count.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    if _compute_lmm_chunk_general_c is None:
        raise RuntimeError(
            "General n_cvt C chunk compute requires the _lmm_accel C extension "
            "with ABI version 4+. Recompile: python -m jamma.lmm._compile_accel"
        )
    return _compute_lmm_chunk_general_c(workspace, uab_varying_soa, n_threads)


def _compute_score_c(
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Hi_eval_null: np.ndarray,
    n_samples: int,
    n_threads: int,
) -> dict[str, np.ndarray]:
    """Compute Score test via C extension (n_cvt=1 only).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, 6).
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        n_samples: Number of samples.
        n_threads: OpenMP thread count.

    Returns:
        Dict with keys: betas, ses, p_scores.
    """
    return _compute_score_batch_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_threads,
    )


def _compute_lrt_c(
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict[str, np.ndarray]:
    """Compute LRT via C extension (n_cvt=1 only).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, 6).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        logl_H0: Null model MLE log-likelihood (scalar).
        n_threads: OpenMP thread count.

    Returns:
        Dict with keys: lambdas_mle, p_lrts.
    """
    return _compute_lrt_batch_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        n_threads,
    )


def _compute_wald_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    Iab_batch: np.ndarray | None = None,
    n_threads: int = 1,
) -> WaldResult:
    """Compute REML-optimized Wald test statistics.

    Dispatches to C extension: n_cvt=1 uses the batch path (compute_lmm_batch_c),
    n_cvt>1 uses the general workspace path. Falls back to Python split path
    (n_cvt=1) or generic Python path (n_cvt>1) when C extension is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance;
            C extension requires >= 1). Runner-level code enforces the minimum.
        Iab_batch: Pre-computed identity-weighted Pab. If None, computed internally.
        n_threads: OpenMP thread count passed to C extension (ignored on Python path).

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds.
    """

    if _C_ACCEL_AVAILABLE and n_cvt == 1:
        if Iab_batch is None:
            Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        return _compute_wald_c(
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            Iab_batch,
            n_threads,
        )

    if _C_GENERAL_AVAILABLE and 1 < n_cvt <= 20:
        # Use C extension for general n_cvt via split-Uab workspace
        from jamma.lmm.likelihood import classify_uab_columns

        inv_indices, var_indices = classify_uab_columns(n_cvt)

        # Build invariant SoA from Uab_batch (shared across all SNPs, use SNP 0)
        # Note: a[0, :, list_idx] returns (n_inv, n_samples) due to numpy advanced
        # indexing rules (integer + list separated by slice -> grouped at front).
        inv_list = list(inv_indices)
        if __debug__ and Uab_batch.shape[0] > 1:
            # Verify columns classified as invariant are actually constant across SNPs
            inv_sample = Uab_batch[:, :, inv_list]
            if not np.allclose(inv_sample, inv_sample[0:1], rtol=1e-12, atol=0):
                raise RuntimeError(
                    f"classify_uab_columns({n_cvt}) classified varying columns as "
                    "invariant — internal error, please report with your n_cvt value"
                )
        uab_invariant_soa = np.ascontiguousarray(
            Uab_batch[0, :, inv_list]  # (n_inv, n_samples)
        )
        # Build varying SoA from Uab_batch
        uab_varying_soa = np.ascontiguousarray(
            Uab_batch[:, :, list(var_indices)].transpose(
                0, 2, 1
            )  # (n_snps, n_var, n_samples)
        )

        # Create per-call workspace and compute
        ws = create_lmm_workspace_general(
            eigenvalues,
            uab_invariant_soa,
            n_samples,
            n_cvt,
            l_min,
            l_max,
            n_grid,
            n_refine,
            n_threads,
        )
        return compute_wald_general_c_ws(ws, uab_varying_soa, n_threads)

    if n_cvt == 1:
        # Python split path for n_cvt=1: separate invariant (ww, wy, yy)
        # and varying (wx, xx, xy) Uab columns to reduce per-SNP computation.
        # Column layout: 0=ww, 1=wx, 2=wy, 3=xx, 4=xy, 5=yy.
        # Invariant columns (ww, wy, yy) are identical across SNPs — use SNP 0.
        uab_varying_soa = np.stack(
            [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
        )  # (n_snps, 3, n_samples): rows [wx, xx, xy]
        uab_invariant_soa = np.stack(
            [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
        )  # (3, n_samples): rows [ww, wy, yy]

        iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
            uab_invariant_soa
        )
        lambdas, logls, Pab_final = golden_section_optimize_lambda_split_ncvt1_numpy(
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            iab_s_ww,
            iab_s_wy,
            iab_s_yy,
            iab_logdet,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_iter=n_refine,
            return_pab=True,
        )
    else:
        # Generic Python path for n_cvt > 1
        if Iab_batch is None:
            Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        lambdas, logls, Pab_final = golden_section_optimize_lambda_numpy(
            n_cvt,
            eigenvalues,
            Uab_batch,
            Iab_batch,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_iter=n_refine,
            return_pab=True,
        )

    betas, ses, pwalds = batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_final, n_samples
    )
    return {
        "lambdas": lambdas,
        "logls": logls,
        "betas": betas,
        "ses": ses,
        "pwalds": pwalds,
    }


def _compute_lrt_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int = 1,
) -> dict[str, np.ndarray]:
    """Compute MLE-optimized LRT statistics.

    Dispatches to C extension when available and n_cvt=1; otherwise falls
    back to Python golden section optimizer.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance).
        logl_H0: Null model MLE log-likelihood (scalar).
        n_threads: OpenMP thread count passed to C extension (ignored on Python path).

    Returns:
        Dict with keys: lambdas_mle, p_lrts.
    """
    if _compute_lrt_batch_c is not None and n_cvt == 1:
        return _compute_lrt_c(
            eigenvalues,
            Uab_batch,
            len(eigenvalues),
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    from loguru import logger

    if _compute_lrt_batch_c is None:
        logger.debug("LRT using Python path (C extension unavailable)")
    else:
        logger.debug("LRT using Python path (n_cvt={} > 1)", n_cvt)

    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts = _batch_lrt_pvalues_numpy(logls_mle, logl_H0)
    return {"lambdas_mle": lambdas_mle, "p_lrts": p_lrts}


def _compute_score_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Hi_eval_null: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    n_threads: int = 1,
) -> dict[str, np.ndarray]:
    """Compute Score test statistics (no optimization needed).

    Dispatches to C extension when available and n_cvt=1; otherwise falls
    back to Python batch Score computation.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,). Used by C path for validation.
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        n_threads: OpenMP thread count passed to C extension (ignored on Python path).

    Returns:
        Dict with keys: betas, ses, p_scores.
    """
    if _compute_score_batch_c is not None and n_cvt == 1:
        return _compute_score_c(
            eigenvalues,
            Uab_batch,
            Hi_eval_null,
            n_samples,
            n_threads,
        )

    from loguru import logger

    if _compute_score_batch_c is None:
        logger.debug("Score using Python path (C extension unavailable)")
    else:
        logger.debug("Score using Python path (n_cvt={} > 1)", n_cvt)

    betas, ses, p_scores = batch_calc_score_stats_numpy(
        n_cvt, Hi_eval_null, Uab_batch, n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _compute_lmm_chunk_numpy(
    lmm_mode: LmmMode,
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    *,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    Hi_eval_null: np.ndarray | None = None,
    logl_H0: float | None = None,
    n_threads: int = 1,
) -> dict[str, np.ndarray | None]:
    """Compute LMM statistics for a chunk of SNPs (NumPy backend).

    Mirrors _compute_lmm_chunk in compute.py but uses NumPy batch functions
    instead of JAX. No async dispatch — results are immediately available.

    Args:
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (minimum 20 enforced).
        Hi_eval_null: Pre-computed 1/(lambda_null*eval+1) for Score test.
        logl_H0: Null model MLE log-likelihood for LRT.
        n_threads: OpenMP thread count passed to C extension (Wald, LRT, Score).

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds,
        lambdas_mle, p_lrts, p_scores. Keys not relevant to the
        mode are set to None.
    """
    n_refine = max(n_refine, 20)

    if lmm_mode in (2, 4) and logl_H0 is None:
        raise ValueError("logl_H0 is required for LRT (mode 2) and All (mode 4)")
    if lmm_mode in (3, 4) and Hi_eval_null is None:
        raise ValueError("Hi_eval_null is required for Score (mode 3) and All (mode 4)")

    result: dict[str, np.ndarray | None] = {
        "lambdas": None,
        "logls": None,
        "betas": None,
        "ses": None,
        "pwalds": None,
        "lambdas_mle": None,
        "p_lrts": None,
        "p_scores": None,
    }

    if lmm_mode == 1:
        result.update(
            _compute_wald_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                n_threads=n_threads,
            )
        )

    elif lmm_mode == 2:
        result.update(
            _compute_lrt_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                logl_H0,
                n_threads=n_threads,
            )
        )

    elif lmm_mode == 3:
        result.update(
            _compute_score_numpy(
                n_cvt,
                eigenvalues,
                Hi_eval_null,
                Uab_batch,
                n_samples,
                n_threads=n_threads,
            )
        )

    elif lmm_mode == 4:
        # Compose all three tests; only take p_scores from Score —
        # Wald provides REML-optimized beta/SE below
        score_result = _compute_score_numpy(
            n_cvt,
            eigenvalues,
            Hi_eval_null,
            Uab_batch,
            n_samples,
            n_threads=n_threads,
        )
        result["p_scores"] = score_result["p_scores"]
        result.update(
            _compute_lrt_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                logl_H0,
                n_threads=n_threads,
            )
        )
        # Pre-compute Iab once for Wald (lambda-independent)
        Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        result.update(
            _compute_wald_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                Iab_batch=Iab_batch,
                n_threads=n_threads,
            )
        )

    else:
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    return result
