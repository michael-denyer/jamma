"""Wall clock time estimates for GWAS pipeline phases.

Polynomial models fitted to v4.2.0 benchmarks on Azure E96ds_v6
(48 physical cores, MKL ILP64, Intel Xeon 8573C). Five data points:
5k, 20k, 50k, 75k, 125k samples at ~95k SNPs.

Models use n_k = n_samples / 1000 for numerical stability:
  - Kinship:    a*n_k² + b*n_k  (× m/m_ref × (cores_ref/cores)^0.7)
  - Eigendecomp: c * n_k^alpha  (× (cores_ref/cores)^0.7)  [power law]
  - LMM:        a*n_k² + b*n_k  (× m/m_ref × (cores_ref/cores)^0.7)

Core scaling uses exponent 0.7 (sub-linear) because BLAS operations are
memory-bandwidth-bound at large matrix sizes — doubling cores does not
halve runtime. Empirically, 48→32 core scaling is ~1.3x not 1.5x.

Kinship and LMM coefficients fitted via weighted NNLS (weight=1/actual)
to equalize relative error across scales. Eigendecomp uses a power law
(exponent ~2.72) because polynomials fit poorly — cache regime
transitions cause non-integer effective scaling.

These are minimum estimates (prefixed with >=). Max relative error
is ~14% for kinship, ~13% for eigendecomp, ~8% for LMM across
the 5k–125k calibration range.

Benchmark data (E96ds_v6, 48 cores, v4.2.0):
  n=5k:   kinship=12.1s,  eigen=0.91s,    LMM=11.5s
  n=20k:  kinship=73.5s,  eigen=46.1s,    LMM=56.6s
  n=50k:  kinship=298.4s, eigen=480.6s,   LMM=216.8s
  n=75k:  kinship=510.0s, eigen=1312.8s,  LMM=418.3s
  n=125k: kinship=1591s,  eigen=6427s,    LMM=882s
"""

from __future__ import annotations

import functools

from jamma.core.threading import get_physical_core_count

_REF_CORES = 48
_REF_SNPS = 91_586
_CORE_SCALING_EXP = 0.7  # Sub-linear: BLAS is memory-BW-bound at large N

# BLAS backend context for time estimates.
# Estimates are calibrated to MKL ILP64. Without vendor ILP64 BLAS,
# actual runtimes can be significantly longer — warn users accordingly.


@functools.cache
def _get_blas_context() -> tuple[str, bool]:
    """Return (blas_backend, is_ilp64) from jlinalg, with safe fallback.

    Returns:
        Tuple of (backend_name, is_ilp64). Falls back to
        ("unknown", False) if jlinalg cannot be imported.
    """
    try:
        from jamma import jlinalg

        return str(jlinalg.blas_backend), bool(jlinalg.blas_is_ilp64)
    except (ImportError, AttributeError) as exc:
        import logging

        logging.getLogger(__name__).debug(
            "Could not determine BLAS backend from jlinalg: %s", exc
        )
        return "unknown", False


def _blas_is_mkl() -> bool:
    """True if the active BLAS backend is MKL."""
    backend, _ = _get_blas_context()
    return "MKL" in backend.upper()


def _blas_caveat() -> str:
    """Return a caveat suffix for time estimates when not on MKL.

    Empty string when on MKL (estimates are calibrated). Otherwise a
    short warning that the estimate may understate actual runtime.
    """
    backend, _ = _get_blas_context()
    if "MKL" in backend.upper():
        return ""
    if backend in ("numpy-fallback", "unknown"):
        return " [estimates calibrated to MKL — no vendor BLAS detected, expect slower]"
    # Some other vendor BLAS (OpenBLAS, Accelerate, BLIS)
    return f" [estimates calibrated to MKL — {backend} may differ]"


def get_blas_estimate_context() -> tuple[str, bool, bool]:
    """Return BLAS context relevant to estimate accuracy.

    Returns:
        Tuple of (backend_name, is_ilp64, estimates_calibrated) where
        estimates_calibrated is True when on MKL (the reference backend).
    """
    backend, is_ilp64 = _get_blas_context()
    return backend, is_ilp64, _blas_is_mkl()


# Kinship: a*n_k^2 + b*n_k (SNP-normalized)
# Weighted NNLS fit (weight=1/actual), no constant term.
# Max error: +13.6% at 75k, most points within 3%.
_KINSHIP_A = 0.072956  # n_k^2 coefficient (DGEMM scaling)
_KINSHIP_B = 1.975440  # n_k coefficient (SNP stats + I/O scaling)

# Eigendecomp: c * n_k^alpha (power law)
# Log-linear OLS fit. Max error: +12.8% at 75k, -11.3% at 20k.
# Power law fits better than any polynomial because BLAS efficiency
# varies with matrix size (cache-bound at small n, BW-bound at large n),
# giving an effective exponent of ~2.72 instead of the theoretical 3.0.
_EIGEN_COEFF = 0.012007  # coefficient
_EIGEN_ALPHA = 2.7152  # exponent

# LMM: a*n_k^2 + b*n_k (SNP-normalized)
# Weighted NNLS fit (weight=1/actual), no constant term.
# Max error: +8.4% at 20k, most points within 4%.
# The n_k^2 term captures UT@G rotation (DGEMM) + eigen load overhead.
# The n_k term captures Wald test compute (C extension) + genotype I/O.
_LMM_A = 0.042075  # n_k^2 coefficient
_LMM_B = 1.984495  # n_k coefficient


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds) // 60
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    remaining_min = minutes % 60
    if remaining_min == 0:
        return f"{hours}h"
    return f"{hours}h {remaining_min}m"


def _resolve_cores(n_cores: int | None) -> int:
    if n_cores is None:
        return get_physical_core_count()
    return n_cores


def estimate_kinship_seconds(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> float:
    """Estimate kinship computation wall time in seconds.

    Polynomial model: a*n_k² + b*n_k, scaled by SNP ratio and core ratio.
    """
    n_cores = _resolve_cores(n_cores)
    n_k = n_samples / 1000
    snp_ratio = n_snps / _REF_SNPS
    core_ratio = (_REF_CORES / n_cores) ** _CORE_SCALING_EXP
    return (_KINSHIP_A * n_k**2 + _KINSHIP_B * n_k) * snp_ratio * core_ratio


def estimate_kinship_time(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> str:
    """Estimate kinship computation wall time as a human-readable string.

    Estimates are calibrated to MKL ILP64 on 48-core Xeon. A caveat is
    appended when the active BLAS backend differs.

    Args:
        n_samples: Number of samples.
        n_snps: Number of SNPs.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Minimum estimate string like ">=5 min", with BLAS caveat if applicable.
    """
    duration = _format_duration(estimate_kinship_seconds(n_samples, n_snps, n_cores))
    return f">={duration}{_blas_caveat()}"


def estimate_eigendecomp_seconds(
    n_samples: int,
    n_cores: int | None = None,
) -> float:
    """Estimate eigendecomposition wall time in seconds.

    Power law model: c * n_k^alpha, scaled by core ratio.
    """
    n_cores = _resolve_cores(n_cores)
    n_k = n_samples / 1000
    core_ratio = (_REF_CORES / n_cores) ** _CORE_SCALING_EXP
    return _EIGEN_COEFF * n_k**_EIGEN_ALPHA * core_ratio


def estimate_eigendecomp_time(
    n_samples: int,
    n_cores: int | None = None,
    *,
    use_dsyevr: bool = False,
) -> str:
    """Estimate eigendecomposition wall time as a human-readable string.

    Estimates are calibrated to MKL ILP64 on 48-core Xeon. A caveat is
    appended when the active BLAS backend differs.

    Args:
        n_samples: Number of samples.
        n_cores: Physical core count. None auto-detects.
        use_dsyevr: Accepted for API compatibility. No multiplier applied.

    Returns:
        Minimum estimate string like ">=1h 47m", with BLAS caveat if applicable.
    """
    duration = _format_duration(estimate_eigendecomp_seconds(n_samples, n_cores))
    return f">={duration}{_blas_caveat()}"


def estimate_lmm_seconds(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> float:
    """Estimate LMM association wall time in seconds.

    Polynomial model: a*n_k² + b*n_k, scaled by SNP ratio and core ratio.
    """
    n_cores = _resolve_cores(n_cores)
    n_k = n_samples / 1000
    snp_ratio = n_snps / _REF_SNPS
    core_ratio = (_REF_CORES / n_cores) ** _CORE_SCALING_EXP
    return (_LMM_A * n_k**2 + _LMM_B * n_k) * snp_ratio * core_ratio


def estimate_lmm_time(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> str:
    """Estimate LMM association wall time as a human-readable string.

    Estimates are calibrated to MKL ILP64 on 48-core Xeon. A caveat is
    appended when the active BLAS backend differs.

    Args:
        n_samples: Number of samples.
        n_snps: Number of filtered SNPs.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Minimum estimate string like ">=15 min", with BLAS caveat if applicable.
    """
    duration = _format_duration(estimate_lmm_seconds(n_samples, n_snps, n_cores))
    return f">={duration}{_blas_caveat()}"
