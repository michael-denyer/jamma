"""Wall clock time estimates for GWAS pipeline phases.

Polynomial models fitted to v4.2.0 benchmarks on Azure E96ds_v6
(48 physical cores, MKL ILP64, Intel Xeon 8573C). Five data points:
5k, 20k, 50k, 75k, 125k samples at ~95k SNPs.

Models use n_k = n_samples / 1000 for numerical stability:
  - Kinship:    a*n_k² + b*n_k  (× m/m_ref × cores_ref/cores)
  - Eigendecomp: c * n_k^alpha  (× cores_ref/cores)  [power law]
  - LMM:        a*n_k² + b*n_k  (× m/m_ref × cores_ref/cores)

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

from jamma.core.threading import get_physical_core_count

_REF_CORES = 48
_REF_SNPS = 91_586

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


def estimate_kinship_time(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> str:
    """Estimate kinship computation wall time.

    Polynomial model: a*n_k² + b*n_k, scaled by SNP ratio and core ratio.
    The quadratic term captures DGEMM accumulation, the linear term
    captures SNP statistics and I/O overhead.

    This is a minimum estimate — memory pressure and I/O contention
    are not accounted for.

    Args:
        n_samples: Number of samples.
        n_snps: Number of SNPs.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Human-readable minimum estimate string like ">=5 min".
    """
    if n_cores is None:
        n_cores = get_physical_core_count()

    n_k = n_samples / 1000
    snp_ratio = n_snps / _REF_SNPS
    core_ratio = _REF_CORES / n_cores

    est = (_KINSHIP_A * n_k**2 + _KINSHIP_B * n_k) * snp_ratio * core_ratio
    return f">={_format_duration(est)}"


def estimate_eigendecomp_time(
    n_samples: int,
    n_cores: int | None = None,
    *,
    use_dsyevr: bool = False,
) -> str:
    """Estimate eigendecomposition wall time.

    Power law model: c * n_k^alpha, scaled by core ratio. The exponent
    (~2.72) is sub-cubic because BLAS efficiency varies with matrix size:
    small matrices are cache-bound (faster per-FLOP), large matrices are
    memory-bandwidth-bound.

    DSYEVR and DSYEVD perform comparably at these scales — no driver-
    specific multiplier is applied.

    This is a minimum estimate — memory pressure is not accounted for.

    Args:
        n_samples: Number of samples.
        n_cores: Physical core count. None auto-detects.
        use_dsyevr: Accepted for API compatibility. No multiplier applied.

    Returns:
        Human-readable minimum estimate string like ">=1h 47m".
    """
    if n_cores is None:
        n_cores = get_physical_core_count()

    n_k = n_samples / 1000
    core_ratio = _REF_CORES / n_cores

    est = _EIGEN_COEFF * n_k**_EIGEN_ALPHA * core_ratio
    return f">={_format_duration(est)}"


def estimate_lmm_time(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> str:
    """Estimate LMM association wall time.

    Polynomial model: a*n_k² + b*n_k, scaled by SNP ratio and core ratio.
    The quadratic term captures UT@G rotation (DGEMM) and eigen/genotype
    loading. The linear term captures Wald test compute (C extension).

    This is a minimum estimate — covariates increase per-SNP Pab
    computation cost, and memory pressure adds further overhead.

    Args:
        n_samples: Number of samples.
        n_snps: Number of filtered SNPs.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Human-readable minimum estimate string like ">=15 min".
    """
    if n_cores is None:
        n_cores = get_physical_core_count()

    n_k = n_samples / 1000
    snp_ratio = n_snps / _REF_SNPS
    core_ratio = _REF_CORES / n_cores

    est = (_LMM_A * n_k**2 + _LMM_B * n_k) * snp_ratio * core_ratio
    return f">={_format_duration(est)}"
