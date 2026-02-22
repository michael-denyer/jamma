"""Wall clock time estimates for GWAS pipeline phases.

Estimates are based on FLOP scaling from a reference benchmark
(125k samples, 91k SNPs, 48 cores, MKL ILP64 on Intel Xeon 8573C).

Accuracy is roughly ±30% for 50k–125k samples. Smaller datasets
will overestimate (startup overhead dominates). Larger datasets may
underestimate (memory pressure grows super-linearly).

Reference benchmark from PERFORMANCE.md (Azure E96ds_v6):
  125k (v2.5): kinship 2,011s, eigendecomp 8,465s, LMM 1,131s
"""

from __future__ import annotations

from jamma.core.threading import get_physical_core_count

# Reference benchmark: 125,632 samples, 91,586 SNPs
# Measured on Azure E96ds_v6 (Xeon Platinum 8573C, 48 physical cores),
# MKL ILP64, JAX 0.9.0, v2.5.5 with psutil threading fix active.
# All 48 physical cores used for BLAS (eigendecomp, rotation).
_REF_SAMPLES = 125_632
_REF_SNPS = 91_586
_REF_CORES = 48

_REF_KINSHIP_SECS = 2_011.0  # 34 min
_REF_EIGEN_SECS = 8_465.0  # 2h 21m
_REF_LMM_SECS = 1_131.0  # 19 min

# Memory bandwidth saturation threshold for eigendecomp.
# Above ~100k samples, eigenvector matrices exceed L3 cache and the
# computation becomes memory-bandwidth-bound. The reference (125k) already
# includes this penalty, so we divide it out for smaller datasets where
# eigendecomp is purely compute-bound.
_EIGEN_BW_THRESHOLD = 100_000
_EIGEN_BW_PENALTY = 1.5  # divisor below threshold (reference includes penalty)


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = minutes / 60
    remaining_min = minutes % 60
    if remaining_min < 1:
        return f"{hours:.0f}h"
    return f"{hours:.0f}h {remaining_min:.0f}m"


def estimate_kinship_time(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> str:
    """Estimate kinship computation wall time.

    Kinship is O(n² × m) batched dgemm. Scales quadratically with
    samples, linearly with SNPs, roughly inversely with core count.

    Args:
        n_samples: Number of samples.
        n_snps: Number of SNPs.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Human-readable estimate string like "~24 min".
    """
    if n_cores is None:
        n_cores = get_physical_core_count()

    sample_ratio = (n_samples / _REF_SAMPLES) ** 2
    snp_ratio = n_snps / _REF_SNPS
    core_ratio = _REF_CORES / n_cores

    est = _REF_KINSHIP_SECS * sample_ratio * snp_ratio * core_ratio
    return f"~{_format_duration(est)}"


def estimate_eigendecomp_time(
    n_samples: int,
    n_cores: int | None = None,
) -> str:
    """Estimate eigendecomposition wall time.

    Eigendecomp (dsyevd) is O(n³). Scales cubically with samples,
    roughly inversely with core count. Above ~100k samples, eigenvector
    matrices exceed L3 cache and memory bandwidth becomes the bottleneck.
    Below ~100k, estimates are reduced by ~33% to account for the
    compute-bound regime being faster than the BW-bound reference.

    Args:
        n_samples: Number of samples.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Human-readable estimate string like "~2h 21m".
    """
    if n_cores is None:
        n_cores = get_physical_core_count()

    sample_ratio = (n_samples / _REF_SAMPLES) ** 3
    core_ratio = _REF_CORES / n_cores

    est = _REF_EIGEN_SECS * sample_ratio * core_ratio

    # Reference already includes BW penalty (125k > threshold).
    # For smaller datasets that are compute-bound, remove the penalty.
    if n_samples <= _EIGEN_BW_THRESHOLD:
        est /= _EIGEN_BW_PENALTY

    return f"~{_format_duration(est)}"


def estimate_lmm_time(
    n_samples: int,
    n_snps: int,
    n_cores: int | None = None,
) -> str:
    """Estimate LMM association wall time.

    LMM is dominated by the U.T @ G rotation which is O(n² × m).
    Same scaling as kinship, but with different constant factor
    (JAX compute per chunk adds overhead).

    Args:
        n_samples: Number of samples.
        n_snps: Number of filtered SNPs.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Human-readable estimate string like "~20 min".
    """
    if n_cores is None:
        n_cores = get_physical_core_count()

    sample_ratio = (n_samples / _REF_SAMPLES) ** 2
    snp_ratio = n_snps / _REF_SNPS
    core_ratio = _REF_CORES / n_cores

    est = _REF_LMM_SECS * sample_ratio * snp_ratio * core_ratio
    return f"~{_format_duration(est)}"
