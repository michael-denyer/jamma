"""Wall clock time estimates for GWAS pipeline phases.

Estimates are based on FLOP scaling from a reference benchmark
(90k samples, 90k SNPs, 32 cores, MKL ILP64 on Intel Xeon 8573C).
Accuracy is ±50% — useful for setting expectations, not for billing.

The reference benchmark is from PERFORMANCE.md (v1.4, Phase 19).
"""

from __future__ import annotations

from jamma.core.threading import get_physical_core_count

# Reference benchmark: 90k samples, 90k SNPs, 32 physical cores
# Measured on Azure E64ds_v6 (Xeon Platinum 8573C), MKL ILP64
_REF_SAMPLES = 90_000
_REF_SNPS = 90_000
_REF_CORES = 32

_REF_KINSHIP_SECS = 1_440.0  # 24 min
_REF_EIGEN_SECS = 3_114.0  # 52 min
_REF_LMM_SECS = 1_211.0  # 20 min


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

    # FLOP ratio: (n/n_ref)² × (m/m_ref) × (cores_ref/cores)
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
    roughly inversely with core count (memory bandwidth limited).

    Args:
        n_samples: Number of samples.
        n_cores: Physical core count. None auto-detects.

    Returns:
        Human-readable estimate string like "~52 min".
    """
    if n_cores is None:
        n_cores = get_physical_core_count()

    sample_ratio = (n_samples / _REF_SAMPLES) ** 3
    core_ratio = _REF_CORES / n_cores

    est = _REF_EIGEN_SECS * sample_ratio * core_ratio
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
