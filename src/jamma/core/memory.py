"""Memory estimation and checking for large-scale GWAS operations.

Provides pre-allocation memory checks to prevent OOM errors at 200k sample scale.
"""

from typing import NamedTuple

import psutil


class MemoryBreakdown(NamedTuple):
    """Detailed memory breakdown for GWAS workflow.

    All values in GB. Peak memory is the maximum of eigendecomp phase
    and LMM phase since they don't overlap.
    """

    kinship_gb: float  # n^2 * 8 bytes (float64)
    genotypes_gb: float  # n * p * 4 bytes (float32)
    eigenvectors_gb: float  # n^2 * 8 bytes (float64)
    eigendecomp_workspace_gb: float  # ~26*n*8 + 10*n*4 bytes for DSYEVR
    lmm_rotated_gb: float  # n * 8 * 3 bytes (Uy, UW, rotated vectors)
    lmm_batch_gb: float  # n * batch_size * 8 bytes
    total_gb: float  # Peak memory (max of phases)
    available_gb: float  # Current available system memory
    sufficient: bool  # Whether available >= total * 1.1


def estimate_workflow_memory(
    n_samples: int,
    n_snps: int,
    lmm_batch_size: int = 20_000,
) -> MemoryBreakdown:
    """Estimate memory requirements for full GWAS workflow.

    Calculates memory for kinship computation, eigendecomposition, and LMM
    association testing. Returns the peak memory requirement.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        lmm_batch_size: Batch size for LMM SNP processing.

    Returns:
        MemoryBreakdown with detailed component estimates and total.

    Example:
        >>> est = estimate_workflow_memory(200_000, 95_000)
        >>> print(f"Need {est.total_gb:.0f}GB, have {est.available_gb:.0f}GB")
    """
    # Component sizes
    kinship_gb = n_samples**2 * 8 / 1e9  # float64
    genotypes_gb = n_samples * n_snps * 4 / 1e9  # float32
    eigenvectors_gb = n_samples**2 * 8 / 1e9  # float64

    # Eigendecomp workspace: DSYEVR uses O(n) workspace
    # Formula: LWORK=26*n doubles + LIWORK=10*n integers
    eigendecomp_workspace_gb = (26 * n_samples * 8 + 10 * n_samples * 4) / 1e9

    # LMM working memory
    lmm_rotated_gb = n_samples * 8 * 3 / 1e9  # Uy, UW, Ux per SNP
    lmm_batch_gb = n_samples * lmm_batch_size * 8 / 1e9

    # Peak memory calculation
    # Workflow: genotypes -> kinship -> eigendecomp -> LMM
    # Kinship can be freed after eigendecomp

    # Phase 1 (kinship): genotypes + kinship (building kinship from genotypes)
    peak_kinship = genotypes_gb + kinship_gb

    # Phase 2 (eigendecomp): kinship + eigenvectors + workspace
    # (genotypes can be freed during eigendecomp if not needed for LMM)
    peak_eigendecomp = kinship_gb + eigenvectors_gb + eigendecomp_workspace_gb

    # Phase 3 (LMM): eigenvectors + genotypes + working
    # (kinship freed, eigenvalues are small ~n*8 bytes)
    eigenvalues_gb = n_samples * 8 / 1e9
    peak_lmm = (
        eigenvectors_gb + genotypes_gb + eigenvalues_gb + lmm_rotated_gb + lmm_batch_gb
    )

    total_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)

    # Check available memory
    available_gb = psutil.virtual_memory().available / 1e9
    sufficient = total_gb * 1.1 < available_gb  # 10% safety margin

    return MemoryBreakdown(
        kinship_gb=kinship_gb,
        genotypes_gb=genotypes_gb,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=eigendecomp_workspace_gb,
        lmm_rotated_gb=lmm_rotated_gb,
        lmm_batch_gb=lmm_batch_gb,
        total_gb=total_gb,
        available_gb=available_gb,
        sufficient=sufficient,
    )


def check_memory_available(
    required_gb: float,
    safety_margin: float = 0.1,
    operation: str = "operation",
) -> bool:
    """Check if sufficient memory is available, raise if not.

    Args:
        required_gb: Memory required in GB.
        safety_margin: Additional margin (0.1 = 10%).
        operation: Description for error message.

    Returns:
        True if sufficient memory available.

    Raises:
        MemoryError: If insufficient memory with detailed message.
    """
    available_gb = psutil.virtual_memory().available / 1e9
    required_with_margin = required_gb * (1 + safety_margin)

    if required_with_margin > available_gb:
        raise MemoryError(
            f"Insufficient memory for {operation}. "
            f"Need {required_gb:.1f}GB (+{safety_margin*100:.0f}% margin = "
            f"{required_with_margin:.1f}GB), but only {available_gb:.1f}GB available. "
            f"Consider using a machine with more RAM or reducing dataset size."
        )

    return True
