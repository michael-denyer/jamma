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

    Note: This is a conservative estimate based on a fixed batch size. The JAX
    runner computes its own chunk size based on int32 buffer limits, which may
    differ. This check ensures sufficient memory for the dominant costs (kinship
    and eigenvector matrices at O(n_samples²)) but may underestimate memory for
    very large SNP counts on small-memory systems.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        lmm_batch_size: Batch size for LMM SNP processing (used for estimate,
            actual JAX chunk size is computed from int32 limits).

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


class StreamingMemoryBreakdown(NamedTuple):
    """Detailed memory breakdown for streaming GWAS workflow.

    All values in GB. Peak memory is the maximum across workflow phases:
    1. Kinship accumulation: kinship + chunk
    2. Eigendecomposition: kinship + eigenvectors + workspace (typically peak)
    3. LMM: eigenvectors + chunk + rotation buffer + grid REML

    The key difference from full-load estimation is that genotypes are
    O(n * chunk_size), not O(n * n_snps).
    """

    kinship_gb: float  # n^2 * 8 bytes (float64)
    eigenvectors_gb: float  # n^2 * 8 bytes (float64)
    eigendecomp_workspace_gb: float  # ~26*n*8 + 10*n*4 bytes for DSYEVR
    chunk_gb: float  # n * chunk_size * 8 bytes (float64 for precision)
    rotation_buffer_gb: float  # n * chunk_size * 8 bytes for UtG
    grid_reml_gb: float  # n_grid * chunk_size * 8 bytes for Grid REML intermediate
    total_peak_gb: float  # Max of phases (eigendecomp is typically peak)
    available_gb: float  # Current available system memory
    sufficient: bool  # Whether available >= total * 1.1


def estimate_streaming_memory(
    n_samples: int,
    n_snps: int,  # Only for logging; not used in peak calculation
    chunk_size: int = 10_000,
    n_grid: int = 50,
) -> StreamingMemoryBreakdown:
    """Estimate memory requirements for streaming GWAS workflow.

    Calculates memory for streaming kinship computation, eigendecomposition,
    and LMM association testing. Returns the peak memory requirement.

    Key difference from full-load estimation:
    - Genotypes: O(n * chunk_size) not O(n * n_snps)
    - Peak is typically eigendecomposition (kinship + eigenvectors simultaneously)

    For 200k samples, 95k SNPs, 10k chunk, n_grid=50:
    - Kinship accumulation: 320GB + 16GB = 336GB
    - Eigendecomp: 320GB + 320GB + 0.04GB = 640GB (PEAK)
    - LMM: 320GB + 16GB + 16GB + 4MB = 356GB (still not peak since eigendecomp is 640GB)

    Note: This reveals the true constraint - eigendecomposition cannot be
    streamed and requires both kinship (input) and eigenvectors (output)
    matrices simultaneously.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (for logging only, not used in peak calculation).
        chunk_size: SNPs per chunk (default 10,000).
        n_grid: Grid points for lambda optimization (default 50).

    Returns:
        StreamingMemoryBreakdown with detailed component estimates.

    Example:
        >>> est = estimate_streaming_memory(200_000, 95_000)
        >>> print(f"Peak: {est.total_peak_gb:.0f}GB (eigendecomp)")
    """
    # Component sizes
    kinship_gb = n_samples**2 * 8 / 1e9  # float64
    eigenvectors_gb = n_samples**2 * 8 / 1e9  # float64

    # Eigendecomp workspace: DSYEVR uses O(n) workspace
    # Formula: LWORK=26*n doubles + LIWORK=10*n integers
    eigendecomp_workspace_gb = (26 * n_samples * 8 + 10 * n_samples * 4) / 1e9

    # Chunk memory (float64 for precision in kinship accumulation)
    chunk_gb = n_samples * chunk_size * 8 / 1e9
    rotation_buffer_gb = n_samples * chunk_size * 8 / 1e9  # UtG buffer

    # Grid REML intermediate: _batch_grid_reml creates (n_grid, chunk_size) arrays
    # during vmap over lambda values for log-likelihood evaluation
    grid_reml_gb = n_grid * chunk_size * 8 / 1e9

    # Peak memory calculation by workflow phase
    # Phase 1 (kinship accumulation): kinship + chunk
    peak_kinship = kinship_gb + chunk_gb

    # Phase 2 (eigendecomp): kinship (input) + eigenvectors (output) + workspace
    peak_eigendecomp = kinship_gb + eigenvectors_gb + eigendecomp_workspace_gb

    # Phase 3 (LMM): eigenvectors + chunk + rotation buffer + grid REML
    # (kinship can be freed after eigendecomp)
    peak_lmm = eigenvectors_gb + chunk_gb + rotation_buffer_gb + grid_reml_gb

    total_peak_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)

    # Check available memory
    available_gb = psutil.virtual_memory().available / 1e9
    sufficient = total_peak_gb * 1.1 < available_gb  # 10% safety margin

    return StreamingMemoryBreakdown(
        kinship_gb=kinship_gb,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=eigendecomp_workspace_gb,
        chunk_gb=chunk_gb,
        rotation_buffer_gb=rotation_buffer_gb,
        grid_reml_gb=grid_reml_gb,
        total_peak_gb=total_peak_gb,
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
