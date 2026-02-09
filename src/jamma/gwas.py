"""Top-level GWAS API for JAMMA.

Provides a single-call entry point for running a complete GWAS pipeline:
load data, compute or load kinship, run LMM association, write results.

Example:
    >>> from jamma import gwas
    >>> result = gwas("data/my_study", kinship_file="data/kinship.cXX.txt")
    >>> print(f"Tested {result.n_snps_tested} SNPs in {result.timing['total_s']:.1f}s")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.memory import estimate_streaming_memory
from jamma.io.covariate import read_covariate_file
from jamma.io.plink import get_plink_metadata
from jamma.kinship import (
    compute_kinship_streaming,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm import run_lmm_association_streaming
from jamma.lmm.stats import AssocResult


@dataclass
class GWASResult:
    """Result of a GWAS pipeline run.

    Attributes:
        associations: Per-SNP association results. Empty when output_path is
            used (results written to disk instead).
        n_samples: Number of samples with valid phenotypes.
        n_snps_tested: Total number of SNPs in the dataset.
        timing: Timing breakdown with keys 'kinship_s', 'eigendecomp_s',
            'lmm_s', 'total_s'.
    """

    associations: list[AssocResult]
    n_samples: int
    n_snps_tested: int
    timing: dict[str, float] = field(default_factory=dict)


def gwas(
    bfile: str | Path,
    *,
    kinship_file: str | Path | None = None,
    covariate_file: str | Path | None = None,
    lmm_mode: int = 1,
    maf: float = 0.01,
    miss: float = 0.05,
    output_dir: str | Path = "output",
    output_prefix: str = "result",
    save_kinship: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
) -> GWASResult:
    """Run a complete GWAS pipeline in a single call.

    Orchestrates data loading, kinship computation (or loading), LMM
    association testing, and result writing. Equivalent to the CLI
    ``jamma lmm`` command but as a Python function.

    Args:
        bfile: PLINK binary file prefix (without .bed/.bim/.fam extension).
        kinship_file: Pre-computed kinship matrix file (.cXX.txt format).
            If None, kinship is computed from genotypes.
        covariate_file: GEMMA-format covariate file (whitespace-delimited,
            no header). If None, intercept-only model is used.
        lmm_mode: LMM test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        maf: Minor allele frequency threshold for SNP filtering.
        miss: Missing rate threshold for SNP filtering.
        output_dir: Directory for output files (created if needed).
        output_prefix: Prefix for output filenames.
        save_kinship: If True, save computed kinship matrix to disk.
        check_memory: If True, check available memory before computation.
        show_progress: If True, show progress bars and log messages.

    Returns:
        GWASResult with association results, sample/SNP counts, and timing.

    Raises:
        FileNotFoundError: If PLINK files (.bed, .bim, .fam) do not exist.
        ValueError: If lmm_mode is not in (1, 2, 3, 4), no valid phenotypes
            found, or covariate row count mismatches sample count.
        MemoryError: If check_memory=True and insufficient memory available.

    Example:
        >>> from jamma import gwas
        >>> result = gwas(
        ...     "data/mouse_hs1940",
        ...     kinship_file="data/kinship.cXX.txt",
        ...     output_dir="results",
        ... )
        >>> print(f"{result.n_snps_tested} SNPs, {result.timing['total_s']:.1f}s")
    """
    t_start = time.perf_counter()

    # --- Validate inputs ---
    bfile = Path(bfile)
    for ext in (".bed", ".bim", ".fam"):
        p = Path(f"{bfile}{ext}")
        if not p.exists():
            raise FileNotFoundError(f"PLINK {ext} file not found: {p}")

    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    # --- Prepare output ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assoc_path = output_dir / f"{output_prefix}.assoc.txt"
    logger.info(f"Output: {assoc_path}")

    # --- Get metadata ---
    meta = get_plink_metadata(bfile)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]

    # --- Memory check ---
    if check_memory:
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size=10_000)
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory: need {est.total_peak_gb:.1f}GB, "
                f"have {est.available_gb:.1f}GB"
            )

    # --- Parse phenotypes from .fam ---
    fam_data = np.loadtxt(f"{bfile}.fam", dtype=str, usecols=(5,))
    missing_mask = np.isin(fam_data, ["-9", "NA"])
    fam_data[missing_mask] = "0"  # placeholder for safe float conversion
    phenotypes = fam_data.astype(np.float64)
    phenotypes[missing_mask] = np.nan

    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9)
    n_analyzed = int(valid_mask.sum())
    if n_analyzed == 0:
        raise ValueError("No samples with valid phenotypes")

    logger.info(
        f"Analyzing {n_analyzed} samples with valid phenotypes "
        f"({n_samples - n_analyzed} filtered)"
    )

    # --- Kinship ---
    t_kinship_start = time.perf_counter()
    if kinship_file is not None:
        kinship_file = Path(kinship_file)
        logger.info(f"Loading kinship from {kinship_file}")
        K = read_kinship_matrix(kinship_file, n_samples=n_samples)
    else:
        logger.info("Computing kinship from genotypes")
        K = compute_kinship_streaming(
            bfile, check_memory=False, show_progress=show_progress
        )
    t_kinship_end = time.perf_counter()
    kinship_s = t_kinship_end - t_kinship_start

    if save_kinship:
        kinship_path = output_dir / f"{output_prefix}.cXX.txt"
        write_kinship_matrix(K, kinship_path)
        logger.info(f"Kinship matrix saved to {kinship_path}")

    # --- Covariates ---
    covariates = None
    if covariate_file is not None:
        covariates, _ = read_covariate_file(Path(covariate_file))
        if covariates.shape[0] != n_samples:
            raise ValueError(
                f"Covariate file has {covariates.shape[0]} rows "
                f"but PLINK data has {n_samples} samples"
            )

    # --- Run LMM ---
    t_lmm_start = time.perf_counter()
    results = run_lmm_association_streaming(
        bed_path=bfile,
        phenotypes=phenotypes,
        kinship=K,
        snp_info=None,
        covariates=covariates,
        maf_threshold=maf,
        miss_threshold=miss,
        output_path=assoc_path,
        lmm_mode=lmm_mode,
        check_memory=False,  # Already checked above
        show_progress=show_progress,
    )
    t_lmm_end = time.perf_counter()
    lmm_s = t_lmm_end - t_lmm_start

    total_s = time.perf_counter() - t_start
    logger.info(f"GWAS complete: {n_snps} SNPs tested in {total_s:.1f}s")

    return GWASResult(
        associations=results,
        n_samples=n_analyzed,
        n_snps_tested=n_snps,
        timing={
            "kinship_s": kinship_s,
            "lmm_s": lmm_s,
            "total_s": total_s,
        },
    )
