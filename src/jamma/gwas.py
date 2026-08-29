"""Top-level GWAS API for JAMMA.

Provides a single-call entry point for running a complete GWAS pipeline:
load data, compute or load kinship, run LMM association, write results.

Example:
    >>> from jamma import gwas
    >>> result = gwas("data/my_study", kinship_file="data/kinship.cXX.txt")
    >>> print(f"Tested {result.n_snps_tested} SNPs in {result.timing.total_s:.1f}s")
"""

from __future__ import annotations

from pathlib import Path

from jamma.lmm.schema import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_MAF,
    DEFAULT_MISS,
    DEFAULT_N_GRID,
    DEFAULT_N_REFINE,
)
from jamma.pipeline import (
    BackendRequest,
    PipelineConfig,
    PipelineResult,
    PipelineRunner,
)


def _opt_path(value: str | Path | None) -> Path | None:
    """Convert optional string or Path to Path."""
    return Path(value) if value is not None else None


def gwas(
    bfile: str | Path,
    *,
    kinship_file: str | Path | None = None,
    covariate_file: str | Path | None = None,
    lmm_mode: int = 1,
    maf: float = DEFAULT_MAF,
    miss: float = DEFAULT_MISS,
    output_dir: str | Path = "output",
    output_prefix: str = "result",
    save_kinship: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    mem_budget: float | None = None,
    loco: bool = False,
    eigenvalue_file: str | Path | None = None,
    eigenvector_file: str | Path | None = None,
    write_eigen: bool = False,
    eigen_dir: str | Path | None = None,
    phenotype_columns: list[int] | None = None,
    snps_file: str | Path | None = None,
    ksnps_file: str | Path | None = None,
    hwe: float = 0.0,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    n_grid: int = DEFAULT_N_GRID,
    n_refine: int = DEFAULT_N_REFINE,
    weight_file: str | Path | None = None,
    cat_columns: list[int] | None = None,
    backend: BackendRequest = "auto",
    legacy_text: bool = False,
    no_telemetry: bool = False,
) -> PipelineResult:
    """Run a complete GWAS pipeline in a single call.

    Orchestrates data loading, kinship computation (or loading), LMM
    association testing, and result writing. Equivalent to the CLI
    ``jamma -lmm`` command but as a Python function: every keyword here is
    one ``PipelineConfig`` field, and ``tests/test_gwas_api.py`` pins that
    the two sets match.

    When ``loco=True``, runs leave-one-chromosome-out analysis: computes
    a separate LOCO kinship matrix for each chromosome, eigendecomposes
    it, and runs LMM association on that chromosome's SNPs. This
    eliminates proximal contamination. The ``kinship_file`` parameter
    must be None when ``loco=True`` (mutually exclusive).

    When ``eigenvalue_file`` and ``eigenvector_file`` are provided,
    loads pre-computed eigendecomposition and skips both kinship loading
    and eigendecomposition. Both must be provided together.

    Args:
        bfile: PLINK binary file prefix (without .bed/.bim/.fam extension).
        kinship_file: Pre-computed kinship matrix file (.cXX.txt format).
            If None, kinship is computed from genotypes. Must be None
            when loco=True.
        covariate_file: GEMMA-format covariate file (whitespace-delimited,
            no header). If None, intercept-only model is used.
        lmm_mode: LMM test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        maf: Minor allele frequency threshold for SNP filtering.
        miss: Missing rate threshold for SNP filtering.
        output_dir: Directory for output files (created if needed).
        output_prefix: Prefix for output filenames.
        save_kinship: If True, save computed kinship matrix to disk.
            In LOCO mode, saves per-chromosome kinship files.
        check_memory: If True, check available memory before computation.
        show_progress: If True, show progress bars and log messages.
        mem_budget: Hard memory budget in GB, or None for no budget.
            The CLI's ``--mem-budget``.
        loco: If True, enable leave-one-chromosome-out analysis.
            Computes per-chromosome kinship internally.
        eigenvalue_file: Pre-computed eigenvalue file (.eigenD.txt).
            Must be paired with eigenvector_file.
        eigenvector_file: Pre-computed eigenvector file (.eigenU.txt).
            Must be paired with eigenvalue_file.
        write_eigen: If True, write eigendecomposition files as
            side effect of the pipeline run.
        eigen_dir: Directory for the LOCO per-chromosome eigen cache. With
            ``loco=True`` and ``write_eigen=True`` it defaults to
            ``output_dir``. The CLI's ``--eigen-dir``.
        phenotype_columns: 1-based phenotype column indices in the .fam
            file, in the order they are tested. ``[1]`` (the default) selects
            the standard phenotype (column 6), ``[2]`` selects column 7, and
            ``[1, 2]`` runs both against one eigendecomposition. Matches
            GEMMA's ``-n`` flag.
        snps_file: File with SNP IDs to restrict association testing. One
            SNP ID per line (first token used). Matches GEMMA's ``-snps`` flag.
            None means test all SNPs.
        ksnps_file: File with SNP IDs to restrict kinship computation. One
            SNP ID per line. Matches GEMMA's ``-ksnps`` flag. None means
            use all SNPs for kinship.
        hwe: HWE p-value threshold. SNPs with Hardy-Weinberg equilibrium
            p-value below this threshold are excluded. 0.0 disables HWE
            filtering. Matches GEMMA's ``-hwe`` flag.
        l_min: Minimum lambda for optimization (default 1e-5, matches GEMMA).
        l_max: Maximum lambda for optimization (default 1e5, matches GEMMA).
        n_grid: Grid search resolution for lambda bracketing (default 50).
        n_refine: Golden section refinement iterations (default 10; the
            runners clamp it to at least 20).
        weight_file: Individual weight file for kinship pre-transformation.
            One weight per line, matching sample order. Applies
            K[i,j] /= sqrt(w_i * w_j) before eigendecomposition.
            GEMMA's ``-widv`` flag. None means no weight application.
        cat_columns: 1-indexed covariate column indices to treat as
            categorical. JAMMA-specific feature (not GEMMA's ``-cat`` which
            is for SNP categories in VC mode). Columns are one-hot encoded
            with the first sorted level dropped as reference. Requires
            covariate_file to be set.
        backend: Compute backend: "auto" (default) or "numpy". "auto"
            selects the best available numpy runner.
        legacy_text: If True, write kinship and eigen files in GEMMA text
            format instead of binary .npy. The CLI's ``--legacy-text``.
        no_telemetry: If True, skip local benchmark telemetry for this run.
            The CLI's ``--no-telemetry``. ``JAMMA_NO_TELEMETRY`` is honoured
            regardless of this flag.

    Returns:
        PipelineResult with association results, sample/SNP counts, output
        paths, timing, and the PVE estimate.

    Raises:
        FileNotFoundError: If PLINK files (.bed, .bim, .fam) do not exist.
        ValueError: If lmm_mode is not in (1, 2, 3, 4), no valid phenotypes
            found, covariate row count mismatches sample count, or if both
            kinship_file and loco are specified.
        MemoryError: If check_memory=True and insufficient memory available.

    Example:
        >>> from jamma import gwas
        >>> result = gwas("data/mouse_hs1940", loco=True)
        >>> print(f"{result.n_snps_tested} SNPs, {result.timing.total_s:.1f}s")
    """
    config = PipelineConfig(
        bfile=Path(bfile),
        kinship_file=_opt_path(kinship_file),
        covariate_file=_opt_path(covariate_file),
        lmm_mode=lmm_mode,
        maf=maf,
        miss=miss,
        output_dir=Path(output_dir),
        output_prefix=output_prefix,
        save_kinship=save_kinship,
        check_memory=check_memory,
        show_progress=show_progress,
        mem_budget=mem_budget,
        loco=loco,
        eigenvalue_file=_opt_path(eigenvalue_file),
        eigenvector_file=_opt_path(eigenvector_file),
        write_eigen=write_eigen,
        eigen_dir=_opt_path(eigen_dir),
        phenotype_columns=[1] if phenotype_columns is None else list(phenotype_columns),
        snps_file=_opt_path(snps_file),
        ksnps_file=_opt_path(ksnps_file),
        hwe_threshold=hwe,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_refine=n_refine,
        weight_file=_opt_path(weight_file),
        cat_columns=cat_columns,
        backend=backend,
        legacy_text=legacy_text,
        no_telemetry=no_telemetry,
    )
    return PipelineRunner(config).run()
