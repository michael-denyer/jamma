"""Top-level GWAS API for JAMMA.

Provides a single-call entry point for running a complete GWAS pipeline:
load data, compute or load kinship, run LMM association, write results.

Example:
    >>> from jamma import gwas
    >>> result = gwas("data/my_study", kinship_file="data/kinship.cXX.txt")
    >>> print(f"Tested {result.n_snps_tested} SNPs in {result.timing['total_s']:.1f}s")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jamma.lmm.schema import GWASTiming
from jamma.lmm.stats import AssocResult
from jamma.pipeline import PipelineConfig, PipelineRunner


def _opt_path(value: str | Path | None) -> Path | None:
    """Convert optional string or Path to Path."""
    return Path(value) if value is not None else None


@dataclass
class GWASResult:
    """Result of a GWAS pipeline run.

    Attributes:
        associations: Per-SNP association results. Empty when output_path is
            used (results written to disk instead).
        n_samples: Number of samples after phenotype and covariate filtering.
        n_snps_tested: Number of SNPs tested after MAF/missingness/HWE/SNP-list
            filtering.
        timing: Timing breakdown with keys 'kinship_s', 'lmm_s', 'total_s'.
        pve_estimate: PVE (proportion of variance explained) from null model REML.
        pve_se: Standard error of PVE from REML second derivative delta method.
            None if not computed or likelihood surface is flat.
    """

    associations: list[AssocResult]
    n_samples: int
    n_snps_tested: int
    timing: GWASTiming = field(default_factory=dict)
    pve_estimate: float | None = None
    pve_se: float | None = None


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
    loco: bool = False,
    eigenvalue_file: str | Path | None = None,
    eigenvector_file: str | Path | None = None,
    write_eigen: bool = False,
    phenotype_column: int = 1,
    snps_file: str | Path | None = None,
    ksnps_file: str | Path | None = None,
    hwe: float = 0.0,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    weight_file: str | Path | None = None,
    cat_columns: list[int] | None = None,
    backend: str = "auto",
) -> GWASResult:
    """Run a complete GWAS pipeline in a single call.

    Orchestrates data loading, kinship computation (or loading), LMM
    association testing, and result writing. Equivalent to the CLI
    ``jamma lmm`` command but as a Python function.

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
        loco: If True, enable leave-one-chromosome-out analysis.
            Computes per-chromosome kinship internally.
        eigenvalue_file: Pre-computed eigenvalue file (.eigenD.txt).
            Must be paired with eigenvector_file.
        eigenvector_file: Pre-computed eigenvector file (.eigenU.txt).
            Must be paired with eigenvalue_file.
        write_eigen: If True, write eigendecomposition files as
            side effect of the pipeline run.
        phenotype_column: 1-based phenotype column index in the .fam file.
            1 selects the standard phenotype (column 6), 2 selects column 7,
            etc. Matches GEMMA's ``-n`` flag.
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

    Returns:
        GWASResult with association results, sample/SNP counts, and timing.

    Raises:
        FileNotFoundError: If PLINK files (.bed, .bim, .fam) do not exist.
        ValueError: If lmm_mode is not in (1, 2, 3, 4), no valid phenotypes
            found, covariate row count mismatches sample count, or if both
            kinship_file and loco are specified.
        MemoryError: If check_memory=True and insufficient memory available.

    Example:
        >>> from jamma import gwas
        >>> result = gwas("data/mouse_hs1940", loco=True)
        >>> print(f"{result.n_snps_tested} SNPs, {result.timing['total_s']:.1f}s")
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
        loco=loco,
        eigenvalue_file=_opt_path(eigenvalue_file),
        eigenvector_file=_opt_path(eigenvector_file),
        write_eigen=write_eigen,
        phenotype_column=phenotype_column,
        snps_file=_opt_path(snps_file),
        ksnps_file=_opt_path(ksnps_file),
        hwe_threshold=hwe,
        l_min=l_min,
        l_max=l_max,
        weight_file=_opt_path(weight_file),
        cat_columns=cat_columns,
        backend=backend,
    )

    pipeline_result = PipelineRunner(config).run()

    return GWASResult(
        associations=pipeline_result.associations,
        n_samples=pipeline_result.n_samples,
        n_snps_tested=pipeline_result.n_snps_tested,
        timing={
            "kinship_s": pipeline_result.timing["kinship_s"],
            "lmm_s": pipeline_result.timing["lmm_s"],
            "total_s": pipeline_result.timing["total_s"],
        },
        pve_estimate=pipeline_result.pve_estimate,
        pve_se=pipeline_result.pve_se,
    )
