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

from jamma.lmm.stats import AssocResult
from jamma.pipeline import PipelineConfig, PipelineRunner


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
    loco: bool = False,
    eigenvalue_file: str | Path | None = None,
    eigenvector_file: str | Path | None = None,
    write_eigen: bool = False,
    phenotype_column: int = 1,
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
        kinship_file=Path(kinship_file) if kinship_file is not None else None,
        covariate_file=(Path(covariate_file) if covariate_file is not None else None),
        lmm_mode=lmm_mode,
        maf=maf,
        miss=miss,
        output_dir=Path(output_dir),
        output_prefix=output_prefix,
        save_kinship=save_kinship,
        check_memory=check_memory,
        show_progress=show_progress,
        loco=loco,
        eigenvalue_file=(
            Path(eigenvalue_file) if eigenvalue_file is not None else None
        ),
        eigenvector_file=(
            Path(eigenvector_file) if eigenvector_file is not None else None
        ),
        write_eigen=write_eigen,
        phenotype_column=phenotype_column,
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
    )
