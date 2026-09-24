"""Top-level GWAS API for JAMMA.

Provides a single-call entry point for running a complete GWAS pipeline:
load data, compute or load kinship, run LMM association, write results.

Example:
    >>> from jamma import gwas
    >>> result = gwas("data/my_study", kinship_file="data/kinship.cXX.txt")
    >>> print(f"Tested {result.n_snps_tested} SNPs in {result.timing.total_s:.1f}s")
"""

from __future__ import annotations

from collections.abc import Sequence
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
    bfile: str | Path | None = None,
    *,
    bgen: str | Path | None = None,
    sample: str | Path | None = None,
    bgi: str | Path | None = None,
    phenotype_file: str | Path | None = None,
    info: float = 0.0,
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
    phenotype_columns: Sequence[int] = (1,),
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
    ``jamma -lmm`` command but as a Python function.

    Pass exactly one genotype input: ``bfile``, or ``bgen`` with
    ``phenotype_file`` (``sample`` and ``bgi`` default from the ``.bgen``
    path). Each keyword is the ``PipelineConfig`` field of the same name,
    except ``hwe``, which is ``hwe_threshold``, and ``info``, which is
    ``info_threshold``. ``PipelineConfig`` documents every field, and
    ``tests/test_gwas_api.py`` pins that the two sets match.

    Returns:
        PipelineResult with association results, sample/SNP counts, output
        paths, timing, and the PVE estimate.

    Raises:
        FileNotFoundError: If a genotype file does not exist.
        ValueError: If the keywords combine illegally, no valid phenotypes
            are found, or the covariate row count mismatches the sample count.
        MemoryError: If check_memory=True and insufficient memory available.

    Example:
        >>> from jamma import gwas
        >>> result = gwas("data/mouse_hs1940", loco=True)
        >>> print(f"{result.n_snps_tested} SNPs, {result.timing.total_s:.1f}s")
    """
    config = PipelineConfig(
        bfile=_opt_path(bfile),
        bgen=_opt_path(bgen),
        sample=_opt_path(sample),
        bgi=_opt_path(bgi),
        phenotype_file=_opt_path(phenotype_file),
        info_threshold=info,
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
        phenotype_columns=phenotype_columns,
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
