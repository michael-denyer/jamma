"""Data shapes for the pipeline: inputs, outputs, and the kinship result.

Split out of ``pipeline.py`` because they are data and it is behaviour. All
three are plain dataclasses — no I/O, no logging, no numerics — while
``PipelineRunner`` is the orchestrator that reads them.

``jamma.pipeline`` re-exports all three, so ``from jamma.pipeline import
PipelineConfig`` keeps working. That path is used by ``jamma.cli``,
``jamma.gwas`` and by the jamma-databricks notebooks, so it is load-bearing
rather than a courtesy.
"""

from __future__ import annotations

import operator
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from jamma.lmm.schema import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_MAF,
    DEFAULT_MISS,
    DEFAULT_N_GRID,
    DEFAULT_N_REFINE,
    ChunkRunStats,
    LmmConfig,
    PipelineTiming,
    parse_lmm_mode,
)
from jamma.lmm.stats import AssocResult

BackendRequest = Literal["auto", "numpy", "numpy-streaming"]
VALID_BACKENDS: tuple[BackendRequest, ...] = ("auto", "numpy", "numpy-streaming")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for a GWAS pipeline run.

    Attributes:
        bfile: PLINK binary file prefix (without .bed/.bim/.fam).
        kinship_file: Pre-computed kinship matrix file, or None to compute.
        covariate_file: GEMMA-format covariate file, or None for intercept-only.
        lmm_mode: LMM test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        maf: Minor allele frequency threshold.
        miss: Missing rate threshold.
        output_dir: Directory for output files.
        output_prefix: Prefix for output filenames.
        save_kinship: If True, save computed kinship matrix to disk.
        check_memory: If True, check available memory before computation.
        show_progress: If True, show progress bars and log messages.
        mem_budget: Hard memory budget in GB, or None for no budget.
        loco: If True, use leave-one-chromosome-out analysis. Computes
            per-chromosome kinship internally; mutually exclusive with
            kinship_file in this version.
        eigenvalue_file: Pre-computed eigenvalue file (.eigenD.npy or .eigenD.txt),
            or None. Must be paired with eigenvector_file (-d flag).
        eigenvector_file: Pre-computed eigenvector file (.eigenU.npy or .eigenU.txt),
            or None. Must be paired with eigenvalue_file (-u flag).
        write_eigen: If True, write eigendecomposition files as side effect
            (-eigen flag).
        eigen_dir: Directory for LOCO per-chromosome eigen cache. When set
            with loco mode, looks for cached eigen files to skip eigendecomp.
            Combined with write_eigen, writes per-chromosome files here.
        snps_file: SNP list file to restrict association testing. One SNP ID
            per line. Matches GEMMA's -snps flag. None means test all SNPs.
        ksnps_file: SNP list file to restrict kinship computation. One SNP ID
            per line. Matches GEMMA's -ksnps flag. None means use all SNPs.
        hwe_threshold: HWE p-value threshold. SNPs with HWE p-value below
            this threshold are excluded from association testing. 0.0 disables
            HWE filtering. Matches GEMMA's -hwe flag.
        l_min: Minimum lambda for optimization (default 1e-5, matches GEMMA).
        l_max: Maximum lambda for optimization (default 1e5, matches GEMMA).
        n_grid: Grid search resolution for lambda bracketing (default 50).
            Must be >= 2 — a one-point grid has no bracket to refine.
        n_refine: Golden section refinement iterations (default 20). A lower
            value is raised to 20 by LmmConfig rather than rejected here.
        weight_file: Individual weight file for kinship pre-transformation.
            One weight per line, matching sample order. Applies
            K[i,j] /= sqrt(w_i * w_j) before eigendecomposition.
            GEMMA's -widv flag.
        cat_columns: 1-indexed covariate column indices to treat as
            categorical. JAMMA-specific feature (not GEMMA's -cat which is
            for SNP categories in VC mode). Columns are one-hot encoded with
            the first sorted level dropped as reference.
        backend: Compute backend selection: "auto" (default) or "numpy".
            "auto" selects based on C extension availability and memory.
            "numpy" forces the pure-NumPy backend.
        legacy_text: If True, write kinship and eigen files in GEMMA text format
            (.cXX.txt / .eigenD.txt / .eigenU.txt) instead of binary .npy.
            Default False writes binary for performance at scale.
        phenotype_columns: 1-based phenotype column indices, in the order they
            are tested. 1 selects column 6 of .fam (the standard phenotype), 2
            selects column 7, and so on, matching GEMMA's -n flag. Defaults to
            [1]. Must name at least one column, every index >= 1 and distinct.
            With more than one column the eigendecomposition is computed once
            and reused; more than one is rejected in loco mode.
    """

    bfile: Path
    kinship_file: Path | None = None
    covariate_file: Path | None = None
    lmm_mode: int = 1
    maf: float = DEFAULT_MAF
    miss: float = DEFAULT_MISS
    output_dir: Path = field(default_factory=lambda: Path("output"))
    output_prefix: str = "result"
    save_kinship: bool = False
    check_memory: bool = True
    show_progress: bool = True
    mem_budget: float | None = None
    loco: bool = False
    eigenvalue_file: Path | None = None
    eigenvector_file: Path | None = None
    write_eigen: bool = False
    eigen_dir: Path | None = None
    snps_file: Path | None = None
    ksnps_file: Path | None = None
    hwe_threshold: float = 0.0
    l_min: float = DEFAULT_L_MIN
    l_max: float = DEFAULT_L_MAX
    n_grid: int = DEFAULT_N_GRID
    n_refine: int = DEFAULT_N_REFINE
    weight_file: Path | None = None
    cat_columns: Sequence[int] | None = None
    backend: BackendRequest = "auto"
    legacy_text: bool = False
    phenotype_columns: Sequence[int] = (1,)
    no_telemetry: bool = False

    def __post_init__(self) -> None:
        if self.cat_columns is not None:
            object.__setattr__(self, "cat_columns", tuple(self.cat_columns))
        try:
            phenotype_columns = tuple(
                operator.index(col) for col in self.phenotype_columns
            )
        except TypeError as exc:
            raise ValueError(
                f"phenotype_columns indices must be integers, "
                f"got {tuple(self.phenotype_columns)!r}"
            ) from exc
        object.__setattr__(self, "phenotype_columns", phenotype_columns)

        if os.sep in self.output_prefix or "/" in self.output_prefix:
            raise ValueError(
                f"output_prefix must not contain path separators, "
                f"got '{self.output_prefix}'. Use output_dir for directory paths."
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {VALID_BACKENDS}, got {self.backend!r}"
            )
        # Build the LmmConfig now and discard it: its __post_init__ owns every
        # rule for the knobs this config carries, and the LOCO branch reaches
        # the runners without building one. Constructing it here is what makes
        # an invalid knob fail at config time instead of after kinship and
        # eigendecomposition — or, on the NumPy fallback, not at all.
        self.lmm_config()
        # Range and emptiness are checked here rather than in
        # PipelineRunner.validate_inputs: an out-of-range column index is a
        # config error, not a filesystem one, so it should fail at construction
        # instead of surviving as far as a runner.
        if not self.phenotype_columns:
            raise ValueError("phenotype_columns must name at least one column")
        for col in self.phenotype_columns:
            if col < 1:
                raise ValueError(
                    f"phenotype_columns indices must be >= 1 (1-based), got {col}"
                )
        if len(self.phenotype_columns) != len(set(self.phenotype_columns)):
            raise ValueError(
                f"phenotype_columns contains duplicate indices: "
                f"{self.phenotype_columns}"
            )
        if self.mem_budget is not None and self.mem_budget <= 0:
            raise ValueError(f"mem_budget must be positive (GB), got {self.mem_budget}")
        if not 0 <= self.hwe_threshold <= 1:
            raise ValueError(
                f"hwe_threshold must be in [0, 1] (p-value threshold), "
                f"got {self.hwe_threshold}"
            )
        if self.hwe_threshold > 0 and self.loco:
            raise ValueError(
                "-hwe is not yet supported with -loco mode. "
                "Apply HWE filtering as a pre-processing step."
            )
        if self.cat_columns is not None:
            if self.covariate_file is None:
                raise ValueError("-cat requires -c (covariate file)")
            for col in self.cat_columns:
                if col < 1:
                    raise ValueError(
                        f"-cat column indices must be >= 1 (1-indexed), got {col}"
                    )
        # LOCO + multi-phenotype guard
        if self.loco and len(self.phenotype_columns) > 1:
            raise ValueError(
                "LOCO mode (-loco) does not support multi-phenotype "
                "(-n with multiple columns). "
                "Run each phenotype separately."
            )
        # LOCO writes a per-chromosome eigen cache keyed by eigen_dir. When the
        # caller asks to write eigen but gives no directory, default it to
        # output_dir so the Python API matches the CLI (which applies the same
        # default) instead of raising in run_lmm_loco. The non-LOCO write_eigen
        # path writes to output_dir directly and never consults eigen_dir.
        if self.loco and self.write_eigen and self.eigen_dir is None:
            object.__setattr__(self, "eigen_dir", self.output_dir)

    @property
    def log_path(self) -> Path:
        """Path to the GEMMA-compatible log file.

        Returns:
            Path to {output_dir}/{output_prefix}.log.txt
        """
        return self.output_dir / f"{self.output_prefix}.log.txt"

    def ensure_outdir(self) -> None:
        """Create the output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def lmm_config(self, *, check_memory: bool = False) -> LmmConfig:
        """Project the LMM knobs onto the config the runners take.

        The one place these fields are mapped onto LmmConfig — every
        dispatch path goes through here, so a knob added to LmmConfig cannot
        reach one runner and miss another.

        Built fresh on each call so the pipeline can choose whether the runner
        repeats the memory gate without duplicating the field mapping.

        Args:
            check_memory: Whether the runner should run its own memory gate.
                Defaults to False for the batch and streaming paths, where
                PipelineRunner._memory_preflight has already gated and
                re-checking would double-count. The LOCO path returns before
                that preflight and owns its per-chromosome estimate, so it
                passes the caller's flag through.

        Returns:
            LmmConfig carrying this config's optimizer and filter knobs.

        Raises:
            ValueError: If any knob falls outside its supported range.
        """
        return LmmConfig(
            maf_threshold=self.maf,
            miss_threshold=self.miss,
            l_min=self.l_min,
            l_max=self.l_max,
            n_grid=self.n_grid,
            n_refine=self.n_refine,
            check_memory=check_memory,
            show_progress=self.show_progress,
            lmm_mode=parse_lmm_mode(self.lmm_mode),
            mem_budget=self.mem_budget,
        )


@dataclass(frozen=True, slots=True)
class PhenotypeResult:
    """Association outcome and run metadata for one phenotype column.

    ``timing`` contains this phenotype's compute and result-write work. Shared
    genotype rotation time is divided evenly among the phenotypes in its
    bounded group, so summing every record matches
    ``PipelineResult.timing.rotation_s`` without counting the rotation twice.
    """

    column: int
    associations: list[AssocResult]
    n_snps_tested: int
    assoc_path: Path
    timing: ChunkRunStats = field(default_factory=ChunkRunStats)
    pve_estimate: float | None = None
    pve_se: float | None = None


@dataclass
class PipelineResult:
    """Result of a pipeline run.

    Attributes:
        associations: Per-SNP association results. Empty when results are
            written to disk via output_path.
        n_samples: Number of samples after phenotype and covariate filtering.
        n_snps_tested: Number of SNPs tested after MAF/missingness/HWE/SNP-list
            filtering.
        assoc_path: Path to the written association results file. For multi-phenotype
            runs, this is the last phenotype's output file. Use assoc_paths for
            the full list.
        assoc_paths: List of all per-phenotype association result paths. For
            single-phenotype runs, this is a single-element list matching assoc_path.
        phenotype_results: One result record per phenotype, including its output,
            count, PVE estimate, and chunk timing.
        timing: Timing breakdown by pipeline phase (seconds).
        n_covariates: Number of covariate columns (1 = intercept-only).
        pve_estimate: PVE from the single phenotype's null model REML. None for
            multi-phenotype runs; use phenotype_results for those estimates.
        pve_se: Standard error of PVE from REML second derivative delta method.
            None if not computed or likelihood surface is flat.
    """

    associations: list[AssocResult]
    n_samples: int
    n_snps_tested: int
    assoc_path: Path
    assoc_paths: list[Path] = field(default_factory=list)
    timing: PipelineTiming = field(default_factory=PipelineTiming)
    n_covariates: int = 1
    pve_estimate: float | None = None
    pve_se: float | None = None
    phenotype_results: list[PhenotypeResult] = field(default_factory=list)


@dataclass
class KinshipResult:
    """Outcome of a kinship computation (the ``-gk`` path).

    Returned by ``PipelineRunner.compute_kinship`` so the CLI can write its
    GEMMA log and summary without owning the compute/write orchestration.

    Attributes:
        kinship_paths: Written kinship matrix paths. One entry for a standard
            run; one per chromosome for LOCO.
        eigen_paths: ``(eigenvalue_path, eigenvector_path)`` when ``write_eigen``
            was set, else None. Always None for LOCO.
        n_samples: Sample count of the computed kinship matrix.
        n_snps: Total SNP count from PLINK metadata.
        mode: Kinship mode (1=centered, 2=standardized).
        is_loco: True when per-chromosome LOCO matrices were written.
        kinship_s: Wall time spent computing (and, for LOCO, writing) the
            kinship matrix.
    """

    kinship_paths: list[Path]
    eigen_paths: tuple[Path, Path] | None
    n_samples: int
    n_snps: int
    mode: int
    is_loco: bool
    kinship_s: float
