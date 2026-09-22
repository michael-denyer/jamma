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

import numpy as np

from jamma.lmm.assoc_output import AssocResult
from jamma.lmm.schema import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_MAF,
    DEFAULT_MISS,
    DEFAULT_N_GRID,
    DEFAULT_N_REFINE,
    ChunkRunStats,
    LmmConfig,
    parse_lmm_mode,
)

BackendRequest = Literal["auto", "numpy", "numpy-streaming"]
VALID_BACKENDS: tuple[BackendRequest, ...] = ("auto", "numpy", "numpy-streaming")


@dataclass
class PipelineTiming:
    """Timing breakdown from pipeline execution.

    All fields default to 0.0; fields from the runner are merged at
    pipeline exit.

    Attributes:
        kinship_s: Kinship load/compute time (seconds).
        load_s: Total data loading time through kinship (seconds).
        lmm_s: LMM association runtime (seconds).
        total_s: Total pipeline wall time (seconds).
        rotation_s: UT@G rotation time from the runner (seconds).
    """

    kinship_s: float = 0.0
    load_s: float = 0.0
    lmm_s: float = 0.0
    total_s: float = 0.0
    rotation_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ProvidedEigen:
    eigenvalue_file: Path
    eigenvector_file: Path
    ignored_kinship_file: Path | None


@dataclass(frozen=True, slots=True)
class ProvidedKinship:
    path: Path


@dataclass(frozen=True, slots=True)
class GenotypeKinship:
    """Whole-genome kinship computed from the genotypes."""


@dataclass(frozen=True, slots=True)
class LocoKinship:
    eigen_dir: Path | None


AnalysisSource = ProvidedEigen | ProvidedKinship | GenotypeKinship | LocoKinship


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
        weight_file: Individual residual weights for kinship and observations.
            One weight per line, matching sample order. Scales kinship before
            eigendecomposition, then scales eigenvector rows for association.
            GEMMA's -widv flag.
        cat_columns: 1-indexed covariate column indices to treat as
            categorical. JAMMA-specific feature (not GEMMA's -cat which is
            for SNP categories in VC mode). Columns are one-hot encoded with
            the first sorted level dropped as reference.
        backend: Compute backend selection: "auto" (default), "numpy", or
            "numpy-streaming". "auto" selects based on C extension availability
            and memory. "numpy" forces the batch runner, and "numpy-streaming"
            the runner that streams genotypes from disk.
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
        # LOCO writes a per-chromosome eigen cache keyed by eigen_dir; without
        # a directory it lands in output_dir. The non-LOCO write_eigen path
        # writes to output_dir directly and never consults eigen_dir.
        if self.loco and self.write_eigen and self.eigen_dir is None:
            object.__setattr__(self, "eigen_dir", self.output_dir)
        self.source()

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

    def source(self) -> AnalysisSource:
        """Parse the kinship and eigen fields into the one source they name.

        Returns:
            The eigen files, kinship file, genotype kinship, or LOCO kinship
            the run reads.

        Raises:
            ValueError: If the kinship, eigen, weight, and LOCO fields
                combine illegally.
        """
        d, u = self.eigenvalue_file, self.eigenvector_file
        if self.loco and self.kinship_file is not None:
            raise ValueError(
                "-k and -loco are mutually exclusive in this version. "
                "LOCO computes kinship internally."
            )
        if (d is None) != (u is None):
            raise ValueError(
                "Both -d (eigenvalues) and -u (eigenvectors) must be provided together"
            )
        if d is not None and self.loco:
            raise ValueError(
                "-d/-u (pre-computed eigen) not supported with -loco mode. "
                "Use --eigen-dir for per-chromosome eigen caching."
            )
        if self.weight_file is not None and self.loco:
            raise ValueError(
                "-widv (individual weights) is not yet supported with -loco mode. "
                "Apply weights to pre-computed kinship and use -k instead."
            )
        if self.weight_file is not None and d is not None:
            raise ValueError(
                "-widv (individual weights) cannot be used with -d/-u "
                "(pre-computed eigen). "
                "Weights must be applied to kinship before eigendecomposition."
            )
        if self.loco:
            return LocoKinship(self.eigen_dir)
        if self.eigen_dir is not None:
            raise ValueError("--eigen-dir is only supported with -loco mode")
        if d is not None and u is not None:
            return ProvidedEigen(d, u, self.kinship_file)
        if self.kinship_file is not None:
            return ProvidedKinship(self.kinship_file)
        return GenotypeKinship()

    def lmm_config(self, *, check_memory: bool = False) -> LmmConfig:
        """Project the LMM knobs onto the config the runners take.

        The one place these fields are mapped onto LmmConfig — every
        dispatch path goes through here, so a knob added to LmmConfig cannot
        reach one runner and miss another.

        Built fresh on each call so the pipeline can choose whether the runner
        repeats the memory gate without duplicating the field mapping.

        Args:
            check_memory: Whether the runner should run its own memory gate.

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

    Every per-phenotype aggregate is derived from ``phenotype_results``, so a
    new per-phenotype field needs adding in one place.

    Attributes:
        phenotype_results: One result record per phenotype, in column order,
            including its output, count, PVE estimate, and chunk timing.
        n_samples: Number of samples after phenotype and covariate filtering.
        timing: Timing breakdown by pipeline phase (seconds).
        n_covariates: Number of covariate columns (1 = intercept-only).
        analyzed_sample_indices: Zero-based input sample indices retained after
            phenotype and covariate filtering, in analysis order.
    """

    phenotype_results: list[PhenotypeResult]
    n_samples: int
    timing: PipelineTiming = field(default_factory=PipelineTiming)
    n_covariates: int = 1
    analyzed_sample_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.intp)
    )

    @property
    def associations(self) -> list[AssocResult]:
        """Per-SNP results of every phenotype. Empty when written to disk."""
        return [a for p in self.phenotype_results for a in p.associations]

    @property
    def n_snps_tested(self) -> int:
        """SNPs tested after MAF, missingness, HWE and SNP-list filtering."""
        return sum(p.n_snps_tested for p in self.phenotype_results)

    @property
    def assoc_path(self) -> Path:
        """The last phenotype's output file; ``assoc_paths`` lists them all."""
        return self.phenotype_results[-1].assoc_path

    @property
    def assoc_paths(self) -> list[Path]:
        """Every phenotype's output file, in column order."""
        return [p.assoc_path for p in self.phenotype_results]

    @property
    def pve_estimate(self) -> float | None:
        """The single phenotype's REML PVE, or None for multi-phenotype runs."""
        return self._single.pve_estimate if self._single is not None else None

    @property
    def pve_se(self) -> float | None:
        """Standard error of ``pve_estimate``, or None when it has none."""
        return self._single.pve_se if self._single is not None else None

    @property
    def _single(self) -> PhenotypeResult | None:
        return self.phenotype_results[0] if len(self.phenotype_results) == 1 else None


@dataclass
class KinshipResult:
    """Outcome of a kinship computation (the ``-gk`` path).

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
