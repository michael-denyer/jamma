"""Pipeline orchestration for JAMMA GWAS analysis.

Provides a single PipelineRunner service class that encapsulates the shared
GWAS pipeline: validate inputs, parse phenotypes, check memory, load kinship,
load covariates, run LMM association. Both the CLI (cli.py) and Python API
(gwas.py) delegate to this runner.

Example:
    >>> from jamma.pipeline import PipelineConfig, PipelineRunner
    >>> config = PipelineConfig(bfile=Path("data/study"), kinship_file=Path("k.txt"))
    >>> result = PipelineRunner(config).run()
    >>> print(f"Tested {result.n_snps_tested} SNPs")
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.backend import (
    BackendRequest,
    BackendResolved,
    format_pipeline_banner,
)
from jamma.core.constants import PHENOTYPE_MISSING
from jamma.core.memory import (
    StreamingMemoryBreakdown,
    estimate_streaming_memory,
)
from jamma.io.covariate import read_covariate_file
from jamma.io.plink import get_plink_metadata, validate_plink_dimensions
from jamma.io.snp_list import read_snp_list_file, resolve_snp_list_to_indices
from jamma.kinship import (
    compute_kinship_streaming,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm.chunk import _compute_chunk_size
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.runner import ExecutionPlan, select_execution_mode
from jamma.lmm.schema import LmmConfig, LmmRunResult, PipelineTiming
from jamma.lmm.stats import AssocResult


@dataclass
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
        phenotype_column: 1-based phenotype column index. 1 selects column 6
            of .fam (standard phenotype), 2 selects column 7, etc. Matches
            GEMMA's -n flag.
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
        n_refine: Golden section refinement iterations (default 10).
        weight_file: Individual weight file for kinship pre-transformation.
            One weight per line, matching sample order. Applies
            K[i,j] /= sqrt(w_i * w_j) before eigendecomposition.
            GEMMA's -widv flag.
        cat_columns: 1-indexed covariate column indices to treat as
            categorical. JAMMA-specific feature (not GEMMA's -cat which is
            for SNP categories in VC mode). Columns are one-hot encoded with
            the first sorted level dropped as reference.
        profile_dir: Directory for JAX XLA profiling traces. None disables
            profiling. When set, wraps the pipeline in jax.profiler.trace()
            and annotates stages with TraceAnnotation. View traces with
            `tensorboard --logdir <profile_dir>`.
        backend: Compute backend selection: "auto" (default), "jax", or "numpy".
            "auto" uses JAX when installed, falling back to NumPy. "jax" requires
            JAX to be installed. "numpy" forces the pure-NumPy backend.
        legacy_text: If True, write kinship and eigen files in GEMMA text format
            (.cXX.txt / .eigenD.txt / .eigenU.txt) instead of binary .npy.
            Default False writes binary for performance at scale.
        phenotype_columns: List of 1-based phenotype column indices, or None to
            derive from phenotype_column. When multiple columns are specified,
            eigendecomposition is computed once and reused. Mutually exclusive
            with loco mode for multiple columns.
    """

    bfile: Path
    kinship_file: Path | None = None
    covariate_file: Path | None = None
    lmm_mode: int = 1
    maf: float = 0.01
    miss: float = 0.05
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
    phenotype_column: int = 1
    snps_file: Path | None = None
    ksnps_file: Path | None = None
    hwe_threshold: float = 0.0
    l_min: float = 1e-5
    l_max: float = 1e5
    n_grid: int = 50
    n_refine: int = 10
    weight_file: Path | None = None
    cat_columns: list[int] | None = None
    profile_dir: Path | None = None
    backend: BackendRequest = "auto"
    legacy_text: bool = False
    phenotype_columns: list[int] | None = None

    def __post_init__(self) -> None:
        if os.sep in self.output_prefix or "/" in self.output_prefix:
            raise ValueError(
                f"output_prefix must not contain path separators, "
                f"got '{self.output_prefix}'. Use output_dir for directory paths."
            )
        if self.backend not in ("auto", "jax", "numpy"):
            raise ValueError(
                f"backend must be 'auto', 'jax', or 'numpy', got {self.backend!r}"
            )
        # Derive phenotype_columns from phenotype_column if not set
        if self.phenotype_columns is None:
            self.phenotype_columns = [self.phenotype_column]
        # Keep phenotype_column in sync as first element
        self.phenotype_column = self.phenotype_columns[0]
        # Validate no duplicates
        if len(self.phenotype_columns) != len(set(self.phenotype_columns)):
            raise ValueError(
                f"phenotype_columns contains duplicate indices: "
                f"{self.phenotype_columns}"
            )
        # LOCO + multi-phenotype guard
        if self.loco and len(self.phenotype_columns) > 1:
            raise ValueError(
                "LOCO mode (-loco) does not support multi-phenotype "
                "(-n with multiple columns). "
                "Run each phenotype separately."
            )


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
        timing: Timing breakdown by pipeline phase (seconds).
        backend: The compute backend used ("jax" or "numpy").
        n_covariates: Number of covariate columns (1 = intercept-only).
        pve_estimate: PVE (proportion of variance explained) from null model REML.
            None if not computed (e.g. LOCO with per-chromosome eigendecomp).
        pve_se: Standard error of PVE from REML second derivative delta method.
            None if not computed or likelihood surface is flat.
    """

    associations: list[AssocResult]
    n_samples: int
    n_snps_tested: int
    assoc_path: Path
    assoc_paths: list[Path] = field(default_factory=list)
    timing: PipelineTiming = field(default_factory=dict)
    backend: BackendResolved = "numpy"  # Set by PipelineRunner.run()
    n_covariates: int = 1
    pve_estimate: float | None = None
    pve_se: float | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("jax", "numpy"):
            raise ValueError(
                f"PipelineResult.backend must be 'jax' or 'numpy', got {self.backend!r}"
            )


class PipelineRunner:
    """Orchestrates a complete GWAS pipeline run.

    Encapsulates the shared pipeline logic used by both the CLI and
    Python API: validate inputs, parse phenotypes, check memory, load
    kinship, load covariates, run LMM association.

    Raises exceptions (ValueError, FileNotFoundError, MemoryError)
    rather than calling sys.exit or click.ClickException. The CLI wrapper catches
    these and converts to user-friendly error messages.

    Args:
        config: Pipeline configuration.

    Example:
        >>> config = PipelineConfig(bfile=Path("data/study"))
        >>> runner = PipelineRunner(config)
        >>> result = runner.run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @staticmethod
    def _compute_valid_mask(
        phenotypes: np.ndarray, covariates: np.ndarray | None
    ) -> np.ndarray:
        """Compute boolean mask of samples with valid phenotype and covariate values."""
        from jamma.lmm.prepare_common import compute_valid_mask

        return compute_valid_mask(phenotypes, covariates)

    def validate_inputs(self) -> None:
        """Validate that all required input files exist and parameters are valid.

        Raises:
            FileNotFoundError: If PLINK files (.bed, .bim, .fam) are missing,
                or if kinship_file/covariate_file is specified but missing.
            ValueError: If lmm_mode is not in (1, 2, 3, 4).
        """
        bfile = self.config.bfile
        for ext in (".bed", ".bim", ".fam"):
            p = Path(f"{bfile}{ext}")
            if not p.exists():
                raise FileNotFoundError(f"PLINK {ext} file not found: {p}")

        # Validate .bed file size matches .fam/.bim dimensions (VALID-01)
        validate_plink_dimensions(bfile)

        if self.config.phenotype_column < 1:
            raise ValueError(
                f"phenotype_column must be >= 1 (1-based), "
                f"got {self.config.phenotype_column}"
            )

        if self.config.lmm_mode not in (1, 2, 3, 4):
            raise ValueError(
                f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), "
                f"got {self.config.lmm_mode}"
            )

        if self.config.loco and self.config.kinship_file is not None:
            raise ValueError(
                "-k and -loco are mutually exclusive in this version. "
                "LOCO computes kinship internally."
            )

        # Eigen file validation: -d and -u must be paired
        has_eigen = self.config.eigenvalue_file is not None
        has_eigenvec = self.config.eigenvector_file is not None
        if has_eigen != has_eigenvec:
            raise ValueError(
                "Both -d (eigenvalues) and -u (eigenvectors) must be provided together"
            )

        if has_eigen:
            if self.config.loco:
                raise ValueError(
                    "-d/-u (pre-computed eigen) not supported with -loco mode. "
                    "Use --eigen-dir for per-chromosome eigen caching."
                )
            if not self.config.eigenvalue_file.exists():
                raise FileNotFoundError(
                    f"Eigenvalue file not found: {self.config.eigenvalue_file}"
                )
            if not self.config.eigenvector_file.exists():
                raise FileNotFoundError(
                    f"Eigenvector file not found: {self.config.eigenvector_file}"
                )

        if (
            self.config.kinship_file is not None
            and not self.config.kinship_file.exists()
        ):
            raise FileNotFoundError(
                f"Kinship matrix file not found: {self.config.kinship_file}"
            )

        if (
            self.config.covariate_file is not None
            and not self.config.covariate_file.exists()
        ):
            raise FileNotFoundError(
                f"Covariate file not found: {self.config.covariate_file}"
            )

        # SNP list file validation
        if self.config.snps_file is not None and not self.config.snps_file.exists():
            raise FileNotFoundError(f"SNP list file not found: {self.config.snps_file}")

        if self.config.ksnps_file is not None and not self.config.ksnps_file.exists():
            raise FileNotFoundError(
                f"Kinship SNP list file not found: {self.config.ksnps_file}"
            )

        # Lambda bounds validation
        if self.config.l_min <= 0:
            raise ValueError(f"l_min must be > 0, got {self.config.l_min}")
        if self.config.l_max <= self.config.l_min:
            raise ValueError(
                f"l_max must be > l_min ({self.config.l_min}), got {self.config.l_max}"
            )

        # Weight file validation
        if self.config.weight_file is not None and not self.config.weight_file.exists():
            raise FileNotFoundError(f"Weight file not found: {self.config.weight_file}")
        if self.config.weight_file is not None and self.config.loco:
            raise ValueError(
                "-widv (individual weights) is not yet supported with -loco mode. "
                "Apply weights to pre-computed kinship and use -k instead."
            )
        if (
            self.config.weight_file is not None
            and self.config.eigenvalue_file is not None
        ):
            raise ValueError(
                "-widv (individual weights) cannot be used with -d/-u "
                "(pre-computed eigen). "
                "Weights must be applied to kinship before eigendecomposition."
            )

        # Categorical columns validation
        if self.config.cat_columns is not None:
            if self.config.covariate_file is None:
                raise ValueError("-cat requires -c (covariate file)")
            for col in self.config.cat_columns:
                if col < 1:
                    raise ValueError(
                        f"-cat column indices must be >= 1 (1-indexed), got {col}"
                    )

        # HWE threshold validation
        if self.config.hwe_threshold < 0:
            raise ValueError(
                f"hwe_threshold must be >= 0, got {self.config.hwe_threshold}"
            )
        if self.config.hwe_threshold > 1.0:
            raise ValueError(
                f"hwe_threshold must be in [0, 1] (p-value threshold), "
                f"got {self.config.hwe_threshold}"
            )
        if self.config.hwe_threshold > 0 and self.config.loco:
            raise ValueError(
                "-hwe is not yet supported with -loco mode. "
                "Apply HWE filtering as a pre-processing step."
            )

    def _parse_phenotype_column(
        self, pheno_col: int, *, fam_data: np.ndarray | None = None
    ) -> tuple[np.ndarray, int]:
        """Parse a specific phenotype column from the .fam file.

        Args:
            pheno_col: 1-based phenotype column index.
            fam_data: Pre-loaded .fam file data as string array. If None,
                reads from disk (for backward compatibility).

        Returns:
            Tuple of (phenotypes array, n_analyzed) where phenotypes has
            NaN for missing values and n_analyzed is the count of valid
            (non-NaN, non-missing) phenotypes.

        Raises:
            ValueError: If no samples have valid phenotypes, or if
                pheno_col is invalid.
        """
        if pheno_col < 1:
            raise ValueError(
                f"phenotype_column must be >= 1 (1-based), got {pheno_col}"
            )

        # Columns 0-4 are FID, IID, father, mother, sex
        col_index = 4 + pheno_col
        fam_path = f"{self.config.bfile}.fam"

        if fam_data is None:
            try:
                all_data = np.loadtxt(fam_path, dtype=str, ndmin=2)
            except (ValueError, OSError) as e:
                raise ValueError(f"Failed to read .fam file {fam_path}: {e}") from e
        else:
            all_data = fam_data

        n_cols = all_data.shape[1]
        if col_index >= n_cols:
            n_pheno_cols = n_cols - 5
            raise ValueError(
                f"phenotype_column {pheno_col} exceeds available columns "
                f"in .fam file ({n_pheno_cols} phenotype column"
                f"{'s' if n_pheno_cols != 1 else ''} available)"
            )

        logger.info(f"Using phenotype column {pheno_col} (file column {col_index + 1})")

        fam_data = all_data[:, col_index]
        missing_mask = np.isin(fam_data, ["-9", "NA"])
        fam_data[missing_mask] = "0"
        phenotypes = fam_data.astype(np.float64)
        phenotypes[missing_mask] = np.nan

        valid_mask = ~np.isnan(phenotypes) & (phenotypes != PHENOTYPE_MISSING)
        n_analyzed = int(valid_mask.sum())

        if n_analyzed == 0:
            raise ValueError("No samples with valid phenotypes")

        return phenotypes, n_analyzed

    def parse_phenotypes(self) -> tuple[np.ndarray, int]:
        """Parse phenotypes from the .fam file.

        Uses vectorized parsing: reads the phenotype column, replaces
        missing indicators ("-9", "NA") with NaN, converts to float64.
        The column is selected by ``self.config.phenotype_column``
        (1-based, matching GEMMA's ``-n`` flag).

        Returns:
            Tuple of (phenotypes array, n_analyzed) where phenotypes has
            NaN for missing values and n_analyzed is the count of valid
            (non-NaN, non-missing) phenotypes.

        Raises:
            ValueError: If no samples have valid phenotypes, or if
                phenotype_column is invalid.
        """
        return self._parse_phenotype_column(self.config.phenotype_column)

    def check_memory_requirements(
        self, n_samples: int, n_snps: int, n_cvt: int = 1
    ) -> StreamingMemoryBreakdown | None:
        """Check memory requirements if memory checking is enabled.

        Computes actual chunk size via _compute_chunk_size, then estimates
        streaming memory. Checks against mem_budget if set, and against
        available system memory.

        Args:
            n_samples: Number of valid samples (after phenotype/covariate filtering).
            n_snps: Number of SNPs in the dataset.
            n_cvt: Number of covariates (affects Uab array sizing).

        Returns:
            StreamingMemoryBreakdown if check_memory is True, None otherwise.

        Raises:
            MemoryError: If estimated memory exceeds budget or available memory.
        """
        if not self.config.check_memory:
            return None

        disk_chunk = _compute_chunk_size(n_snps)
        jax_chunk = _compute_chunk_size(n_snps, n_samples=n_samples, pipeline_buffers=2)
        est = estimate_streaming_memory(
            n_samples,
            chunk_size=disk_chunk,
            n_cvt=n_cvt,
            jax_chunk_size=jax_chunk,
        )

        logger.info(
            f"Memory estimate: {est.total_peak_gb:.1f}GB required, "
            f"{est.available_gb:.1f}GB available"
        )

        if (
            self.config.mem_budget is not None
            and est.total_peak_gb > self.config.mem_budget
        ):
            raise MemoryError(
                f"Estimated memory ({est.total_peak_gb:.1f}GB) exceeds "
                f"budget ({self.config.mem_budget}GB). "
                f"Use --no-check-memory to override."
            )

        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory: need {est.total_peak_gb:.1f}GB "
                f"(with 10% margin), have {est.available_gb:.1f}GB. "
                f"Use --no-check-memory to override."
            )

        return est

    def load_kinship(
        self,
        n_samples: int,
        ksnps_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Load or compute the kinship matrix.

        If kinship_file is provided, loads from disk. Otherwise, computes
        from genotypes using streaming kinship computation.

        If weight_file is configured, applies individual weights to K via
        K[i,j] /= sqrt(w_i * w_j) before returning (and before saving).

        If save_kinship is True, writes the kinship matrix to the
        output directory (whether loaded or computed).

        Args:
            n_samples: Number of samples (for validation of loaded kinship).
            ksnps_indices: Optional SNP indices to restrict kinship computation.
                Ignored when loading pre-computed kinship from file.

        Returns:
            Kinship matrix of shape (n_samples, n_samples).
        """
        if self.config.kinship_file is not None:
            logger.info(f"Loading kinship from {self.config.kinship_file}")
            K = read_kinship_matrix(self.config.kinship_file, n_samples=n_samples)
        else:
            logger.info("Computing kinship from genotypes")
            K = compute_kinship_streaming(
                self.config.bfile,
                check_memory=False,
                show_progress=self.config.show_progress,
                ksnps_indices=ksnps_indices,
            )

        # Apply individual weights before eigendecomposition
        if self.config.weight_file is not None:
            from jamma.io.weight import apply_individual_weights, read_weight_file

            weights = read_weight_file(self.config.weight_file)
            if len(weights) != n_samples:
                raise ValueError(
                    f"Weight file has {len(weights)} entries but expected "
                    f"{n_samples} (matching sample count)"
                )
            logger.info(f"Applying individual weights from {self.config.weight_file}")
            K = apply_individual_weights(K, weights)

        if self.config.save_kinship:
            kinship_base = (
                self.config.output_dir / f"{self.config.output_prefix}.cXX.txt"
            )
            actual_path = write_kinship_matrix(
                K, kinship_base, legacy_text=self.config.legacy_text
            )
            logger.info(f"Kinship matrix saved to {actual_path}")

        return K

    def load_covariates(self, n_samples: int) -> np.ndarray | None:
        """Load and validate the covariate file.

        Args:
            n_samples: Number of samples for row-count validation.

        Returns:
            Covariate array of shape (n_samples, n_covariates), or None
            if no covariate file was specified.

        Raises:
            ValueError: If covariate row count does not match n_samples.
        """
        if self.config.covariate_file is None:
            return None

        logger.info(f"Loading covariates from {self.config.covariate_file}")
        covariates, _ = read_covariate_file(self.config.covariate_file)

        if covariates.shape[0] != n_samples:
            raise ValueError(
                f"Covariate file has {covariates.shape[0]} rows "
                f"but PLINK data has {n_samples} samples. "
                f"Covariate rows must match sample count exactly."
            )

        logger.info(f"Loaded {covariates.shape[1]} covariates")

        # Warn if first column is not an intercept
        first_col = covariates[:, 0]
        valid_first = first_col[~np.isnan(first_col)]
        if not np.allclose(valid_first, 1.0):
            logger.warning(
                "Warning: Covariate file does not have intercept column "
                "(first column is not all 1s). "
                "Model will NOT include intercept."
            )

        # Apply categorical encoding if -cat specified
        if self.config.cat_columns is not None:
            from jamma.io.covariate import encode_categorical_covariates

            covariates = encode_categorical_covariates(
                covariates, self.config.cat_columns
            )
            logger.info(
                f"Categorical encoding applied to columns {self.config.cat_columns}: "
                f"expanded to {covariates.shape[1]} covariate columns"
            )

        return covariates

    @staticmethod
    def _resolve_snp_list(
        snp_file: Path | None, sid_array: np.ndarray, label: str
    ) -> np.ndarray | None:
        """Resolve a SNP list file to column indices, or return None.

        Args:
            snp_file: Path to SNP list file, or None.
            sid_array: Array of SNP IDs from PLINK metadata.
            label: Label for log message (e.g. "-snps", "-ksnps").

        Returns:
            Sorted array of column indices, or None if snp_file is None.
        """
        if snp_file is None:
            return None
        snp_ids = read_snp_list_file(snp_file)
        indices = resolve_snp_list_to_indices(snp_ids, sid_array)
        logger.info(f"SNP list ({label}): {len(indices)} SNPs resolved")
        return indices

    @staticmethod
    def _log_banner(
        n_total: int,
        n_analyzed: int,
        n_snps: int,
        n_covariates: int = 1,
        n_phenotypes: int = 1,
    ) -> None:
        """Log GEMMA-style startup banner with dataset summary.

        Prints version, release date, and dataset dimensions to match
        GEMMA's startup output format for user familiarity.

        Args:
            n_total: Total number of individuals in the PLINK file.
            n_analyzed: Number of individuals after phenotype/covariate filtering.
            n_snps: Total number of SNPs in the dataset.
            n_covariates: Number of covariate columns (1 = intercept-only).
            n_phenotypes: Number of phenotype columns being analyzed.
        """
        import jamma

        logger.info(f"JAMMA v{jamma.__version__} ({jamma.__release_date__})")
        logger.info("Reading Files ...")
        logger.info(f"## number of total individuals = {n_total:,}")
        logger.info(f"## number of analyzed individuals = {n_analyzed:,}")
        logger.info(f"## number of covariates = {n_covariates}")
        logger.info(f"## number of phenotypes = {n_phenotypes}")
        logger.info(f"## number of total SNPs/var = {n_snps:,}")

    def _check_hwe_support(self, plan: ExecutionPlan) -> None:
        """Raise if HWE filtering requested but backend doesn't support it."""
        if self.config.hwe_threshold > 0 and plan.backend == "numpy":
            raise ValueError(
                "HWE filtering (--hwe) is not yet supported with the NumPy backend. "
                "Use the JAX backend (pip install jamma[jax]) or set --hwe 0."
            )

    @staticmethod
    def _log_pipeline_banner(
        plan: ExecutionPlan,
    ) -> None:
        """Emit a consolidated one-line pipeline configuration banner.

        Gathers runner type, BLAS backend, C extension status, and
        thread count into a single log line. The banner shows "pending"
        for the eigen driver; the actual driver is logged separately by
        eigendecompose_kinship once the matrix size is known.

        This method is purely diagnostic — failures are caught and logged
        as warnings to avoid aborting the GWAS pipeline.

        Args:
            plan: ExecutionPlan with backend and mode already decided.
        """
        try:
            from jamma.core.threading import (
                get_blas_backend,
                get_physical_core_count,
                is_blas_controllable,
            )
            from jamma.lmm._compile_utils import is_c_extension_usable

            c_ext = is_c_extension_usable()
            runner = plan.runner_name

            blas = get_blas_backend()

            # Respect JAMMA_BLAS_THREADS if set, otherwise use physical
            # core count. We avoid get_blas_thread_count() because it
            # imports jax unconditionally, crashing numpy-only installs.
            max_threads = os.cpu_count() or 64
            env_threads = os.environ.get("JAMMA_BLAS_THREADS")
            if env_threads is not None:
                try:
                    threads = max(1, min(int(env_threads), max_threads))
                except ValueError:
                    threads = get_physical_core_count()
            elif is_blas_controllable():
                threads = get_physical_core_count()
            else:
                # Accelerate or no BLAS — use halved core count
                # (same logic as runner_numpy.py for OpenMP).
                cores = get_physical_core_count()
                threads = max(1, cores // 2)

            jax_devices = 0
            if plan.backend == "jax":
                import jax

                from jamma.core.jax_config import is_jax_configured

                if is_jax_configured():
                    jax_devices = len(jax.devices())

            banner = format_pipeline_banner(
                runner=runner,
                blas=blas,
                eigen_driver="pending",
                c_ext=c_ext,
                threads=threads,
                jax_devices=jax_devices,
            )
            logger.info(banner)
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            logger.warning(f"Could not build pipeline banner: {exc}")

    def run(self) -> PipelineResult:
        """Execute the full GWAS pipeline.

        Pipeline steps:
        1. Validate inputs
        2. Get PLINK metadata
        3. Check memory requirements
        4. Parse phenotypes (all columns, compute mask intersection)
        5. Resolve SNP list files
        6. Prepare output directory
        7. Load covariates (early, for eigen validation)
        8. Load eigen files or kinship matrix (once, shared)
        9. Per-phenotype loop: run LMM association and write results
        10. Return aggregated PipelineResult

        Returns:
            PipelineResult with associations, counts, output path, and timing.
        """
        t_start = time.perf_counter()

        from jamma.core.backend import detect_backend, log_backend_selection

        # Resolve env override first: JAMMA_BACKEND takes priority in all paths.
        env_backend = os.environ.get("JAMMA_BACKEND")
        requested = env_backend if env_backend is not None else self.config.backend

        # Fail fast: HWE + explicit numpy is always invalid, before touching disk.
        if self.config.hwe_threshold > 0 and requested == "numpy":
            raise ValueError(
                "HWE filtering (--hwe) is not yet supported with the NumPy backend. "
                "Use the JAX backend (pip install jamma[jax]) or set --hwe 0."
            )

        # PLINK metadata is lightweight (reads .fam/.bim header only) and needed
        # for memory-based mode selection in both auto and explicit-JAX paths.
        from jamma.io.plink import get_plink_metadata as _get_meta

        _meta = _get_meta(self.config.bfile)

        if requested == "auto":
            plan = select_execution_mode(
                n_samples=_meta["n_samples"],
                n_snps=_meta["n_snps"],
                requested=requested,
            )
            active_backend = plan.backend
        else:
            active_backend = detect_backend(requested)
            # Build an ExecutionPlan for explicit backend requests.
            # NumPy always batch; JAX mode (batch vs streaming) determined by memory.
            if active_backend == "numpy":
                plan = ExecutionPlan(
                    "numpy", "batch", f"Backend '{requested}' explicitly requested"
                )
            else:
                plan = select_execution_mode(
                    n_samples=_meta["n_samples"],
                    n_snps=_meta["n_snps"],
                    requested=requested,
                )
                active_backend = plan.backend

        log_backend_selection(active_backend, self.config.backend, env_backend)
        logger.info(f"Execution plan: {plan.runner_name} ({plan.reason})")

        trace_ctx = contextlib.nullcontext()
        if active_backend == "jax":
            import jax

            # Enable x64 without initializing the JAX backend — device count is
            # deferred until LMM phase via ensure_jax_configured().
            jax.config.update("jax_enable_x64", True)

            # Optional XLA profiling — degrade gracefully so profiling issues
            # never prevent GWAS results from being produced.
            if self.config.profile_dir is not None:
                try:
                    self.config.profile_dir.mkdir(parents=True, exist_ok=True)
                    trace_ctx = jax.profiler.trace(
                        str(self.config.profile_dir), create_perfetto_link=False
                    )
                    logger.info(f"XLA profiling enabled: {self.config.profile_dir}")
                except (OSError, ImportError, AttributeError) as e:
                    logger.warning(f"Could not enable XLA profiling: {e}")
        elif self.config.profile_dir is not None:
            logger.warning("XLA profiling requires JAX backend; ignoring --profile-dir")

        with trace_ctx:
            return self._run_inner(t_start, plan, requested)

    def _run_inner(
        self,
        t_start: float,
        plan: ExecutionPlan,
        requested: BackendRequest = "auto",
    ) -> PipelineResult:
        """Execute the pipeline body, called within the optional profiling context.

        Args:
            t_start: Pipeline start time from time.perf_counter().
            plan: ExecutionPlan with backend, mode, and reason.
            requested: Resolved backend request (respects JAMMA_BACKEND env var).
        """
        self._check_hwe_support(plan)

        self.validate_inputs()

        meta = get_plink_metadata(self.config.bfile)
        n_samples = meta["n_samples"]
        n_snps = meta["n_snps"]

        snps_indices = self._resolve_snp_list(
            self.config.snps_file, meta["sid"], "-snps"
        )
        ksnps_indices = self._resolve_snp_list(
            self.config.ksnps_file, meta["sid"], "-ksnps"
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        assoc_path = self.config.output_dir / f"{self.config.output_prefix}.assoc.txt"

        # LOCO branch: skip standard kinship, run LOCO orchestrator
        # (single-phenotype only — guard in __post_init__)
        if self.config.loco:
            phenotypes, n_analyzed = self.parse_phenotypes()
            n_filtered = len(phenotypes) - n_analyzed
            logger.info(
                f"Analyzing {n_analyzed} samples with valid "
                f"phenotypes ({n_filtered} filtered)"
            )
            from jamma.lmm import run_lmm_loco

            covariates = self.load_covariates(n_samples)
            valid_mask = self._compute_valid_mask(phenotypes, covariates)
            n_valid = int(np.sum(valid_mask))
            n_cvt = covariates.shape[1] if covariates is not None else 1
            self._log_banner(n_samples, n_valid, n_snps, n_covariates=n_cvt)
            self._log_pipeline_banner(plan)

            t_loco = time.perf_counter()
            loco = run_lmm_loco(
                bed_path=self.config.bfile,
                phenotypes=phenotypes,
                covariates=covariates,
                maf_threshold=self.config.maf,
                miss_threshold=self.config.miss,
                lmm_mode=self.config.lmm_mode,
                output_path=assoc_path,
                check_memory=self.config.check_memory,
                show_progress=self.config.show_progress,
                save_kinship=self.config.save_kinship,
                kinship_output_dir=self.config.output_dir,
                kinship_output_prefix=self.config.output_prefix,
                snps_indices=snps_indices,
                ksnps_indices=ksnps_indices,
                l_min=self.config.l_min,
                l_max=self.config.l_max,
                backend=plan.backend,
                write_eigen=self.config.write_eigen,
                eigen_dir=self.config.eigen_dir,
                eigen_prefix=self.config.output_prefix,
            )
            loco_s = time.perf_counter() - t_loco
            total_s = time.perf_counter() - t_start
            logger.info(
                f"LOCO GWAS complete: {loco.n_tested} SNPs tested in {total_s:.1f}s"
            )

            return PipelineResult(
                associations=loco.associations,
                n_samples=n_valid,
                n_snps_tested=loco.n_tested,
                assoc_path=assoc_path,
                assoc_paths=[assoc_path],
                timing={
                    "kinship_s": 0.0,
                    "load_s": 0.0,
                    "lmm_s": loco_s,
                    "total_s": total_s,
                },
                backend=plan.backend,
                n_covariates=covariates.shape[1] if covariates is not None else 1,
                pve_estimate=loco.pve,
                pve_se=loco.pve_se,
            )

        covariates = self.load_covariates(n_samples)

        # Compute valid mask as intersection across all phenotype columns.
        # This ensures eigendecomposition uses the same sample set for all
        # phenotypes. Load .fam data once and extract all columns.
        pheno_columns = self.config.phenotype_columns
        fam_path = f"{self.config.bfile}.fam"
        try:
            fam_data = np.loadtxt(fam_path, dtype=str, ndmin=2)
        except (ValueError, OSError) as e:
            raise ValueError(f"Failed to read .fam file {fam_path}: {e}") from e

        all_pheno_data: dict[int, tuple[np.ndarray, int]] = {}
        all_masks: list[np.ndarray] = []
        for col in pheno_columns:
            pheno, n_anal = self._parse_phenotype_column(col, fam_data=fam_data)
            all_pheno_data[col] = (pheno, n_anal)
            mask = self._compute_valid_mask(pheno, covariates)
            all_masks.append(mask)

        # Intersect all masks for shared eigendecomposition
        valid_mask = np.all(all_masks, axis=0)
        n_valid = int(np.sum(valid_mask))

        if n_valid == 0:
            per_pheno_counts = {
                col: int(m.sum())
                for col, m in zip(pheno_columns, all_masks, strict=True)
            }
            raise ValueError(
                f"No samples have valid values across all {len(pheno_columns)} "
                f"phenotype columns. Per-column valid counts: {per_pheno_counts}"
            )

        # Warn if intersection excludes samples valid in some phenotypes
        per_pheno_counts = [int(m.sum()) for m in all_masks]
        if n_valid < min(per_pheno_counts):
            logger.warning(
                f"Sample mask intersection reduced valid samples: "
                f"per-phenotype counts {per_pheno_counts}, "
                f"intersection {n_valid}"
            )

        n_cvt = covariates.shape[1] if covariates is not None else 1
        is_multi = len(pheno_columns) > 1
        self._log_banner(
            n_samples,
            n_valid,
            n_snps,
            n_covariates=n_cvt,
            n_phenotypes=len(pheno_columns),
        )

        # Re-evaluate the plan with actual n_valid (initial plan may have used
        # raw n_samples from PLINK header; valid_mask filtering can reduce it).
        initial_plan = plan
        plan = select_execution_mode(n_valid, n_snps, requested=requested)
        if plan != initial_plan:
            # Mode changes (batch↔streaming) are expected when sample filtering
            # reduces n_valid. Backend changes (jax↔numpy) are not safe because
            # run() already configured the trace context for the initial backend.
            if plan.backend != initial_plan.backend:
                raise RuntimeError(
                    f"Backend changed from {initial_plan.backend} to "
                    f"{plan.backend} after sample filtering ({n_valid} valid "
                    f"samples). Trace context was configured for "
                    f"{initial_plan.backend}. Use an explicit --backend to "
                    f"avoid this."
                )
            logger.info(
                f"Execution plan changed after sample filtering: "
                f"{initial_plan.runner_name} -> {plan.runner_name} ({plan.reason})"
            )
            self._check_hwe_support(plan)
        else:
            logger.debug(
                f"Execution plan (post-filter): {plan.runner_name} ({plan.reason})"
            )

        # Banner after re-evaluation so it shows the final plan
        self._log_pipeline_banner(plan)

        # Memory preflight: streaming estimate for JAX streaming, in-memory for batch.
        # NOTE: actual_chunk must be set AFTER re-evaluation — if the plan changed
        # from batch to streaming, we need a valid chunk size.
        if plan.mode == "streaming":
            actual_chunk = _compute_chunk_size(n_snps)
            self.check_memory_requirements(n_valid, n_snps, n_cvt=n_cvt)
        else:
            from jamma.core.memory import estimate_lmm_memory

            if self.config.check_memory:
                est = estimate_lmm_memory(n_valid, n_snps)
                logger.info(
                    f"Memory estimate ({plan.runner_name}): "
                    f"{est.total_gb:.1f}GB required, "
                    f"{est.available_gb:.1f}GB available"
                )
                exceeds_budget = (
                    self.config.mem_budget is not None
                    and est.total_gb > self.config.mem_budget
                )
                if exceeds_budget:
                    raise MemoryError(
                        f"Estimated memory ({est.total_gb:.1f}GB) exceeds "
                        f"budget ({self.config.mem_budget}GB). "
                        f"Use --no-check-memory to override."
                    )
                if not est.sufficient:
                    raise MemoryError(
                        f"Insufficient memory: "
                        f"need {est.total_gb:.1f}GB, "
                        f"have {est.available_gb:.1f}GB. "
                        f"Use --no-check-memory to override."
                    )
            actual_chunk = None  # Not used by batch runners

        # Load/compute eigendecomposition ONCE (shared across phenotypes)
        t_kinship = time.perf_counter()
        eigenvalues = None
        eigenvectors = None

        if self.config.eigenvalue_file and self.config.eigenvector_file:
            eigenvalues, eigenvectors = read_eigen_files(
                self.config.eigenvalue_file,
                self.config.eigenvector_file,
                n_samples=n_valid,
            )
            logger.info(
                f"Loaded pre-computed eigendecomposition "
                f"({len(eigenvalues)} eigenvalues)"
            )
            if self.config.kinship_file:
                logger.warning(
                    "Both kinship (-k) and eigen files (-d/-u) "
                    "provided. Using eigen files; kinship will "
                    "be ignored."
                )
            K = None
        else:
            K = self.load_kinship(n_samples, ksnps_indices=ksnps_indices)
            # Eigendecompose once at the pipeline level so multi-phenotype
            # loops reuse the same eigendecomposition.
            K_valid = K if np.all(valid_mask) else K[np.ix_(valid_mask, valid_mask)]
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K_valid, check_memory=self.config.check_memory
            )
            if self.config.write_eigen:
                d_path, u_path = write_eigen_files(
                    eigenvalues,
                    eigenvectors,
                    self.config.output_dir,
                    self.config.output_prefix,
                    legacy_text=self.config.legacy_text,
                )
                logger.info(f"Wrote eigenvalues to {d_path}")
                logger.info(f"Wrote eigenvectors to {u_path}")
            K = None  # Runner uses eigen directly

        kinship_s = time.perf_counter() - t_kinship
        load_s = time.perf_counter() - t_start

        # Per-phenotype LMM loop
        t_lmm = time.perf_counter()
        all_results: list[AssocResult] = []
        total_tested = 0
        all_assoc_paths: list[Path] = []

        # Pre-load PLINK data once for batch multi-phenotype runs
        _plink_data = None
        if plan.mode == "batch" and len(pheno_columns) > 1:
            from jamma.io import load_plink_binary

            hint = (
                " (for large datasets, use JAX streaming: pip install jamma[jax])"
                if plan.backend == "numpy"
                else ""
            )
            logger.info(f"{plan.runner_name}: loading all genotypes into memory{hint}")
            _plink_data = load_plink_binary(self.config.bfile)

        prefix = self.config.output_prefix
        for col in pheno_columns:
            if is_multi:
                logger.info(f"Starting LMM for phenotype column {col}")
            # Mark samples outside the shared intersection as NaN so the
            # runner computes the same valid_mask used for eigendecomposition.
            # We pass full-length arrays (not pre-filtered) because the
            # streaming runner indexes genotypes streamed from disk using
            # the mask it computes internally.
            phenotypes_col = all_pheno_data[col][0].copy()
            phenotypes_col[~valid_mask] = np.nan

            if is_multi:
                col_path = self.config.output_dir / f"{prefix}.pheno{col}.assoc.txt"
            else:
                col_path = assoc_path

            if plan.backend == "jax":
                run_result, n_tested = self._run_jax_backend(
                    phenotypes_col,
                    K,
                    covariates,
                    eigenvalues,
                    eigenvectors,
                    col_path,
                    snps_indices,
                    actual_chunk,
                    plan=plan,
                    plink_data=_plink_data,
                    clear_caches=False,
                )
            else:
                run_result, n_tested = self._run_numpy_backend(
                    phenotypes_col,
                    K,
                    covariates,
                    eigenvalues,
                    eigenvectors,
                    col_path,
                    snps_indices,
                    plink_data=_plink_data,
                )

            all_results.extend(run_result.associations)
            total_tested += n_tested
            all_assoc_paths.append(col_path)
            logger.info(f"Phenotype {col}: {n_tested} SNPs tested -> {col_path}")

        # Clear JAX caches once after all phenotypes, not per-phenotype.
        # Compiled kernels have identical shapes across phenotypes so clearing
        # inside the loop forces redundant JIT recompilation.
        if plan.backend == "jax":
            try:
                import jax

                jax.clear_caches()
            except Exception:
                logger.warning(
                    "Failed to clear JAX caches after phenotype loop",
                    exc_info=True,
                )

        lmm_s = time.perf_counter() - t_lmm
        total_s = time.perf_counter() - t_start
        logger.info(f"GWAS complete: {total_tested} SNPs tested in {total_s:.1f}s")

        # Pull runner-level rotation timing (JAX backend only)
        runner_timing: dict[str, float] = {}
        if plan.backend == "jax" and plan.mode == "streaming":
            from jamma.lmm.runner_streaming import get_last_run_timing

            runner_timing = get_last_run_timing()
        elif plan.backend == "jax" and plan.mode == "batch":
            from jamma.lmm.runner_jax import last_run_timing as _jax_timing

            runner_timing = dict(_jax_timing)

        # Capture PVE from the most recent runner call
        pve = run_result.pve
        pve_se = run_result.pve_se

        return PipelineResult(
            associations=all_results,
            n_samples=n_valid,
            n_snps_tested=total_tested,
            assoc_path=all_assoc_paths[-1],
            assoc_paths=all_assoc_paths,
            timing={
                "kinship_s": kinship_s,
                "load_s": load_s,
                "lmm_s": lmm_s,
                "total_s": total_s,
                "rotation_s": runner_timing.get("rotation_s", 0.0),
                "rotation_exposed_s": runner_timing.get("rotation_exposed_s", 0.0),
            },
            backend=plan.backend,
            n_covariates=(covariates.shape[1] if covariates is not None else 1),
            pve_estimate=pve,
            pve_se=pve_se,
        )

    def _run_jax_backend(
        self,
        phenotypes: np.ndarray,
        K: np.ndarray | None,
        covariates: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        eigenvectors: np.ndarray | None,
        assoc_path: Path,
        snps_indices: np.ndarray | None,
        actual_chunk: int | None,
        *,
        plan: ExecutionPlan,
        plink_data: object | None = None,
        clear_caches: bool = True,
    ) -> tuple[LmmRunResult, int]:
        """Run LMM association using the JAX backend.

        Dispatches to batch or streaming runner based on plan.mode.

        Args:
            plink_data: Pre-loaded PLINK data for batch mode. If None and
                batch mode selected, loads from disk.
            plan: ExecutionPlan determining batch vs streaming mode.
            clear_caches: Clear JAX caches on exit. False for multi-phenotype
                loops where compiled kernels are reused across iterations.
        """
        from jamma.core.jax_config import ensure_jax_configured

        ensure_jax_configured()

        lmm_config = LmmConfig(
            maf_threshold=self.config.maf,
            miss_threshold=self.config.miss,
            l_min=self.config.l_min,
            l_max=self.config.l_max,
            n_grid=self.config.n_grid,
            n_refine=self.config.n_refine,
            check_memory=False,  # Already checked above
            show_progress=self.config.show_progress,
            lmm_mode=self.config.lmm_mode,
        )

        if plan.mode == "batch":
            from jamma.io import load_plink_binary
            from jamma.lmm import run_lmm_association_jax

            if plink_data is None:
                plink_data = load_plink_binary(self.config.bfile)

            genotypes = plink_data.genotypes
            indices = (
                snps_indices if snps_indices is not None else range(plink_data.n_snps)
            )
            if snps_indices is not None:
                genotypes = genotypes[:, snps_indices]
            snp_info = [
                {
                    "chr": str(plink_data.chromosome[i]),
                    "rs": plink_data.sid[i],
                    "pos": int(plink_data.bp_position[i]),
                    "a1": plink_data.allele_1[i],
                    "a0": plink_data.allele_2[i],
                }
                for i in indices
            ]

            run_result = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=K,
                snp_info=snp_info,
                covariates=covariates,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                config=lmm_config,
                output_path=assoc_path,
                clear_caches=clear_caches,
            )

            return run_result, run_result.snp_count

        # Streaming mode
        if actual_chunk is None:
            raise ValueError(
                "JAX streaming mode requires a chunk_size, but actual_chunk=None. "
                "This is a bug — the plan was streaming but no chunk size was "
                "computed."
            )
        from jamma.lmm import run_lmm_association_streaming

        return run_lmm_association_streaming(
            bed_path=self.config.bfile,
            phenotypes=phenotypes,
            kinship=K,
            snp_info=None,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            output_path=assoc_path,
            snps_indices=snps_indices,
            hwe_threshold=self.config.hwe_threshold,
            chunk_size=actual_chunk,
            config=lmm_config,
            clear_caches=clear_caches,
        )

    def _run_numpy_backend(
        self,
        phenotypes: np.ndarray,
        K: np.ndarray | None,
        covariates: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        eigenvectors: np.ndarray | None,
        assoc_path: Path,
        snps_indices: np.ndarray | None,
        plink_data: object | None = None,
    ) -> tuple[LmmRunResult, int]:
        """Run LMM association using the pure-NumPy batch backend.

        Args:
            plink_data: Pre-loaded PLINK data. If None, loads from disk.
                Pass this to avoid reloading genotypes in multi-phenotype runs.
        """
        from jamma.io import load_plink_binary
        from jamma.lmm import run_lmm_association_numpy

        if plink_data is None:
            logger.info(
                "NumPy backend: loading all genotypes into memory "
                "(for large datasets, use JAX backend: pip install jamma[jax])"
            )
            plink_data = load_plink_binary(self.config.bfile)

        genotypes = plink_data.genotypes

        # Apply snps_indices filter before passing to runner
        indices = snps_indices if snps_indices is not None else range(plink_data.n_snps)
        if snps_indices is not None:
            genotypes = genotypes[:, snps_indices]
        snp_info = [
            {
                "chr": str(plink_data.chromosome[i]),
                "rs": plink_data.sid[i],
                "pos": int(plink_data.bp_position[i]),
                "a1": plink_data.allele_1[i],
                "a0": plink_data.allele_2[i],
            }
            for i in indices
        ]

        lmm_config = LmmConfig(
            maf_threshold=self.config.maf,
            miss_threshold=self.config.miss,
            l_min=self.config.l_min,
            l_max=self.config.l_max,
            n_grid=self.config.n_grid,
            n_refine=self.config.n_refine,
            check_memory=False,  # Already checked above
            show_progress=self.config.show_progress,
            lmm_mode=self.config.lmm_mode,
        )

        run_result = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=K,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=lmm_config,
            output_path=assoc_path,
        )

        return run_result, run_result.snp_count
