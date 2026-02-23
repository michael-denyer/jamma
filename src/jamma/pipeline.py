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

import jax
import numpy as np
from loguru import logger

from jamma.core.jax_config import ensure_jax_configured
from jamma.core.memory import StreamingMemoryBreakdown, estimate_streaming_memory
from jamma.io.covariate import read_covariate_file
from jamma.io.plink import get_plink_metadata, validate_plink_dimensions
from jamma.io.snp_list import read_snp_list_file, resolve_snp_list_to_indices
from jamma.kinship import (
    compute_kinship_streaming,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm import run_lmm_association_streaming, run_lmm_loco
from jamma.lmm.chunk import _compute_chunk_size
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
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
        eigenvalue_file: Pre-computed eigenvalue file (.eigenD.txt), or None.
            Must be paired with eigenvector_file (-d flag).
        eigenvector_file: Pre-computed eigenvector file (.eigenU.txt), or None.
            Must be paired with eigenvalue_file (-u flag).
        write_eigen: If True, write eigendecomposition files as side effect
            (-eigen flag).
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
    phenotype_column: int = 1
    snps_file: Path | None = None
    ksnps_file: Path | None = None
    hwe_threshold: float = 0.0
    l_min: float = 1e-5
    l_max: float = 1e5
    weight_file: Path | None = None
    cat_columns: list[int] | None = None
    profile_dir: Path | None = None

    def __post_init__(self) -> None:
        if os.sep in self.output_prefix or "/" in self.output_prefix:
            raise ValueError(
                f"output_prefix must not contain path separators, "
                f"got '{self.output_prefix}'. Use output_dir for directory paths."
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
        assoc_path: Path to the written association results file.
        timing: Timing breakdown by pipeline phase.
    """

    associations: list[AssocResult]
    n_samples: int
    n_snps_tested: int
    assoc_path: Path
    timing: dict[str, float] = field(default_factory=dict)


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
        """Compute boolean mask of samples with valid phenotype and covariate values.

        Args:
            phenotypes: Phenotype vector (n_samples,).
            covariates: Covariate matrix (n_samples, n_cvt) or None.

        Returns:
            Boolean mask array of shape (n_samples,) where True indicates
            a sample with valid phenotype and covariate values.
        """
        valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
        if covariates is not None:
            valid_covariate = np.all(~np.isnan(covariates), axis=1)
            valid_mask = valid_mask & valid_covariate
        return valid_mask

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
                    "-d/-u (pre-computed eigen) not supported with -loco mode"
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

    def parse_phenotypes(self) -> tuple[np.ndarray, int]:
        """Parse phenotypes from the .fam file.

        Uses vectorized parsing: reads the phenotype column, replaces
        missing indicators ("-9", "NA") with NaN, converts to float64.
        The column is selected by ``self.config.phenotype_column`` (1-based,
        matching GEMMA's ``-n`` flag).

        Returns:
            Tuple of (phenotypes array, n_analyzed) where phenotypes has
            NaN for missing values and n_analyzed is the count of valid
            (non-NaN, non-missing) phenotypes.

        Raises:
            ValueError: If no samples have valid phenotypes, or if
                phenotype_column is invalid.
        """
        pheno_col = self.config.phenotype_column
        if pheno_col < 1:
            raise ValueError(
                f"phenotype_column must be >= 1 (1-based), got {pheno_col}"
            )

        # Columns 0-4 are FID, IID, father, mother, sex; phenotype 1 is column 5
        col_index = 4 + pheno_col

        fam_path = f"{self.config.bfile}.fam"

        # Check column count from first line
        try:
            first_line = np.loadtxt(fam_path, dtype=str, max_rows=1)
        except (ValueError, OSError) as e:
            raise ValueError(f"Failed to read .fam file {fam_path}: {e}") from e
        n_cols = first_line.size
        if col_index >= n_cols:
            n_pheno_cols = n_cols - 5
            raise ValueError(
                f"phenotype_column {pheno_col} exceeds available columns "
                f"in .fam file ({n_pheno_cols} phenotype column"
                f"{'s' if n_pheno_cols != 1 else ''} available)"
            )

        logger.info(f"Using phenotype column {pheno_col} (file column {col_index + 1})")

        try:
            fam_data = np.loadtxt(fam_path, dtype=str, usecols=(col_index,))
        except (ValueError, OSError) as e:
            raise ValueError(
                f"Failed to read phenotype column {pheno_col} from {fam_path}: {e}"
            ) from e
        missing_mask = np.isin(fam_data, ["-9", "NA"])
        fam_data[missing_mask] = "0"  # placeholder for safe float conversion
        phenotypes = fam_data.astype(np.float64)
        phenotypes[missing_mask] = np.nan

        valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9)
        n_analyzed = int(valid_mask.sum())

        if n_analyzed == 0:
            raise ValueError("No samples with valid phenotypes")

        return phenotypes, n_analyzed

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

        actual_chunk = _compute_chunk_size(n_samples, n_snps, n_cvt=n_cvt)
        est = estimate_streaming_memory(n_samples, chunk_size=actual_chunk, n_cvt=n_cvt)

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
            kinship_path = (
                self.config.output_dir / f"{self.config.output_prefix}.cXX.txt"
            )
            write_kinship_matrix(K, kinship_path)
            logger.info(f"Kinship matrix saved to {kinship_path}")

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

    def run(self) -> PipelineResult:
        """Execute the full GWAS pipeline.

        Pipeline steps:
        1. Validate inputs
        2. Get PLINK metadata
        3. Check memory requirements
        4. Parse phenotypes
        5. Resolve SNP list files
        6. Prepare output directory
        7. Load covariates (early, for eigen validation)
        8. Load eigen files or kinship matrix
        9. Run LMM association (streaming)

        Returns:
            PipelineResult with associations, counts, output path, and timing.
        """
        t_start = time.perf_counter()

        # Enable x64 without initializing the JAX backend — device count is
        # deferred until LMM phase via ensure_jax_configured().
        jax.config.update("jax_enable_x64", True)

        # Optional XLA profiling — degrade gracefully so profiling issues
        # never prevent GWAS results from being produced.
        trace_ctx = contextlib.nullcontext()
        if self.config.profile_dir is not None:
            try:
                self.config.profile_dir.mkdir(parents=True, exist_ok=True)
                trace_ctx = jax.profiler.trace(
                    str(self.config.profile_dir), create_perfetto_link=False
                )
                logger.info(
                    "XLA profiling enabled: traces will be written to "
                    f"{self.config.profile_dir}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not enable XLA profiling: {type(e).__name__}: {e}. "
                    "Continuing without profiling."
                )

        with trace_ctx:
            return self._run_inner(t_start)

    def _run_inner(self, t_start: float) -> PipelineResult:
        """Execute the pipeline body, called within the optional profiling context."""
        self.validate_inputs()

        meta = get_plink_metadata(self.config.bfile)
        n_samples = meta["n_samples"]
        n_snps = meta["n_snps"]

        phenotypes, n_analyzed = self.parse_phenotypes()
        n_filtered = len(phenotypes) - n_analyzed
        logger.info(
            f"Analyzing {n_analyzed} samples with valid phenotypes "
            f"({n_filtered} filtered)"
        )

        snps_indices = self._resolve_snp_list(
            self.config.snps_file, meta["sid"], "-snps"
        )
        ksnps_indices = self._resolve_snp_list(
            self.config.ksnps_file, meta["sid"], "-ksnps"
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        assoc_path = self.config.output_dir / f"{self.config.output_prefix}.assoc.txt"

        # LOCO branch: skip standard kinship, run LOCO orchestrator
        if self.config.loco:
            covariates = self.load_covariates(n_samples)
            valid_mask = self._compute_valid_mask(phenotypes, covariates)
            n_valid = int(np.sum(valid_mask))
            n_cvt = covariates.shape[1] if covariates is not None else 1
            self._log_banner(n_samples, n_valid, n_snps, n_covariates=n_cvt)

            t_loco = time.perf_counter()
            results, n_tested = run_lmm_loco(
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
            )
            loco_s = time.perf_counter() - t_loco
            total_s = time.perf_counter() - t_start
            logger.info(f"LOCO GWAS complete: {n_tested} SNPs tested in {total_s:.1f}s")

            return PipelineResult(
                associations=results,
                n_samples=n_valid,
                n_snps_tested=n_tested,
                assoc_path=assoc_path,
                timing={
                    "kinship_s": 0.0,
                    "load_s": 0.0,
                    "lmm_s": loco_s,
                    "total_s": total_s,
                    "n_covariates": (
                        covariates.shape[1] if covariates is not None else 1
                    ),
                },
            )

        covariates = self.load_covariates(n_samples)
        valid_mask = self._compute_valid_mask(phenotypes, covariates)
        n_valid = int(np.sum(valid_mask))
        n_cvt = covariates.shape[1] if covariates is not None else 1
        self._log_banner(n_samples, n_valid, n_snps, n_covariates=n_cvt)

        actual_chunk = _compute_chunk_size(n_valid, n_snps, n_cvt=n_cvt)
        self.check_memory_requirements(n_valid, n_snps, n_cvt=n_cvt)

        t_kinship = time.perf_counter()
        eigenvalues = None
        eigenvectors = None

        if self.config.eigenvalue_file and self.config.eigenvector_file:
            # Load pre-computed eigendecomposition
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
            if self.config.write_eigen:
                K_valid = K if np.all(valid_mask) else K[np.ix_(valid_mask, valid_mask)]
                eigenvalues, eigenvectors = eigendecompose_kinship(
                    K_valid, check_memory=self.config.check_memory
                )
                d_path, u_path = write_eigen_files(
                    eigenvalues,
                    eigenvectors,
                    self.config.output_dir,
                    self.config.output_prefix,
                )
                logger.info(f"Wrote eigenvalues to {d_path}")
                logger.info(f"Wrote eigenvectors to {u_path}")
                K = None  # Runner uses eigenvalues/eigenvectors directly

        kinship_s = time.perf_counter() - t_kinship
        load_s = time.perf_counter() - t_start

        # Initialize JAX backend after kinship + eigendecomp completes,
        # to avoid XLA thread pools competing with MKL during eigendecomp.
        ensure_jax_configured()

        t_lmm = time.perf_counter()
        results, n_tested = run_lmm_association_streaming(
            bed_path=self.config.bfile,
            phenotypes=phenotypes,
            kinship=K,
            snp_info=None,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            maf_threshold=self.config.maf,
            miss_threshold=self.config.miss,
            l_min=self.config.l_min,
            l_max=self.config.l_max,
            output_path=assoc_path,
            lmm_mode=self.config.lmm_mode,
            check_memory=False,  # Already checked above
            show_progress=self.config.show_progress,
            snps_indices=snps_indices,
            hwe_threshold=self.config.hwe_threshold,
            chunk_size=actual_chunk,
        )
        lmm_s = time.perf_counter() - t_lmm

        total_s = time.perf_counter() - t_start
        logger.info(f"GWAS complete: {n_tested} SNPs tested in {total_s:.1f}s")

        return PipelineResult(
            associations=results,
            n_samples=n_valid,
            n_snps_tested=n_tested,
            assoc_path=assoc_path,
            timing={
                "kinship_s": kinship_s,
                "load_s": load_s,
                "lmm_s": lmm_s,
                "total_s": total_s,
                "n_covariates": covariates.shape[1] if covariates is not None else 1,
            },
        )
