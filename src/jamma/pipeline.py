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

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from loguru import logger

from jamma.core.backend import format_pipeline_banner, log_backend_selection
from jamma.core.chunk import _compute_chunk_size
from jamma.core.constants import PHENOTYPE_MISSING
from jamma.core.memory import (
    StreamingMemoryBreakdown,
    estimate_streaming_memory,
)
from jamma.io.covariate import read_covariate_file
from jamma.io.plink import PlinkData, get_plink_metadata, validate_plink_dimensions
from jamma.io.snp_list import read_snp_list_file, resolve_snp_list_to_indices
from jamma.kinship import (
    compute_kinship_streaming,
    compute_loco_kinship_streaming,
    compute_standardized_kinship,
    read_kinship_matrix,
    write_kinship_matrix,
    write_loco_kinship_matrices,
)
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.runner import ExecutionPlan, select_execution_mode, warn_if_small_sample
from jamma.lmm.schema import LmmRunResult, RunnerTiming
from jamma.lmm.stats import AssocResult
from jamma.pipeline_config import (
    VALID_BACKENDS,
    BackendRequest,
    KinshipResult,
    PipelineConfig,
    PipelineResult,
)

__all__ = [
    "BackendRequest",
    "KinshipResult",
    "PipelineConfig",
    "PipelineResult",
    "PipelineRunner",
]


def _parse_backend_override(value: str) -> BackendRequest:
    """Validate a JAMMA_BACKEND value against the accepted backend requests.

    Args:
        value: Raw ``JAMMA_BACKEND`` environment variable value.

    Returns:
        The value, narrowed to a valid backend request.

    Raises:
        ValueError: If the value is not a recognised backend.
    """
    if value not in VALID_BACKENDS:
        raise ValueError(
            f"JAMMA_BACKEND must be one of {VALID_BACKENDS}, got {value!r}"
        )
    return value


class _PhenoLoopOutcome(NamedTuple):
    """Aggregated results of the per-phenotype LMM loop.

    Returned by ``PipelineRunner._run_phenotype_loop`` so ``_run_inner`` can
    assemble the final ``PipelineResult`` without holding the loop's locals.
    """

    associations: list[AssocResult]
    n_tested: int
    assoc_paths: list[Path]
    lmm_s: float
    runner_timing: RunnerTiming
    pve: float | None
    pve_se: float | None


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

    def _emit_telemetry(self, result: PipelineResult, plan: ExecutionPlan) -> None:
        """Emit benchmark telemetry record. Never raises."""
        try:
            import jamma
            from jamma.core.telemetry import (
                BenchmarkRecord,
                append_benchmark_record,
            )
        except ImportError:
            logger.warning("Telemetry module not available", exc_info=True)
            return

        try:
            record: BenchmarkRecord = {
                "timestamp": datetime.now(UTC).isoformat(),
                "jamma_version": jamma.__version__,
                "n_samples": result.n_samples,
                "n_snps": result.n_snps_tested,
                "n_cvt": result.n_covariates,
                "backend": plan.runner_name,
                "lmm_mode": self.config.lmm_mode,
                "loco": self.config.loco,
            }
            for key in ("kinship_s", "lmm_s", "total_s", "rotation_s"):
                val = result.timing.get(key)
                if val is not None:
                    record[key] = val  # type: ignore[literal-required]
            append_benchmark_record(record)
        except Exception:  # noqa: BLE001 — telemetry must never break the pipeline; log and continue
            logger.warning("Telemetry emission failed", exc_info=True)

    @staticmethod
    def _compute_valid_mask(
        phenotypes: np.ndarray, covariates: np.ndarray | None
    ) -> np.ndarray:
        """Compute boolean mask of samples with valid phenotype and covariate values."""
        from jamma.lmm.prepare_common import compute_valid_mask

        return compute_valid_mask(phenotypes, covariates)

    def validate_inputs(self) -> None:
        """Validate that required input files exist and combine legally.

        Only checks that need the filesystem or span several fields live here.
        The LMM knobs (lmm_mode, maf, miss, l_min, l_max, n_grid) are already
        guaranteed by PipelineConfig.__post_init__, which builds an LmmConfig.

        Raises:
            FileNotFoundError: If PLINK files (.bed, .bim, .fam) are missing,
                or if kinship_file/covariate_file is specified but missing.
            ValueError: If mutually exclusive options are combined.
        """
        bfile = self.config.bfile
        for ext in (".bed", ".bim", ".fam"):
            p = Path(f"{bfile}{ext}")
            if not p.exists():
                raise FileNotFoundError(f"PLINK {ext} file not found: {p}")

        # Validate .bed file size matches .fam/.bim dimensions (VALID-01)
        validate_plink_dimensions(bfile)

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

        if has_eigen and self.config.loco:
            raise ValueError(
                "-d/-u (pre-computed eigen) not supported with -loco mode. "
                "Use --eigen-dir for per-chromosome eigen caching."
            )

        # Every option that names an input file gets the same check, so they
        # share one. The order is part of the contract: a config naming two
        # missing files reports the earlier one, and
        # tests/test_pipeline_validation_order.py pins that.
        required_files: tuple[tuple[Path | None, str], ...] = (
            (self.config.eigenvalue_file, "Eigenvalue file"),
            (self.config.eigenvector_file, "Eigenvector file"),
            (self.config.kinship_file, "Kinship matrix file"),
            (self.config.covariate_file, "Covariate file"),
            (self.config.snps_file, "SNP list file"),
            (self.config.ksnps_file, "Kinship SNP list file"),
            (self.config.weight_file, "Weight file"),
        )
        for path, label in required_files:
            if path is not None and not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

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
            ValueError: If the .fam file cannot be read, if pheno_col names a
                column the .fam file does not have, or if no sample has a valid
                phenotype. pheno_col is trusted to be >= 1;
                PipelineConfig.__post_init__ is where that is enforced.
        """
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
                f"phenotype column {pheno_col} exceeds available columns "
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
        """Parse the single configured phenotype column from the .fam file.

        Uses vectorized parsing: reads the phenotype column, replaces
        missing indicators ("-9", "NA") with NaN, converts to float64.

        Reads ``phenotype_columns[0]``. The LOCO path calls this, and
        PipelineConfig rejects multi-phenotype LOCO, so there is exactly one
        column to read there. The multi-phenotype path goes through
        ``_load_phenotypes_and_intersect_masks`` instead.

        Returns:
            Tuple of (phenotypes array, n_analyzed) where phenotypes has
            NaN for missing values and n_analyzed is the count of valid
            (non-NaN, non-missing) phenotypes.

        Raises:
            ValueError: If no samples have valid phenotypes, or if the column
                is not present in the .fam file.
        """
        return self._parse_phenotype_column(self.config.phenotype_columns[0])

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
            logger.info("Memory preflight skipped (streaming): check_memory=False")
            return None

        disk_chunk = _compute_chunk_size(n_snps)
        compute_chunk = _compute_chunk_size(
            n_snps, n_samples=n_samples, n_cvt=n_cvt, pipeline_buffers=2
        )
        est = estimate_streaming_memory(
            n_samples,
            chunk_size=disk_chunk,
            n_cvt=n_cvt,
            compute_chunk_size=compute_chunk,
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
        valid_indices: np.ndarray | None = None,
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
            valid_indices: Optional array of sample indices to keep. When provided,
                computed kinship is accumulated at (n_valid, n_valid) size directly;
                pre-computed kinship loaded from file is subsetted post-load via
                np.ix_. Must be sorted, unique, and within [0, n_samples).

        Returns:
            Kinship matrix of shape (n_out, n_out) where n_out = len(valid_indices)
            or n_samples.
        """
        if valid_indices is not None:
            from jamma.kinship.compute import _validate_valid_indices

            _validate_valid_indices(valid_indices, n_samples)

        if self.config.kinship_file is not None:
            logger.info(f"Loading kinship from {self.config.kinship_file}")
            K = read_kinship_matrix(self.config.kinship_file, n_samples=n_samples)
            # Pre-computed kinship is full-size; subset post-load
            if valid_indices is not None:
                K = K[np.ix_(valid_indices, valid_indices)]
        else:
            logger.info("Computing kinship from genotypes")
            K = compute_kinship_streaming(
                self.config.bfile,
                check_memory=False,
                show_progress=self.config.show_progress,
                ksnps_indices=ksnps_indices,
                valid_indices=valid_indices,
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
            # Filter weights to match valid samples
            if valid_indices is not None:
                weights = weights[valid_indices]
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
        if self.config.hwe_threshold > 0 and plan.mode == "batch":
            raise ValueError(
                "HWE filtering (--hwe) is not supported with the NumPy "
                "batch backend. Use --backend numpy-streaming or set --hwe 0."
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
                get_c_extension_thread_count,
                get_physical_core_count,
                is_blas_controllable,
            )
            from jamma.lmm._compile_utils import get_c_extension_capabilities

            c_ext, c_has_openmp = get_c_extension_capabilities()
            runner = plan.runner_name

            blas = get_blas_backend()

            # Respect JAMMA_BLAS_THREADS if set, otherwise use physical
            # core count. We avoid get_blas_thread_count() because it
            # imports threading module unconditionally.
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
                # (same fallback used by the NumPy LMM chunk runner).
                cores = get_physical_core_count()
                threads = max(1, cores // 2)

            # A single-threaded _lmm_accel build should not be logged as a
            # multi-threaded compute kernel.
            if c_ext:
                threads = min(
                    threads,
                    get_c_extension_thread_count(
                        c_accel_available=c_ext,
                        c_has_openmp=c_has_openmp,
                    ),
                )

            banner = format_pipeline_banner(
                runner=runner,
                blas=blas,
                eigen_driver="pending",
                c_ext=c_ext,
                threads=threads,
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

        # Resolve env override first: JAMMA_BACKEND takes priority in all paths.
        # It arrives as an unvalidated string, so check it here rather than
        # letting an unknown value reach select_execution_mode after the
        # pipeline has already read PLINK metadata off disk.
        env_backend = os.environ.get("JAMMA_BACKEND")
        requested: BackendRequest = (
            _parse_backend_override(env_backend)
            if env_backend is not None
            else self.config.backend
        )

        # Fail fast: HWE + explicit numpy is always invalid, before touching disk.
        if self.config.hwe_threshold > 0 and requested == "numpy":
            raise ValueError(
                "HWE filtering (--hwe) is not supported with the NumPy "
                "batch backend. Use --backend numpy-streaming or set --hwe 0."
            )

        # PLINK metadata is lightweight (reads .fam/.bim header only) and needed
        # for memory-based mode selection in both auto and explicit paths.
        from jamma.io.plink import get_plink_metadata as _get_meta

        _meta = _get_meta(self.config.bfile)

        # Route through select_execution_mode for all backend requests.
        plan = select_execution_mode(
            n_samples=_meta["n_samples"],
            n_snps=_meta["n_snps"],
            requested=requested,
        )

        log_backend_selection("numpy", self.config.backend, env_backend)
        logger.info(f"Execution plan: {plan.runner_name} ({plan.reason})")

        return self._run_inner(t_start, plan, requested)

    def compute_kinship(self, mode: int) -> KinshipResult:
        """Compute and write the kinship matrix (the ``-gk`` path).

        Orchestrates kinship computation end-to-end so the CLI is a thin shell
        (like ``run()`` for the ``-lmm`` path). Honours ``config.loco`` (writes
        per-chromosome LOCO matrices), ``config.write_eigen`` (eigendecomposes
        and writes the eigen files), and ``config.ksnps_file`` (restricts the
        SNPs used). Caller-facing validation (mode range, file existence,
        flag-combination guards) stays in the CLI.

        Args:
            mode: Kinship mode — 1 (centered, streaming) or 2 (standardized,
                in-memory).

        Returns:
            A KinshipResult with the written paths, dimensions, and timing.
        """
        meta = get_plink_metadata(self.config.bfile)
        n_samples = meta["n_samples"]
        n_snps = meta["n_snps"]

        # GEMMA-style banner — kinship uses all samples (n_analyzed == n_total).
        self._log_banner(n_total=n_samples, n_analyzed=n_samples, n_snps=n_snps)

        ksnps_indices = self._resolve_snp_list(
            self.config.ksnps_file, meta["sid"], "-ksnps"
        )

        t_kinship = time.perf_counter()

        if self.config.loco:
            logger.info(f"Computing LOCO kinship matrices from {self.config.bfile}")
            loco_iter = compute_loco_kinship_streaming(
                self.config.bfile,
                maf_threshold=self.config.maf,
                miss_threshold=self.config.miss,
                check_memory=self.config.check_memory,
                show_progress=self.config.show_progress,
                ksnps_indices=ksnps_indices,
                _copy_yielded_matrices=False,
            )
            written_paths = write_loco_kinship_matrices(
                loco_iter,
                output_dir=self.config.output_dir,
                prefix=self.config.output_prefix,
                legacy_text=self.config.legacy_text,
            )
            kinship_s = time.perf_counter() - t_kinship
            logger.info(
                f"Wrote {len(written_paths)} LOCO kinship matrices in {kinship_s:.2f}s"
            )
            return KinshipResult(
                kinship_paths=written_paths,
                eigen_paths=None,
                n_samples=n_samples,
                n_snps=n_snps,
                mode=mode,
                is_loco=True,
                kinship_s=kinship_s,
            )

        if self.config.maf > 0.0 or self.config.miss < 1.0:
            logger.info(
                f"Filtering: MAF >= {self.config.maf}, "
                f"missing rate <= {self.config.miss}"
            )

        if mode == 1:
            logger.info("Computing centered kinship matrix (streaming)")
            K = compute_kinship_streaming(
                self.config.bfile,
                maf_threshold=self.config.maf,
                miss_threshold=self.config.miss,
                check_memory=self.config.check_memory,
                show_progress=self.config.show_progress,
                ksnps_indices=ksnps_indices,
            )
        else:
            # Standardized kinship needs the full genotype matrix (no streaming).
            from jamma.io import load_plink_binary

            logger.info(f"Loading PLINK data from {self.config.bfile}")
            plink_data = load_plink_binary(self.config.bfile)
            genotypes = plink_data.genotypes
            if ksnps_indices is not None:
                genotypes = genotypes[:, ksnps_indices]
                logger.info(f"Using {genotypes.shape[1]} SNPs for kinship computation")
            logger.info("Computing standardized kinship matrix")
            K = compute_standardized_kinship(
                genotypes,
                maf_threshold=self.config.maf,
                miss_threshold=self.config.miss,
                check_memory=self.config.check_memory,
            )

        kinship_s = time.perf_counter() - t_kinship

        kinship_base = self.config.output_dir / f"{self.config.output_prefix}.cXX.txt"
        kinship_path = write_kinship_matrix(
            K, kinship_base, legacy_text=self.config.legacy_text
        )
        logger.info(f"Kinship matrix written to {kinship_path}")
        n_out = K.shape[0]

        eigen_paths: tuple[Path, Path] | None = None
        if self.config.write_eigen:
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K, check_memory=self.config.check_memory
            )
            del K  # K may be overwritten by eigendecomp; prevent accidental reuse
            d_path, u_path = write_eigen_files(
                eigenvalues,
                eigenvectors,
                self.config.output_dir,
                self.config.output_prefix,
                legacy_text=self.config.legacy_text,
            )
            eigen_paths = (d_path, u_path)
            logger.info(f"Eigenvalues written to {d_path}")
            logger.info(f"Eigenvectors written to {u_path}")

        return KinshipResult(
            kinship_paths=[kinship_path],
            eigen_paths=eigen_paths,
            n_samples=n_out,
            n_snps=n_snps,
            mode=mode,
            is_loco=False,
            kinship_s=kinship_s,
        )

    def _load_phenotypes_and_intersect_masks(
        self,
        pheno_columns: list[int],
        covariates: np.ndarray | None,
    ) -> tuple[dict[int, tuple[np.ndarray, int]], np.ndarray, int]:
        """Load each phenotype column and intersect their valid-sample masks.

        Reads .fam once, parses each phenotype column, computes the valid
        mask (non-NaN phenotype + non-NaN covariates) per column, then
        intersects across columns so eigendecomposition runs on the
        sample set common to every phenotype.

        Args:
            pheno_columns: Phenotype column numbers (1-based, as PLINK).
            covariates: Covariate matrix (n_samples, n_cvt) or None.

        Returns:
            ``(all_pheno_data, valid_mask, n_valid)`` where
            ``all_pheno_data[col] = (phenotype_array, n_analyzed)``,
            ``valid_mask`` is the boolean intersection across all columns,
            and ``n_valid`` is its sum.

        Raises:
            ValueError: If the .fam file can't be read, or if no sample is
                valid across all columns (with per-column counts in the
                message for diagnosis).
        """
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
            all_masks.append(self._compute_valid_mask(pheno, covariates))

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

        per_pheno_counts = [int(m.sum()) for m in all_masks]
        if n_valid < min(per_pheno_counts):
            logger.warning(
                f"Sample mask intersection reduced valid samples: "
                f"per-phenotype counts {per_pheno_counts}, "
                f"intersection {n_valid}"
            )

        return all_pheno_data, valid_mask, n_valid

    def _memory_preflight(
        self,
        plan: ExecutionPlan,
        n_valid: int,
        n_snps: int,
        n_cvt: int,
    ) -> None:
        """Run the memory preflight gate for the chosen execution plan.

        Streaming mode delegates to ``check_memory_requirements`` which
        uses streaming-specific accounting. Batch mode uses the
        in-memory estimator and additionally enforces ``mem_budget`` if
        the user set one. Both raise ``MemoryError`` with actionable
        messages on failure.

        Args:
            plan: Resolved ExecutionPlan (mode determines which estimator).
            n_valid: Sample count after valid-mask intersection.
            n_snps: Total SNPs from PLINK metadata (pre-MAF/missingness).
            n_cvt: Covariate count including the intercept.
        """
        if plan.mode == "streaming":
            self.check_memory_requirements(n_valid, n_snps, n_cvt=n_cvt)
            return

        if not self.config.check_memory:
            logger.info(
                f"Memory preflight skipped ({plan.runner_name}): check_memory=False"
            )
            return

        from jamma.core.memory import estimate_lmm_memory

        est = estimate_lmm_memory(n_valid, n_snps, n_cvt=n_cvt)
        logger.info(
            f"Memory estimate ({plan.runner_name}): "
            f"{est.total_gb:.1f}GB required, "
            f"{est.available_gb:.1f}GB available"
        )
        if self.config.mem_budget is not None and est.total_gb > self.config.mem_budget:
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

    def _run_inner(
        self,
        t_start: float,
        plan: ExecutionPlan,
        requested: Literal["auto", "numpy", "numpy-streaming"] = "auto",
    ) -> PipelineResult:
        """Execute the pipeline body.

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
            return self._run_loco(
                t_start=t_start,
                plan=plan,
                n_samples=n_samples,
                n_snps=n_snps,
                assoc_path=assoc_path,
                snps_indices=snps_indices,
                ksnps_indices=ksnps_indices,
            )

        covariates = self.load_covariates(n_samples)

        pheno_columns = self.config.phenotype_columns
        all_pheno_data, valid_mask, n_valid = self._load_phenotypes_and_intersect_masks(
            pheno_columns, covariates
        )

        n_cvt = covariates.shape[1] if covariates is not None else 1
        self._log_banner(
            n_samples,
            n_valid,
            n_snps,
            n_covariates=n_cvt,
            n_phenotypes=len(pheno_columns),
        )
        warn_if_small_sample(n_valid)

        # Re-evaluate the plan with the post-filter sample count (valid_mask
        # intersection can reduce n_valid below the PLINK-header n_samples, and
        # may flip batch<->streaming). Banner after, so it shows the final plan.
        plan = self._reselect_plan_after_filtering(
            plan, n_valid, n_snps, n_cvt, requested
        )
        self._log_pipeline_banner(plan)

        self._memory_preflight(plan, n_valid, n_snps, n_cvt)

        # Load/compute eigendecomposition ONCE (shared across phenotypes). The
        # kinship matrix is consumed here; runners use the eigen arrays directly.
        eigenvalues, eigenvectors, kinship_s = self._acquire_eigendecomposition(
            n_samples, n_valid, valid_mask, ksnps_indices
        )
        K = None
        load_s = time.perf_counter() - t_start

        outcome = self._run_phenotype_loop(
            plan,
            all_pheno_data,
            valid_mask,
            K,
            covariates,
            eigenvalues,
            eigenvectors,
            assoc_path,
            snps_indices,
        )

        total_s = time.perf_counter() - t_start
        logger.info(f"GWAS complete: {outcome.n_tested} SNPs tested in {total_s:.1f}s")

        result = PipelineResult(
            associations=outcome.associations,
            n_samples=n_valid,
            n_snps_tested=outcome.n_tested,
            assoc_path=outcome.assoc_paths[-1],
            assoc_paths=outcome.assoc_paths,
            timing={
                "kinship_s": kinship_s,
                "load_s": load_s,
                "lmm_s": outcome.lmm_s,
                "total_s": total_s,
                "rotation_s": outcome.runner_timing.get("rotation_s", 0.0),
                "rotation_exposed_s": outcome.runner_timing.get(
                    "rotation_exposed_s", 0.0
                ),
            },
            backend="numpy",
            n_covariates=(covariates.shape[1] if covariates is not None else 1),
            pve_estimate=outcome.pve,
            pve_se=outcome.pve_se,
        )
        self._emit_telemetry(result, plan)
        return result

    def _reselect_plan_after_filtering(
        self,
        initial_plan: ExecutionPlan,
        n_valid: int,
        n_snps: int,
        n_cvt: int,
        requested: Literal["auto", "numpy", "numpy-streaming"],
    ) -> ExecutionPlan:
        """Re-select the execution plan using the post-filter sample count.

        The initial plan used raw ``n_samples`` from the PLINK header; the
        valid-mask intersection can reduce it (and flip batch<->streaming).
        Re-selects with ``n_valid`` and ``n_cvt`` (so memory estimates account
        for a larger Uab at n_cvt>1), logs any change, and re-checks HWE support
        when the runner changes.

        Returns:
            The post-filter ExecutionPlan (unchanged when filtering had no
            effect on mode selection).
        """
        plan = select_execution_mode(n_valid, n_snps, requested=requested, n_cvt=n_cvt)
        if plan != initial_plan:
            logger.info(
                f"Execution plan changed after sample filtering: "
                f"{initial_plan.runner_name} -> {plan.runner_name} ({plan.reason})"
            )
            self._check_hwe_support(plan)
        else:
            logger.debug(
                f"Execution plan (post-filter): {plan.runner_name} ({plan.reason})"
            )
        return plan

    def _acquire_eigendecomposition(
        self,
        n_samples: int,
        n_valid: int,
        valid_mask: np.ndarray,
        ksnps_indices: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Load or compute the shared eigendecomposition (once for all phenotypes).

        Either reads pre-computed eigen files (-d/-u), or loads/computes the
        kinship matrix, subsets it to the valid samples, and eigendecomposes it
        (optionally writing the eigen files). The kinship matrix is consumed
        here — the runners use the eigenvalues/eigenvectors directly.

        Returns:
            ``(eigenvalues, eigenvectors, kinship_s)`` where ``kinship_s`` is the
            wall time spent acquiring the eigendecomposition.
        """
        t_kinship = time.perf_counter()

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
        else:
            # Early sample filtering: when save_kinship=False and some
            # samples are invalid, pass valid_indices so kinship is
            # accumulated at (n_valid, n_valid) size directly, avoiding
            # allocation of the full (n_samples, n_samples) matrix.
            # When save_kinship=True, compute full-size kinship so the
            # saved file is reusable across different phenotype masks.
            all_valid = np.all(valid_mask)
            kinship_valid_indices = (
                None
                if all_valid or self.config.save_kinship
                else np.where(valid_mask)[0]
            )
            K = self.load_kinship(
                n_samples,
                ksnps_indices=ksnps_indices,
                valid_indices=kinship_valid_indices,
            )
            # When kinship_valid_indices was passed, K is already
            # (n_valid, n_valid). Otherwise K is full-size and may need
            # subsetting (save_kinship=True with invalid samples).
            K_valid = (
                K
                if (kinship_valid_indices is not None or all_valid)
                else K[np.ix_(valid_mask, valid_mask)]
            )
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

        kinship_s = time.perf_counter() - t_kinship
        return eigenvalues, eigenvectors, kinship_s

    def _run_phenotype_loop(
        self,
        plan: ExecutionPlan,
        all_pheno_data: dict[int, tuple[np.ndarray, int]],
        valid_mask: np.ndarray,
        K: np.ndarray | None,
        covariates: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        eigenvectors: np.ndarray | None,
        assoc_path: Path,
        snps_indices: np.ndarray | None,
    ) -> _PhenoLoopOutcome:
        """Run the per-phenotype LMM loop and aggregate its results.

        Iterates the configured phenotype columns, masking each to the shared
        valid-sample intersection, dispatching to the batch or streaming runner
        per the plan, and collecting associations, counts, and output paths.
        Captures PVE and runner rotation timing from the final phenotype.

        Returns:
            A _PhenoLoopOutcome bundling associations, total SNPs tested, the
            per-phenotype output paths, the loop wall time, runner timing, and
            the PVE estimate.
        """
        pheno_columns = self.config.phenotype_columns
        is_multi = len(pheno_columns) > 1

        t_lmm = time.perf_counter()
        all_results: list[AssocResult] = []
        total_tested = 0
        all_assoc_paths: list[Path] = []

        # Pre-load PLINK data once for batch multi-phenotype runs
        _plink_data = None
        if plan.mode == "batch" and is_multi:
            from jamma.io import load_plink_binary

            logger.info(
                f"{plan.runner_name}: loading all genotypes into memory"
                " (for large datasets, use --backend numpy-streaming)"
            )
            _plink_data = load_plink_binary(self.config.bfile)

        # The loop's last run carries the PVE estimate; both stay None if
        # pheno_columns is empty, which PipelineConfig already rejects.
        prefix = self.config.output_prefix
        pve: float | None = None
        pve_se: float | None = None
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

            if plan.mode == "streaming":
                run_result, n_tested = self._run_streaming(
                    phenotypes_col,
                    covariates,
                    eigenvalues,
                    eigenvectors,
                    col_path,
                    snps_indices,
                )
            else:
                run_result, n_tested = self._run_batch(
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
            pve, pve_se = run_result.pve, run_result.pve_se
            logger.info(f"Phenotype {col}: {n_tested} SNPs tested -> {col_path}")

        lmm_s = time.perf_counter() - t_lmm

        # Pull runner-level rotation timing from the most recent runner call.
        runner_timing: RunnerTiming = {}
        if plan.mode == "streaming":
            from jamma.lmm.runner_numpy_streaming import (
                get_last_run_timing as _np_stream_timing,
            )

            runner_timing = _np_stream_timing()

        return _PhenoLoopOutcome(
            associations=all_results,
            n_tested=total_tested,
            assoc_paths=all_assoc_paths,
            lmm_s=lmm_s,
            runner_timing=runner_timing,
            pve=pve,
            pve_se=pve_se,
        )

    def _run_loco(
        self,
        *,
        t_start: float,
        plan: ExecutionPlan,
        n_samples: int,
        n_snps: int,
        assoc_path: Path,
        snps_indices: np.ndarray | None,
        ksnps_indices: np.ndarray | None,
    ) -> PipelineResult:
        """LOCO branch of the pipeline.

        Self-contained early return from _run_inner: parses single
        phenotype, loads covariates, runs the LOCO orchestrator (which
        owns its own per-chromosome kinship + eigendecomposition), and
        assembles a PipelineResult.

        Single-phenotype only — multi-phenotype LOCO is rejected at
        PipelineConfig.__post_init__.
        """
        from jamma.lmm import run_lmm_loco
        from jamma.lmm.loco import LocoConfig

        phenotypes, n_analyzed = self.parse_phenotypes()
        n_filtered = len(phenotypes) - n_analyzed
        logger.info(
            f"Analyzing {n_analyzed} samples with valid "
            f"phenotypes ({n_filtered} filtered)"
        )

        covariates = self.load_covariates(n_samples)
        valid_mask = self._compute_valid_mask(phenotypes, covariates)
        n_valid = int(np.sum(valid_mask))
        n_cvt = covariates.shape[1] if covariates is not None else 1
        self._log_banner(n_samples, n_valid, n_snps, n_covariates=n_cvt)
        self._log_pipeline_banner(plan)
        warn_if_small_sample(n_valid)

        t_loco = time.perf_counter()
        loco = run_lmm_loco(
            bed_path=self.config.bfile,
            phenotypes=phenotypes,
            covariates=covariates,
            # check_memory passed through rather than forced off: this branch
            # returns from _run_inner before _memory_preflight, so run_lmm_loco
            # owns the only memory gate on the LOCO path.
            config=self.config.lmm_config(check_memory=self.config.check_memory),
            loco=LocoConfig(
                save_kinship=self.config.save_kinship,
                kinship_output_dir=self.config.output_dir,
                kinship_output_prefix=self.config.output_prefix,
                snps_indices=snps_indices,
                ksnps_indices=ksnps_indices,
                write_eigen=self.config.write_eigen,
                eigen_dir=self.config.eigen_dir,
                eigen_prefix=self.config.output_prefix,
                legacy_text=self.config.legacy_text,
            ),
            output_path=assoc_path,
        )
        loco_s = time.perf_counter() - t_loco
        total_s = time.perf_counter() - t_start
        logger.info(
            f"LOCO GWAS complete: {loco.n_tested} SNPs tested in {total_s:.1f}s"
        )

        result = PipelineResult(
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
            backend="numpy",
            n_covariates=covariates.shape[1] if covariates is not None else 1,
            pve_estimate=loco.pve,
            pve_se=loco.pve_se,
        )
        self._emit_telemetry(result, plan)
        return result

    def _run_batch(
        self,
        phenotypes: np.ndarray,
        K: np.ndarray | None,
        covariates: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        eigenvectors: np.ndarray | None,
        assoc_path: Path,
        snps_indices: np.ndarray | None,
        plink_data: PlinkData | None = None,
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
                "(for large datasets, use --backend numpy-streaming)"
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

        run_result = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=K,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=self.config.lmm_config(),
            output_path=assoc_path,
        )

        return run_result, run_result.snp_count

    def _run_streaming(
        self,
        phenotypes: np.ndarray,
        covariates: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        eigenvectors: np.ndarray | None,
        assoc_path: Path,
        snps_indices: np.ndarray | None,
    ) -> tuple[LmmRunResult, int]:
        """Run LMM via NumPy streaming backend (disk I/O + C extension)."""
        from jamma.lmm.runner_numpy_streaming import (
            run_lmm_association_numpy_streaming,
        )

        return run_lmm_association_numpy_streaming(
            bed_path=self.config.bfile,
            phenotypes=phenotypes,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            output_path=assoc_path,
            snps_indices=snps_indices,
            hwe_threshold=self.config.hwe_threshold,
            config=self.config.lmm_config(),
        )
