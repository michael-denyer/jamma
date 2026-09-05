"""The ``-lmm`` path: PipelineRunner, which orchestrates an association run.

Validate inputs, parse phenotypes, check memory, load kinship, load covariates,
run LMM association. Both the CLI (cli.py) and the Python API (gwas.py) delegate
here.

The pieces that are not orchestration live in sibling modules, so this file holds
the flow and not the detail:

- ``pipeline_config.py`` — the config, result, and kinship-result dataclasses
- ``pipeline_banner.py`` — the two startup banners
- ``pipeline_phenotype_loop.py`` — the per-phenotype loop and the runner calls
- ``pipeline_kinship.py`` — the separate ``-gk`` program
- ``pipeline_memory.py`` — the memory preflight gate for both modes

Example:
    >>> from jamma.pipeline import PipelineConfig, PipelineRunner
    >>> config = PipelineConfig(bfile=Path("data/study"), kinship_file=Path("k.txt"))
    >>> result = PipelineRunner(config).run()
    >>> print(f"Tested {result.n_snps_tested} SNPs")
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.constants import PHENOTYPE_MISSING, Env
from jamma.io.covariate import read_covariate_file
from jamma.io.plink import (
    get_plink_metadata,
    parse_fam_phenotype_column,
    validate_plink_dimensions,
)
from jamma.io.snp_list import resolve_snp_list_file
from jamma.kinship import (
    compute_kinship_streaming,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm.association_plan import ExecutionPlan, plan_association
from jamma.lmm.eigen import center_kinship, eigendecompose_kinship
from jamma.lmm.eigen_io import (
    managed_eigen_pair_exists,
    read_eigen_files,
    write_eigen_files,
)
from jamma.lmm.loco_config import DEFAULT_LOCO_CONFIG
from jamma.lmm.prepare_common import compute_valid_mask
from jamma.lmm.schema import PipelineTiming, parse_lmm_mode
from jamma.pipeline_banner import log_dataset_banner, log_pipeline_banner
from jamma.pipeline_config import (
    VALID_BACKENDS,
    BackendRequest,
    KinshipResult,
    PhenotypeResult,
    PipelineConfig,
    PipelineResult,
)
from jamma.pipeline_memory import memory_preflight
from jamma.pipeline_phenotype_loop import run_phenotype_loop
from jamma.pipeline_plan import (
    KinshipSource,
    LocoAnalysisPlan,
    ProvidedEigen,
    ProvidedKinship,
    StandardAnalysisPlan,
    resolve_analysis_plan,
)

__all__ = [
    "BackendRequest",
    "KinshipResult",
    "PhenotypeResult",
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


SMALL_SAMPLE_WARNING_THRESHOLD = 50


def warn_if_small_sample(n_samples: int) -> None:
    """Warn once when sample size is below the practical LMM threshold.

    JAMMA is designed for large-scale GWAS (thousands to hundreds of thousands
    of samples). Below ~50 samples, two concerns apply:

    1. LMM has insufficient statistical power regardless of optimizer — kinship
       estimation and variance component inference are unreliable with so few
       samples.
    2. JAMMA's batch-vectorized grid+golden-section lambda optimizer assumes
       the log-likelihood is unimodal in log-lambda space. Very small samples
       are one of the scenarios where that assumption can fail, and unlike
       GEMMA's Brent's method JAMMA has no mechanism to detect multimodality.
       Results may diverge meaningfully from GEMMA on such adversarial inputs.

    See docs/GEMMA_DIVERGENCES.md §6 for full context.

    Args:
        n_samples: Number of samples actually entering the LMM (post
            phenotype/covariate filtering, not the raw PLINK header count).
    """
    if n_samples < SMALL_SAMPLE_WARNING_THRESHOLD:
        logger.warning(
            f"Small sample size ({n_samples} < {SMALL_SAMPLE_WARNING_THRESHOLD}): "
            "LMM-based GWAS has insufficient statistical power at this scale, "
            "and JAMMA's batch golden-section lambda optimizer may diverge from "
            "GEMMA's Brent's method on multimodal likelihoods. "
            "See docs/GEMMA_DIVERGENCES.md §6."
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
            record["kinship_s"] = result.timing.kinship_s
            record["lmm_s"] = result.timing.lmm_s
            record["total_s"] = result.timing.total_s
            record["rotation_s"] = result.timing.rotation_s
            # JAMMA_NO_TELEMETRY takes priority in all paths, matching
            # JAMMA_BACKEND above: the CLI's --no-telemetry sets
            # config.no_telemetry, a pure-Python-API caller only has the
            # env var, so both must reach append_benchmark_record.
            no_telemetry = self.config.no_telemetry or Env.current().no_telemetry
            append_benchmark_record(record, no_telemetry=no_telemetry)
        except Exception:  # noqa: BLE001 — telemetry must never break the pipeline; log and continue
            logger.warning("Telemetry emission failed", exc_info=True)

    def validate_inputs(self) -> None:
        """Validate that required input files exist and combine legally.

        Only checks that need the filesystem live here, plus the cross-field
        rules that mention a file. Everything decidable from the config alone
        (the LMM knobs, hwe_threshold, cat_columns, phenotype_columns) is
        already guaranteed by PipelineConfig.__post_init__.

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
        managed_pair = (
            has_eigen
            and self.config.eigenvalue_file is not None
            and self.config.eigenvector_file is not None
            and managed_eigen_pair_exists(
                self.config.eigenvalue_file, self.config.eigenvector_file
            )
        )
        for path, label in required_files:
            if managed_pair and label in {"Eigenvalue file", "Eigenvector file"}:
                continue
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

    def _parse_phenotype_column(
        self, pheno_col: int, fam_data: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Parse a specific phenotype column from pre-loaded .fam data.

        Args:
            pheno_col: 1-based phenotype column index.
            fam_data: The whole .fam file as a 2-D string array, read once by
                ``_load_phenotypes_and_intersect_masks``.

        Returns:
            Tuple of (phenotypes array, n_analyzed) where phenotypes has
            NaN for missing values and n_analyzed is the count of valid
            (non-NaN, non-missing) phenotypes.

        Raises:
            ValueError: If pheno_col names a column the .fam file does not
                have, or if no sample has a valid phenotype. pheno_col is
                trusted to be >= 1; PipelineConfig.__post_init__ is where that
                is enforced.
        """
        phenotypes = parse_fam_phenotype_column(fam_data, pheno_col)
        logger.info(f"Using phenotype column {pheno_col} (file column {pheno_col + 5})")

        valid_mask = ~np.isnan(phenotypes) & (phenotypes != PHENOTYPE_MISSING)
        n_analyzed = int(valid_mask.sum())

        if n_analyzed == 0:
            raise ValueError("No samples with valid phenotypes")

        return phenotypes, n_analyzed

    def _load_kinship_from_source(
        self,
        source: KinshipSource,
        n_samples: int,
        valid_indices: np.ndarray | None,
    ) -> np.ndarray:
        """Load or compute the kinship matrix over the valid samples.

        A ``ProvidedKinship`` source loads from disk; ``ComputedKinship``
        streams from genotypes. Derive the source with
        ``pipeline_plan.resolve_kinship_source`` so it cannot drift from
        the resolver's choice.

        If weight_file is configured, applies individual weights to K via
        K[i,j] /= sqrt(w_i * w_j) after centering the analysed matrix.

        If save_kinship is True, writes the kinship matrix to the output
        directory before analysis centering and weighting. The saved matrix is always
        full (n_samples, n_samples), so it can be reused under a different
        phenotype mask using the same SNP set; subsetting happens after the write.
        Without save_kinship a computed kinship is accumulated at
        (n_valid, n_valid) directly and the full matrix is never allocated.

        Args:
            source: Where the kinship comes from, per the resolved plan.
            n_samples: Number of samples (for validation of loaded kinship).
            valid_indices: Sample indices to keep, or None for all samples.
                Must be sorted, unique, and within [0, n_samples).

        Returns:
            Kinship matrix of shape (n_out, n_out) where n_out = len(valid_indices)
            or n_samples.
        """
        if valid_indices is not None:
            from jamma.kinship import validate_valid_indices

            validate_valid_indices(valid_indices, n_samples)

        # The full matrix is needed when it is going to be saved; otherwise a
        # computed kinship is accumulated over the valid samples directly.
        full = valid_indices is None or self.config.save_kinship

        if isinstance(source, ProvidedKinship):
            logger.info(f"Loading kinship from {source.path}")
            K = read_kinship_matrix(source.path, n_samples=n_samples)
            if not full:
                K = K[np.ix_(valid_indices, valid_indices)]
        else:
            logger.info("Computing kinship from genotypes")
            K = compute_kinship_streaming(
                self.config.bfile,
                maf_threshold=self.config.maf,
                miss_threshold=self.config.miss,
                check_memory=False,
                show_progress=self.config.show_progress,
                ksnps_indices=source.ksnps_indices,
                valid_indices=None if full else valid_indices,
                filter_sample_indices=valid_indices,
            )

        if self.config.save_kinship:
            kinship_base = (
                self.config.output_dir / f"{self.config.output_prefix}.cXX.txt"
            )
            actual_path = write_kinship_matrix(
                K, kinship_base, legacy_text=self.config.legacy_text
            )
            logger.info(f"Kinship matrix saved to {actual_path}")

        if full and valid_indices is not None:
            K = K[np.ix_(valid_indices, valid_indices)]
        center_kinship(K)

        # Apply individual weights before eigendecomposition
        if self.config.weight_file is not None:
            from jamma.io.weight import apply_individual_weights, read_weight_file

            weights = read_weight_file(self.config.weight_file)
            if len(weights) != n_samples:
                raise ValueError(
                    f"Weight file has {len(weights)} entries but expected "
                    f"{n_samples} (matching sample count)"
                )
            if valid_indices is not None:
                weights = weights[valid_indices]
            logger.info(f"Applying individual weights from {self.config.weight_file}")
            K = apply_individual_weights(K, weights)

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

    def run(self) -> PipelineResult:
        """Execute the full GWAS pipeline.

        Pipeline steps:
        1. Resolve the backend request and read PLINK metadata
        2. Validate inputs
        3. Resolve SNP list files, prepare the output directory
        4. Load covariates, then every phenotype column (one .fam read) and
           intersect their valid-sample masks
        5. Select the execution plan once, with the post-mask sample count
        6. LOCO returns here to its own orchestrator, which owns
           per-chromosome kinship, eigendecomposition and the memory gate
        7. Check memory against the selected plan
        8. Load eigen files or kinship matrix (once, shared)
        9. Per-phenotype loop: run LMM association and write results

        Returns:
            PipelineResult with associations, counts, output path, and timing.
        """
        t_start = time.perf_counter()

        # Resolve env override first: JAMMA_BACKEND takes priority in all paths.
        # It arrives as an unvalidated string, so check it here rather than
        # letting an unknown value reach plan_association after the
        # pipeline has already read PLINK metadata off disk.
        env_backend = Env.current().backend_raw
        requested: BackendRequest = (
            _parse_backend_override(env_backend)
            if env_backend is not None
            else self.config.backend
        )
        # Read once and pass it down. get_plink_metadata parses the whole .bim
        # (sid, chromosome, bp_position and both allele arrays).
        meta = get_plink_metadata(self.config.bfile)

        if env_backend is not None:
            logger.info(f"Backend: numpy (from JAMMA_BACKEND={env_backend})")
        elif self.config.backend != "auto":
            logger.info("Backend: numpy (explicitly requested)")
        else:
            logger.info("Backend: numpy (auto-selected)")

        self.validate_inputs()

        n_samples = meta.n_samples
        n_snps = meta.n_snps

        snps_indices = resolve_snp_list_file(self.config.snps_file, meta.sid, "-snps")
        ksnps_indices = resolve_snp_list_file(
            self.config.ksnps_file, meta.sid, "-ksnps"
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        assoc_path = self.config.output_dir / f"{self.config.output_prefix}.assoc.txt"

        covariates = self.load_covariates(n_samples)

        pheno_columns = self.config.phenotype_columns
        all_pheno_data, valid_mask, n_valid = self._load_phenotypes_and_intersect_masks(
            pheno_columns, covariates
        )

        n_cvt = covariates.shape[1] if covariates is not None else 1
        log_dataset_banner(
            n_samples,
            n_valid,
            n_snps,
            n_covariates=n_cvt,
            n_phenotypes=len(pheno_columns),
        )
        warn_if_small_sample(n_valid)

        # Select the plan once, with the post-mask sample count and the real
        # n_cvt: masking can reduce n_valid below the PLINK-header n_samples,
        # and Uab sizing depends on n_cvt. A prior version selected twice
        # (once here with the pre-mask n_samples, once again after masking),
        # re-running estimate_lmm_memory both times; this is the single call.
        execution = plan_association(
            n_samples=n_valid,
            n_input_samples=n_samples,
            n_snps=n_snps,
            requested=requested,
            n_cvt=n_cvt,
            lmm_mode=parse_lmm_mode(self.config.lmm_mode),
            n_grid=self.config.n_grid,
            n_refine=self.config.n_refine,
            n_phenotypes=len(pheno_columns),
            mem_budget=self.config.mem_budget,
            max_chunk_size=DEFAULT_LOCO_CONFIG.col_chunk_size
            if self.config.loco
            else None,
            loco=self.config.loco,
        )
        analysis = resolve_analysis_plan(
            self.config,
            execution=execution,
            snps_indices=snps_indices,
            ksnps_indices=ksnps_indices,
        )
        plan = analysis.execution.summary
        logger.info(f"Execution plan: {plan.runner_name} ({plan.reason})")

        # LOCO is single-phenotype (PipelineConfig rejects more) and owns its
        # own per-chromosome kinship and eigendecomposition, so it leaves
        # before the shared eigen acquisition below; its branch runs the same
        # memory preflight on the same plan.
        if isinstance(analysis, LocoAnalysisPlan):
            phenotypes, _n_analyzed = all_pheno_data[pheno_columns[0]]
            return self._run_loco(
                analysis=analysis,
                t_start=t_start,
                phenotypes=phenotypes,
                covariates=covariates,
                valid_mask=valid_mask,
                assoc_path=assoc_path,
            )

        log_pipeline_banner(plan)

        memory_preflight(self.config, analysis.execution)

        # Load/compute eigendecomposition ONCE (shared across phenotypes). The
        # kinship matrix is consumed here; runners use the eigen arrays directly.
        eigenvalues, eigenvectors, kinship_s = self._acquire_eigendecomposition(
            analysis, n_samples, n_valid, valid_mask
        )
        load_s = time.perf_counter() - t_start

        outcome = run_phenotype_loop(
            self.config,
            analysis,
            all_pheno_data,
            valid_mask,
            covariates,
            eigenvalues,
            eigenvectors,
            assoc_path,
            meta,
        )

        total_s = time.perf_counter() - t_start
        logger.info(f"GWAS complete: {outcome.n_tested} SNPs tested in {total_s:.1f}s")

        result = PipelineResult(
            associations=outcome.associations,
            n_samples=n_valid,
            n_snps_tested=outcome.n_tested,
            assoc_path=outcome.assoc_paths[-1],
            assoc_paths=outcome.assoc_paths,
            phenotype_results=outcome.phenotype_results,
            timing=PipelineTiming(
                kinship_s=kinship_s,
                load_s=load_s,
                lmm_s=outcome.lmm_s,
                total_s=total_s,
                rotation_s=outcome.runner_timing.rotation_s,
            ),
            n_covariates=n_cvt,
            pve_estimate=outcome.pve,
            pve_se=outcome.pve_se,
        )
        self._emit_telemetry(result, plan)
        return result

    def _load_phenotypes_and_intersect_masks(
        self,
        pheno_columns: Sequence[int],
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
            pheno, n_anal = self._parse_phenotype_column(col, fam_data)
            all_pheno_data[col] = (pheno, n_anal)
            all_masks.append(compute_valid_mask(pheno, covariates))

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

    def _acquire_eigendecomposition(
        self,
        analysis: StandardAnalysisPlan,
        n_samples: int,
        n_valid: int,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Load or compute the shared eigendecomposition (once for all phenotypes).

        Either reads pre-computed eigen files (-d/-u), or has
        ``_load_kinship_from_source`` produce the kinship matrix over the
        valid samples and eigendecomposes
        it (optionally writing the eigen files). The kinship matrix is consumed
        here — the runners use the eigenvalues/eigenvectors directly.

        Returns:
            ``(eigenvalues, eigenvectors, kinship_s)`` where ``kinship_s`` is the
            wall time spent acquiring the eigendecomposition.
        """
        t_kinship = time.perf_counter()

        source = analysis.eigen_source
        if isinstance(source, ProvidedEigen):
            eigenvalues, eigenvectors = read_eigen_files(
                source.eigenvalue_file,
                source.eigenvector_file,
                n_samples=n_valid,
            )
            logger.info(
                f"Loaded pre-computed eigendecomposition "
                f"({len(eigenvalues)} eigenvalues)"
            )
            if source.ignored_kinship_file is not None:
                logger.warning(
                    "Both kinship (-k) and eigen files (-d/-u) "
                    "provided. Using eigen files; kinship will "
                    "be ignored."
                )
        else:
            K = self._load_kinship_from_source(
                source.source,
                n_samples,
                valid_indices=None if np.all(valid_mask) else np.where(valid_mask)[0],
            )
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K, check_memory=self.config.check_memory
            )
            if source.write_eigen:
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

    def _run_loco(
        self,
        *,
        analysis: LocoAnalysisPlan,
        t_start: float,
        phenotypes: np.ndarray,
        covariates: np.ndarray | None,
        valid_mask: np.ndarray,
        assoc_path: Path,
    ) -> PipelineResult:
        """LOCO branch of the pipeline.

        Entered from ``run`` once the shared preamble has loaded the single
        phenotype and the covariates. Prices the run's one association plan
        through the shared preflight, hands that plan to the LOCO orchestrator
        (which owns its own per-chromosome kinship and eigendecomposition) and
        assembles a PipelineResult.

        Single-phenotype only — multi-phenotype LOCO is rejected at
        PipelineConfig.__post_init__.
        """
        from jamma.lmm import run_lmm_loco

        n_valid = int(np.sum(valid_mask))
        n_cvt = covariates.shape[1] if covariates is not None else 1
        plan = analysis.execution.summary
        log_pipeline_banner(plan)
        memory_preflight(self.config, analysis.execution)

        t_loco = time.perf_counter()
        loco = run_lmm_loco(
            bed_path=self.config.bfile,
            phenotypes=phenotypes,
            covariates=covariates,
            config=analysis.lmm,
            loco=analysis.loco,
            output_path=assoc_path,
            execution=analysis.execution,
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
            phenotype_results=[
                PhenotypeResult(
                    column=self.config.phenotype_columns[0],
                    associations=loco.associations,
                    n_snps_tested=loco.n_tested,
                    assoc_path=assoc_path,
                    pve_estimate=loco.pve,
                    pve_se=loco.pve_se,
                )
            ],
            timing=PipelineTiming(
                lmm_s=loco_s,
                total_s=total_s,
            ),
            n_covariates=n_cvt,
            pve_estimate=loco.pve,
            pve_se=loco.pve_se,
        )
        self._emit_telemetry(result, plan)
        return result
