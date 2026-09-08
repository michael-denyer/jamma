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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.constants import Env
from jamma.core.eigen_plan import EigenDriverPlan
from jamma.io.plink import get_plink_metadata, validate_plink_dimensions
from jamma.io.snp_list import resolve_snp_list_file
from jamma.kinship import (
    compute_kinship_streaming,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm.association_plan import (
    ExecutionPlan,
    KinshipShape,
    plan_association,
)
from jamma.lmm.eigen import center_kinship, eigendecompose_kinship
from jamma.lmm.eigen_io import (
    managed_eigen_pair_exists,
    read_eigen_files,
    write_eigen_files,
)
from jamma.lmm.loco_config import DEFAULT_LOCO_CONFIG
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
from jamma.pipeline_samples import load_analysed_samples

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

    def _load_kinship_from_source(
        self,
        source: KinshipSource,
        n_samples: int,
        kinship: KinshipShape,
        valid_indices: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> np.ndarray:
        """Load or compute the kinship matrix over the valid samples.

        A ``ProvidedKinship`` source loads from disk; ``ComputedKinship``
        streams from genotypes. Derive the source with
        ``pipeline_plan.resolve_kinship_source`` so it cannot drift from
        the resolver's choice.

        If weights are provided, applies individual weights to K via
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
            kinship: The matrix order the plan resolved, full or analysed.
            valid_indices: Sample indices to keep, or None for all samples.
                Must be sorted, unique, and within [0, n_samples).
            weights: Weights already selected into analyzed-sample order, or None.

        Returns:
            Kinship matrix of shape (n_out, n_out) where n_out = len(valid_indices)
            or n_samples.
        """
        if valid_indices is not None:
            from jamma.kinship import validate_valid_indices

            validate_valid_indices(valid_indices, n_samples)

        full = kinship.n_samples == n_samples

        if isinstance(source, ProvidedKinship):
            logger.info(f"Loading kinship from {source.path}")
            K = read_kinship_matrix(source.path, n_samples=n_samples)
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
        if weights is not None:
            from jamma.io.weight import apply_individual_weights

            logger.info(f"Applying individual weights from {self.config.weight_file}")
            K = apply_individual_weights(K, weights)

        return K

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

        pheno_columns = self.config.phenotype_columns
        samples = load_analysed_samples(self.config, n_samples)
        covariates = samples.covariates
        valid_mask = samples.valid_mask
        analyzed_sample_indices = samples.basis.positions
        n_valid = samples.basis.analyzed_sample_count
        n_cvt = samples.n_covariates

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
            return self._run_loco(
                analysis=analysis,
                t_start=t_start,
                phenotypes=samples.phenotypes[pheno_columns[0]],
                covariates=covariates,
                valid_mask=valid_mask,
                analyzed_sample_indices=analyzed_sample_indices,
                assoc_path=assoc_path,
            )

        log_pipeline_banner(plan)

        eigen_plan = memory_preflight(self.config, analysis.execution)

        # Load/compute eigendecomposition ONCE (shared across phenotypes). The
        # kinship matrix is consumed here; runners use the eigen arrays directly.
        eigenvalues, eigenvectors, kinship_s = self._acquire_eigendecomposition(
            analysis, n_samples, n_valid, analyzed_sample_indices, eigen_plan=eigen_plan
        )
        load_s = time.perf_counter() - t_start

        outcome = run_phenotype_loop(
            self.config,
            analysis,
            samples.phenotypes,
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
            analyzed_sample_indices=analyzed_sample_indices,
        )
        self._emit_telemetry(result, plan)
        return result

    def _acquire_eigendecomposition(
        self,
        analysis: StandardAnalysisPlan,
        n_samples: int,
        n_valid: int,
        analyzed_sample_indices: np.ndarray,
        *,
        eigen_plan: EigenDriverPlan | None = None,
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
            from jamma.io.weight import (
                apply_weights_to_eigenvectors,
                read_analysis_weights,
            )

            valid_indices = None if n_valid == n_samples else analyzed_sample_indices
            weights = (
                read_analysis_weights(self.config.weight_file, n_samples, valid_indices)
                if self.config.weight_file is not None
                else None
            )
            K = self._load_kinship_from_source(
                source.source,
                n_samples,
                analysis.execution.resolved_kinship,
                valid_indices=valid_indices,
                weights=weights,
            )
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K,
                check_memory=self.config.check_memory,
                mem_budget=self.config.mem_budget,
                eigen_plan=eigen_plan,
            )
            if weights is not None:
                # GEMMA -widv scales the eigenvector rows by sqrt(w) after
                # decomposing D^-1/2 K D^-1/2, and writes the scaled U, so a
                # weighted eigenU round-trips through -d/-u without -widv.
                eigenvectors = apply_weights_to_eigenvectors(eigenvectors, weights)
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
        analyzed_sample_indices: np.ndarray,
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
            analyzed_sample_indices=analyzed_sample_indices,
        )
        self._emit_telemetry(result, plan)
        return result
