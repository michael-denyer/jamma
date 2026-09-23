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

import jamma
from jamma.core import memory
from jamma.core.constants import Env
from jamma.core.eigen_plan import EigenDriverPlan
from jamma.core.telemetry import BenchmarkRecord, append_benchmark_record
from jamma.io.plink import (
    PlinkMetadata,
    get_plink_metadata,
    validate_plink_dimensions,
)
from jamma.io.snp_list import resolve_snp_list_file
from jamma.io.weight import (
    apply_individual_weights,
    apply_weights_to_eigenvectors,
    read_analysis_weights,
)
from jamma.kinship import (
    compute_kinship_streaming,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm.association_plan import (
    VALID_BACKENDS,
    BackendRequest,
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
from jamma.lmm.genotype_source import SampleBasis
from jamma.lmm.loco import LocoRun, run_loco
from jamma.lmm.loco_config import DEFAULT_LOCO_CONFIG
from jamma.lmm.loco_eigen import plan_loco_eigen_driver
from jamma.lmm.prepare_common import AnalysedPhenotype
from jamma.pipeline_banner import log_dataset_banner, log_pipeline_banner
from jamma.pipeline_config import (
    KinshipResult,
    PhenotypeResult,
    PipelineConfig,
    PipelineResult,
    PipelineTiming,
    ProvidedEigen,
    ProvidedKinship,
)
from jamma.pipeline_memory import memory_preflight
from jamma.pipeline_phenotype_loop import run_phenotype_loop
from jamma.pipeline_plan import (
    KinshipSource,
    LocoAnalysisPlan,
    StandardAnalysisPlan,
    resolve_analysis_plan,
)
from jamma.pipeline_samples import AnalysedSamples, load_analysed_samples

__all__ = [
    "BackendRequest",
    "KinshipResult",
    "PhenotypeResult",
    "PipelineConfig",
    "PipelineResult",
    "PipelineRunner",
]


def requested_backend(config: PipelineConfig) -> BackendRequest:
    """Resolve the backend request, letting ``JAMMA_BACKEND`` override config.

    Args:
        config: Pipeline config carrying the validated ``backend`` field.

    Returns:
        The environment override when set, otherwise ``config.backend``.

    Raises:
        ValueError: If ``JAMMA_BACKEND`` is not a recognised backend.
    """
    env_backend = Env.current().backend_raw
    if env_backend is None:
        logger.info(f"Backend request: {config.backend} (config)")
        return config.backend
    if env_backend not in VALID_BACKENDS:
        raise ValueError(
            f"JAMMA_BACKEND must be one of {VALID_BACKENDS}, got {env_backend!r}"
        )
    logger.info(f"Backend request: {env_backend} (JAMMA_BACKEND)")
    return env_backend


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

        Only checks that need the filesystem live here. Everything decidable
        from the config alone, including how the kinship and eigen fields
        combine, is already guaranteed by PipelineConfig.__post_init__.

        Raises:
            FileNotFoundError: If PLINK files (.bed, .bim, .fam) are missing,
                or if an input file the config names is missing.
            ValueError: If the .bed size disagrees with the .fam and .bim.
        """
        bfile = self.config.bfile
        for ext in (".bed", ".bim", ".fam"):
            p = Path(f"{bfile}{ext}")
            if not p.exists():
                raise FileNotFoundError(f"PLINK {ext} file not found: {p}")

        # Validate .bed file size matches .fam/.bim dimensions (VALID-01)
        validate_plink_dimensions(bfile)

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
            self.config.eigenvalue_file is not None
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

    def _load_kinship_from_source(
        self,
        source: KinshipSource,
        kinship: KinshipShape,
        basis: SampleBasis,
        weights: np.ndarray | None,
    ) -> np.ndarray:
        """Load or compute the kinship matrix over the valid samples.

        A ``ProvidedKinship`` source loads from disk; ``ComputedKinship``
        streams from genotypes. ``resolve_analysis_plan`` derives the source
        from ``PipelineConfig.source()``.

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
            kinship: The matrix order the plan resolved, full or analysed.
            basis: The analysed samples within the PLINK sample order.
            weights: Weights already selected into analyzed-sample order, or None.

        Returns:
            Kinship matrix of shape (n_valid, n_valid) over ``basis``.
        """
        n_samples = basis.source_row_count
        valid_indices = None if basis.is_all_samples else basis.positions
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
        6. Check memory against the selected plan
        7. Dispatch on the plan. LOCO hands the phenotype to its orchestrator,
           which owns per-chromosome kinship and eigendecomposition. The
           standard path loads eigen files or the kinship matrix once, then
           runs the per-phenotype loop.

        Returns:
            PipelineResult with per-phenotype results and timing.
        """
        t_start = time.perf_counter()

        # Before any disk read, so a bad JAMMA_BACKEND fails first.
        requested = requested_backend(self.config)
        # Read once and pass it down. get_plink_metadata parses the whole .bim
        # (sid, chromosome, bp_position and both allele arrays).
        meta = get_plink_metadata(self.config.bfile)

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
        n_valid = samples.basis.analyzed_sample_count

        log_dataset_banner(
            n_samples,
            n_valid,
            n_snps,
            n_covariates=samples.n_covariates,
            n_phenotypes=len(pheno_columns),
        )
        warn_if_small_sample(n_valid)

        # Select the plan once, with the post-mask sample count and the real
        # n_cvt: masking can reduce n_valid below the PLINK-header n_samples,
        # and Uab sizing depends on n_cvt. A prior version selected twice
        # (once here with the pre-mask n_samples, once again after masking),
        # re-running estimate_lmm_memory both times; this is the single call.
        execution = plan_association(
            n_valid,
            n_snps,
            config=self.config.lmm_config(),
            backend="loco" if self.config.loco else requested,
            n_cvt=samples.n_covariates,
            n_input_samples=n_samples,
            n_phenotypes=len(pheno_columns),
            max_chunk_size=DEFAULT_LOCO_CONFIG.col_chunk_size
            if self.config.loco
            else None,
        )
        analysis = resolve_analysis_plan(
            self.config,
            execution=execution,
            snps_indices=snps_indices,
            ksnps_indices=ksnps_indices,
        )
        plan = analysis.execution.summary
        logger.info(f"Execution plan: {plan.runner_name} ({plan.reason})")
        log_pipeline_banner(plan)
        eigen_plan = memory_preflight(analysis, check_memory=self.config.check_memory)

        match analysis:
            case LocoAnalysisPlan():
                phenotype_results, timing = self._associate_loco(
                    analysis, samples, meta, assoc_path, eigen_plan
                )
            case StandardAnalysisPlan():
                phenotype_results, timing = self._associate_standard(
                    analysis, samples, meta, assoc_path, eigen_plan, t_start
                )

        timing.total_s = time.perf_counter() - t_start
        result = PipelineResult(
            phenotype_results=phenotype_results,
            n_samples=n_valid,
            timing=timing,
            n_covariates=samples.n_covariates,
            analyzed_sample_indices=samples.basis.positions,
        )
        logger.info(
            f"GWAS complete: {result.n_snps_tested} SNPs tested "
            f"in {timing.total_s:.1f}s"
        )
        self._emit_telemetry(result, plan)
        return result

    def _associate_standard(
        self,
        analysis: StandardAnalysisPlan,
        samples: AnalysedSamples,
        meta: PlinkMetadata,
        assoc_path: Path,
        eigen_plan: EigenDriverPlan | None,
        t_start: float,
    ) -> tuple[list[PhenotypeResult], PipelineTiming]:
        """Decompose the kinship once and run every phenotype over it."""
        eigenvalues, eigenvectors, kinship_s = self._acquire_eigendecomposition(
            analysis, samples, eigen_plan=eigen_plan
        )
        load_s = time.perf_counter() - t_start

        t_lmm = time.perf_counter()
        phenotype_results, rotation_s = run_phenotype_loop(
            self.config,
            analysis,
            samples,
            eigenvalues,
            eigenvectors,
            assoc_path,
            meta,
        )
        return phenotype_results, PipelineTiming(
            kinship_s=kinship_s,
            load_s=load_s,
            lmm_s=time.perf_counter() - t_lmm,
            rotation_s=rotation_s,
        )

    def _associate_loco(
        self,
        analysis: LocoAnalysisPlan,
        samples: AnalysedSamples,
        meta: PlinkMetadata,
        assoc_path: Path,
        eigen_plan: EigenDriverPlan | None,
    ) -> tuple[list[PhenotypeResult], PipelineTiming]:
        """Hand the single phenotype to the LOCO orchestrator.

        The orchestrator owns its per-chromosome kinship and
        eigendecomposition. Multi-phenotype LOCO is rejected at
        ``PipelineConfig.__post_init__``.
        """
        column = self.config.phenotype_columns[0]
        t_loco = time.perf_counter()
        execution = analysis.execution
        run = LocoRun(
            self.config.bfile,
            meta,
            AnalysedPhenotype.from_mask(
                samples.phenotypes[column], samples.covariates, samples.valid_mask
            ),
            analysis.lmm,
            analysis.loco,
            execution,
            eigen_plan
            if eigen_plan is not None
            else plan_loco_eigen_driver(execution, memory.available_ram_gb()),
        )
        loco = run_loco(run, assoc_path)
        phenotype = PhenotypeResult(
            column=column,
            associations=loco.associations,
            n_snps_tested=loco.n_tested,
            assoc_path=assoc_path,
            pve_estimate=loco.pve,
            pve_se=loco.pve_se,
        )
        return [phenotype], PipelineTiming(lmm_s=time.perf_counter() - t_loco)

    def _acquire_eigendecomposition(
        self,
        analysis: StandardAnalysisPlan,
        samples: AnalysedSamples,
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
                n_samples=samples.basis.analyzed_sample_count,
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
            weights = (
                read_analysis_weights(
                    self.config.weight_file,
                    samples.basis.source_row_count,
                    samples.filter_indices,
                )
                if self.config.weight_file is not None
                else None
            )
            K = self._load_kinship_from_source(
                source.source,
                analysis.execution.resolved_kinship,
                samples.basis,
                weights,
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
