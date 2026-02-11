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

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.memory import StreamingMemoryBreakdown, estimate_streaming_memory
from jamma.io.covariate import read_covariate_file
from jamma.io.plink import get_plink_metadata
from jamma.kinship import (
    compute_kinship_streaming,
    read_kinship_matrix,
    write_kinship_matrix,
)
from jamma.lmm import run_lmm_association_streaming, run_lmm_loco
from jamma.lmm.chunk import _compute_chunk_size
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


@dataclass
class PipelineResult:
    """Result of a pipeline run.

    Attributes:
        associations: Per-SNP association results. Empty when results are
            written to disk via output_path.
        n_samples: Number of samples with valid phenotypes.
        n_snps_tested: Total number of SNPs in the dataset.
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
    rather than calling sys.exit or typer.Exit. The CLI wrapper catches
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

    def parse_phenotypes(self) -> tuple[np.ndarray, int]:
        """Parse phenotypes from the .fam file.

        Uses vectorized parsing: reads the phenotype column, replaces
        missing indicators ("-9", "NA") with NaN, converts to float64.

        Returns:
            Tuple of (phenotypes array, n_analyzed) where phenotypes has
            NaN for missing values and n_analyzed is the count of valid
            (non-NaN, non-missing) phenotypes.

        Raises:
            ValueError: If no samples have valid phenotypes.
        """
        fam_path = f"{self.config.bfile}.fam"
        fam_data = np.loadtxt(fam_path, dtype=str, usecols=(5,))
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
        self, n_samples: int, n_snps: int
    ) -> StreamingMemoryBreakdown | None:
        """Check memory requirements if memory checking is enabled.

        Computes actual chunk size via _compute_chunk_size, then estimates
        streaming memory. Checks against mem_budget if set, and against
        available system memory.

        Args:
            n_samples: Number of samples in the dataset.
            n_snps: Number of SNPs in the dataset.

        Returns:
            StreamingMemoryBreakdown if check_memory is True, None otherwise.

        Raises:
            MemoryError: If estimated memory exceeds budget or available memory.
        """
        if not self.config.check_memory:
            return None

        actual_chunk = _compute_chunk_size(n_samples, n_snps)
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size=actual_chunk)

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

    def load_kinship(self, n_samples: int) -> np.ndarray:
        """Load or compute the kinship matrix.

        If kinship_file is provided, loads from disk. Otherwise, computes
        from genotypes using streaming kinship computation.

        If save_kinship is True and kinship was computed (not loaded),
        writes it to the output directory.

        Args:
            n_samples: Number of samples (for validation of loaded kinship).

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
            )

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

        return covariates

    def run(self) -> PipelineResult:
        """Execute the full GWAS pipeline.

        Pipeline steps:
        1. Validate inputs
        2. Get PLINK metadata
        3. Check memory requirements
        4. Parse phenotypes
        5. Load kinship matrix
        6. Load covariates
        7. Run LMM association (streaming)

        Returns:
            PipelineResult with associations, counts, output path, and timing.
        """
        t_start = time.perf_counter()

        # 1. Validate
        self.validate_inputs()

        # 2. Metadata
        meta = get_plink_metadata(self.config.bfile)
        n_samples = meta["n_samples"]
        n_snps = meta["n_snps"]

        # 3. Memory check
        self.check_memory_requirements(n_samples, n_snps)

        # 4. Phenotypes
        phenotypes, n_analyzed = self.parse_phenotypes()
        n_filtered = len(phenotypes) - n_analyzed
        logger.info(
            f"Analyzing {n_analyzed} samples with valid phenotypes "
            f"({n_filtered} filtered)"
        )

        # 5. Output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        assoc_path = self.config.output_dir / f"{self.config.output_prefix}.assoc.txt"

        # 6. LOCO branch: skip standard kinship, run LOCO orchestrator
        if self.config.loco:
            covariates = self.load_covariates(n_samples)

            t_loco = time.perf_counter()
            results = run_lmm_loco(
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
            )
            loco_s = time.perf_counter() - t_loco
            total_s = time.perf_counter() - t_start
            logger.info(f"LOCO GWAS complete: {n_snps} SNPs in {total_s:.1f}s")

            return PipelineResult(
                associations=results,
                n_samples=n_analyzed,
                n_snps_tested=n_snps,
                assoc_path=assoc_path,
                timing={
                    "kinship_s": 0.0,
                    "load_s": 0.0,
                    "lmm_s": loco_s,
                    "total_s": total_s,
                },
            )

        # 7. Standard path: Kinship
        t_kinship = time.perf_counter()
        K = self.load_kinship(n_samples)
        kinship_s = time.perf_counter() - t_kinship

        # 8. Covariates
        covariates = self.load_covariates(n_samples)
        load_s = time.perf_counter() - t_start

        # 9. Run LMM
        t_lmm = time.perf_counter()
        results = run_lmm_association_streaming(
            bed_path=self.config.bfile,
            phenotypes=phenotypes,
            kinship=K,
            snp_info=None,
            covariates=covariates,
            maf_threshold=self.config.maf,
            miss_threshold=self.config.miss,
            output_path=assoc_path,
            lmm_mode=self.config.lmm_mode,
            check_memory=False,  # Already checked above
            show_progress=self.config.show_progress,
        )
        lmm_s = time.perf_counter() - t_lmm

        total_s = time.perf_counter() - t_start
        logger.info(f"GWAS complete: {n_snps} SNPs tested in {total_s:.1f}s")

        return PipelineResult(
            associations=results,
            n_samples=n_analyzed,
            n_snps_tested=n_snps,
            assoc_path=assoc_path,
            timing={
                "kinship_s": kinship_s,
                "load_s": load_s,
                "lmm_s": lmm_s,
                "total_s": total_s,
            },
        )
