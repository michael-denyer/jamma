"""The LMM mode model, run configuration, and run results.

``MODE_SPECS`` is the single source of truth for what each ``-lmm`` mode
runs and which columns it reports. ``jamma.lmm.assoc_output`` turns a
mode's columns into output rows.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from jamma.lmm.assoc_output import AssocResult

LmmMode = Literal[1, 2, 3, 4]


def parse_lmm_mode(value: int) -> LmmMode:
    """Narrow a boundary int (a CLI flag, a public-API argument) to LmmMode.

    The one place an int becomes an LmmMode. Inside the package the literal
    type flows through untouched, so a bad mode can only enter through a
    boundary that forgot to call this, and the type checker names it.
    """
    if value not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {value}"
        )
    return cast(LmmMode, value)


@dataclass(frozen=True, slots=True)
class ChunkRunStats:
    """What the chunk runner did and how long each stage took.

    Returned once per phenotype by ``run_lmm_chunk_source_numpy_group``, which
    reports the shared rotation time separately; ``run_single`` folds it into
    ``rotation_s`` on ``LmmRunResult.timing``. Zero everywhere when no SNP
    passed filtering.

    Attributes:
        processed: SNPs the chunk loop handed to the result sink.
        rotation_s: Total UT@G rotation time (seconds).
        compute_s: Total NumPy/C compute time (seconds).
        result_write_s: Total result write time (seconds).
    """

    processed: int = 0
    rotation_s: float = 0.0
    compute_s: float = 0.0
    result_write_s: float = 0.0


@dataclass(frozen=True, slots=True)
class StatColumn:
    """One statistical output column.

    ``array_key`` names the kernel's output array. ``field_name`` is both the
    ``AssocResult`` field and the TSV column header.
    """

    array_key: str
    field_name: str


_META_HEADERS = ("chr", "rs", "ps", "n_miss", "allele1", "allele0", "af")


class LmmTest(enum.Flag):
    """The three association tests a mode can run.

    ``-lmm 4`` runs all three. Every per-mode decision (which null-model
    inputs a kernel needs, which lambdas it optimises, which kernel runs)
    is a membership question against a mode's ``tests``.
    """

    WALD = enum.auto()
    LRT = enum.auto()
    SCORE = enum.auto()


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """Complete specification for one LMM mode.

    ``test_type`` is the mode's name.
    ``tests`` is the set of tests the mode runs.
    ``stat_columns`` defines column order, array keys, and column names.
    """

    test_type: str
    tests: LmmTest
    stat_columns: tuple[StatColumn, ...]

    @property
    def lambda_keys(self) -> tuple[str, ...]:
        """Array keys of the optimised lambdas: REML for Wald, MLE for LRT."""
        keys: tuple[str, ...] = ()
        if LmmTest.WALD in self.tests:
            keys += ("lambdas",)
        if LmmTest.LRT in self.tests:
            keys += ("lambdas_mle",)
        return keys

    @property
    def output_bytes_per_snp(self) -> int:
        """Bytes of float64 result arrays the mode holds per SNP."""
        return 8 * len(self.stat_columns)

    @property
    def header(self) -> str:
        """The ``.assoc.txt`` header line, without the trailing newline."""
        return "\t".join((*_META_HEADERS, *(c.field_name for c in self.stat_columns)))


# ── Column definitions ──────────────────────────────────────────────

_BETA = StatColumn("betas", "beta")
_SE = StatColumn("ses", "se")
_LOGL = StatColumn("logls", "logl_H1")
_L_REMLE = StatColumn("lambdas", "l_remle")
_P_WALD = StatColumn("pwalds", "p_wald")
_L_MLE = StatColumn("lambdas_mle", "l_mle")
_P_LRT = StatColumn("p_lrts", "p_lrt")
_P_SCORE = StatColumn("p_scores", "p_score")


# ── The single source of truth ──────────────────────────────────────

MODE_SPECS: Mapping[LmmMode, ModeSpec] = MappingProxyType(
    {
        1: ModeSpec("wald", LmmTest.WALD, (_BETA, _SE, _LOGL, _L_REMLE, _P_WALD)),
        2: ModeSpec("lrt", LmmTest.LRT, (_LOGL, _L_MLE, _P_LRT)),
        3: ModeSpec("score", LmmTest.SCORE, (_BETA, _SE, _P_SCORE)),
        4: ModeSpec(
            "all",
            LmmTest.WALD | LmmTest.LRT | LmmTest.SCORE,
            (_BETA, _SE, _LOGL, _L_REMLE, _L_MLE, _P_WALD, _P_LRT, _P_SCORE),
        ),
    }
)


def get_spec(mode: int) -> ModeSpec:
    """Look up ModeSpec by lmm_mode int, or raise ValueError."""
    return MODE_SPECS[parse_lmm_mode(mode)]


# Default LMM knobs — single source of truth for the config surface
# (PipelineConfig, LmmConfig, and the CLI/gwas() maf/miss/l_min/l_max options)
# and the runner dispatch entry points, so a default cannot silently drift
# between them. maf/miss/l_min/l_max match GEMMA v0.98.5 CLI defaults; n_grid and
# n_refine are JAMMA's golden-section knobs with no GEMMA equivalent (GEMMA uses
# Brent — see docs/GEMMA_DIVERGENCES.md §6; never "align" these toward GEMMA).
DEFAULT_MAF = 0.01
DEFAULT_MISS = 0.05
DEFAULT_L_MIN = 1e-5
DEFAULT_L_MAX = 1e5
DEFAULT_N_GRID = 50
DEFAULT_N_REFINE = 20

# Minimum coarse-grid resolution. A one-point grid has no bracket: the
# golden-section stage collapses (idx_low == idx_high, so a == b) and every SNP
# silently returns lambda = l_min instead of its optimum. This is a correctness
# bound, not a quality preference. The C kernels enforce the same minimum via
# validate_batch_params in _lmm_support.c — keep the two in step.
MIN_N_GRID = 2

# Golden-section iterations for ~1e-5 relative tolerance on lambda. LmmConfig
# raises a lower value to this rather than rejecting it, and it is the one place
# that does so: every runner reads ``config.n_refine`` as-is.
MIN_N_REFINE = 20


@dataclass(frozen=True)
class LmmConfig:
    """Configuration for LMM association runners.

    Groups the common parameters shared by all runner entry points.
    Frozen, so the values runners read are the values validated here.

    Attributes:
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing. Must be >= 2 —
            a one-point grid has no bracket to refine (see MIN_N_GRID).
        n_refine: Golden section iterations. Raised to MIN_N_REFINE (20,
            for ~1e-5 tolerance) rather than rejected.
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        mem_budget: User-set memory ceiling in GB, or None for no ceiling.
            Reaches LOCO's and -gk's chunk sizers the same way the pipeline's
            preflight already reaches the batch/streaming path.
    """

    maf_threshold: float = DEFAULT_MAF
    miss_threshold: float = DEFAULT_MISS
    l_min: float = DEFAULT_L_MIN
    l_max: float = DEFAULT_L_MAX
    n_grid: int = DEFAULT_N_GRID
    n_refine: int = DEFAULT_N_REFINE
    check_memory: bool = True
    show_progress: bool = True
    lmm_mode: LmmMode = 1
    mem_budget: float | None = None

    def __post_init__(self) -> None:
        parse_lmm_mode(self.lmm_mode)
        if not 0 <= self.maf_threshold <= 0.5:
            raise ValueError(
                f"maf_threshold must be in [0, 0.5], got {self.maf_threshold}"
            )
        if not 0 <= self.miss_threshold <= 1:
            raise ValueError(
                f"miss_threshold must be in [0, 1], got {self.miss_threshold}"
            )
        if self.l_min <= 0:
            raise ValueError(f"l_min must be positive, got {self.l_min}")
        if self.l_max <= self.l_min:
            raise ValueError(
                f"l_max ({self.l_max}) must be greater than l_min ({self.l_min})"
            )
        if self.n_grid < MIN_N_GRID:
            raise ValueError(f"n_grid must be >= {MIN_N_GRID}, got {self.n_grid}")
        if self.n_refine < MIN_N_REFINE:
            object.__setattr__(self, "n_refine", MIN_N_REFINE)


DEFAULT_LMM_CONFIG = LmmConfig()
"""The all-defaults config, shared as the runners' default argument.

LmmConfig is frozen, so one instance is safe to share; naming it keeps the
constructor out of a function signature's default.
"""


@dataclass(frozen=True, slots=True)
class LmmRunResult:
    """Return type for LMM runner functions.

    Bundles per-SNP association results with run-level metadata
    such as heritability estimates.

    Attributes:
        associations: Per-SNP association results. Empty when output_path
            routed results to disk; n_tested still counts them.
        n_tested: Number of SNPs tested, in every mode.
        pve: PVE (proportion of variance explained) from null model REML.
            None if no SNPs passed filtering (early return).
        pve_se: Standard error of PVE from REML second derivative delta method.
            None if not computed or likelihood surface is flat.
        timing: Wall-clock breakdown of the run's chunk loop.
    """

    associations: list[AssocResult]
    n_tested: int
    pve: float | None = None
    pve_se: float | None = None
    timing: ChunkRunStats = field(default_factory=ChunkRunStats)
