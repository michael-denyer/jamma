"""Unified output schema for LMM association results.

Single source of truth for the mapping between lmm_mode (int),
test_type (str), array keys, AssocResult field names, TSV column
headers, and format specifiers.  All other dispatch tables in the
LMM subsystem are derived views of MODE_SPECS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np

if TYPE_CHECKING:
    from jamma.io.plink import PlinkMetadata

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


class RunnerTiming(TypedDict, total=False):
    """Timing breakdown from LMM runner execution.

    All keys are optional because not all runners populate all fields.

    Attributes:
        rotation_s: Total UT@G rotation time (seconds).
        numpy_compute_s: Total NumPy/C compute time (seconds).
        result_write_s: Total result write time (seconds).
    """

    rotation_s: float
    numpy_compute_s: float
    result_write_s: float


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
class StatColumn:
    """One statistical output column.

    Maps an output array key to an AssocResult field name, a TSV
    column header, and a format specifier for numeric output.
    """

    array_key: str
    field_name: str
    header: str
    fmt: str = ".6e"

    def __post_init__(self) -> None:
        for attr in ("array_key", "field_name", "header", "fmt"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val:
                raise ValueError(
                    f"StatColumn.{attr} must be a non-empty string, got {val!r}"
                )
        try:
            f"{0.0:{self.fmt}}"
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid format spec {self.fmt!r}: {e}") from None


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """Complete output specification for one LMM mode.

    ``test_type`` is the string name used for headers and format lookup.
    ``stat_columns`` defines column order, array keys, and formatting.
    """

    test_type: str
    stat_columns: tuple[StatColumn, ...]

    def __post_init__(self) -> None:
        if not self.stat_columns:
            raise ValueError("stat_columns must not be empty")
        keys = [c.array_key for c in self.stat_columns]
        if len(keys) != len(set(keys)):
            dupes = [k for k in keys if keys.count(k) > 1]
            raise ValueError(f"Duplicate array_key in stat_columns: {set(dupes)}")
        names = [c.field_name for c in self.stat_columns]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate field_name in stat_columns: {set(dupes)}")
        headers = [c.header for c in self.stat_columns]
        if len(headers) != len(set(headers)):
            dupes = [h for h in headers if headers.count(h) > 1]
            raise ValueError(f"Duplicate header in stat_columns: {set(dupes)}")


# ── Column definitions ──────────────────────────────────────────────

_BETA = StatColumn("betas", "beta", "beta")
_SE = StatColumn("ses", "se", "se")
_LOGL = StatColumn("logls", "logl_H1", "logl_H1")
_L_REMLE = StatColumn("lambdas", "l_remle", "l_remle")
_P_WALD = StatColumn("pwalds", "p_wald", "p_wald")
_L_MLE = StatColumn("lambdas_mle", "l_mle", "l_mle")
_P_LRT = StatColumn("p_lrts", "p_lrt", "p_lrt")
_P_SCORE = StatColumn("p_scores", "p_score", "p_score")


# ── The single source of truth ──────────────────────────────────────

MODE_SPECS: Mapping[int, ModeSpec] = MappingProxyType(
    {
        1: ModeSpec("wald", (_BETA, _SE, _LOGL, _L_REMLE, _P_WALD)),
        2: ModeSpec("lrt", (_L_MLE, _P_LRT)),
        3: ModeSpec("score", (_BETA, _SE, _P_SCORE)),
        4: ModeSpec(
            "all",
            (_BETA, _SE, _LOGL, _L_REMLE, _L_MLE, _P_WALD, _P_LRT, _P_SCORE),
        ),
    }
)


def get_spec(mode: int) -> ModeSpec:
    """Look up ModeSpec by lmm_mode int, or raise ValueError."""
    if mode not in MODE_SPECS:
        raise ValueError(f"Unknown lmm_mode={mode}; expected one of {list(MODE_SPECS)}")
    return MODE_SPECS[mode]


# ── Derived views (replace old per-module dispatch tables) ──────────

TEST_TYPE_MAP: dict[int, str] = {m: s.test_type for m, s in MODE_SPECS.items()}

ACCUM_KEYS: dict[int, tuple[str, ...]] = {
    m: tuple(c.array_key for c in s.stat_columns) for m, s in MODE_SPECS.items()
}

RESULT_FIELDS: dict[int, dict[str, str]] = {
    m: {c.array_key: c.field_name for c in s.stat_columns}
    for m, s in MODE_SPECS.items()
}

FORMAT_COLUMNS: dict[str, list[str]] = {
    s.test_type: [c.header for c in s.stat_columns] for s in MODE_SPECS.values()
}

_HEADER_PREFIX = "chr\trs\tps\tn_miss\tallele1\tallele0\taf"
HEADERS: dict[str, str] = {
    tt: _HEADER_PREFIX + "\t" + "\t".join(cols) for tt, cols in FORMAT_COLUMNS.items()
}

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
DEFAULT_N_REFINE = 10

# Minimum coarse-grid resolution. A one-point grid has no bracket: the
# golden-section stage collapses (idx_low == idx_high, so a == b) and every SNP
# silently returns lambda = l_min instead of its optimum. This is a correctness
# bound, not a quality preference. The C kernel enforces the same minimum in
# validate_batch_params (src/jamma/lmm/_lmm_accel.c) — keep the two in step.
MIN_N_GRID = 2


@dataclass(frozen=True)
class LmmConfig:
    """Configuration for LMM association runners.

    Groups the common parameters shared by all runner entry points.
    Frozen to prevent accidental mutation — runners clamp values (e.g.,
    n_refine >= 20) on locals after reading them off the config.

    Attributes:
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing. Must be >= 2 —
            a one-point grid has no bracket to refine (see MIN_N_GRID).
        n_refine: Golden section iterations (clamped to min 20 internally
            for ~1e-5 tolerance, so low values are raised rather than
            rejected).
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
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

    def __post_init__(self) -> None:
        if self.lmm_mode not in (1, 2, 3, 4):
            raise ValueError(
                f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), "
                f"got {self.lmm_mode}"
            )
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

    associations: list  # list[AssocResult] -- avoid circular import with stats.py
    n_tested: int
    pve: float | None = None
    pve_se: float | None = None
    timing: RunnerTiming = field(default_factory=RunnerTiming)


@dataclass(frozen=True, slots=True)
class LocoResult:
    """Return type for run_lmm_loco.

    Attributes:
        associations: Per-SNP results in biological chromosome order.
            Empty list if output_path is set (results written to disk).
        n_tested: Total number of SNPs tested across all chromosomes.
        pve: PVE estimate from the first chromosome's null model.
        pve_se: Standard error of PVE via delta method.
            None if likelihood surface is flat.
    """

    associations: list  # list[AssocResult]
    n_tested: int
    pve: float | None = None
    pve_se: float | None = None


@dataclass(frozen=True, slots=True)
class SnpMeta:
    """SNP metadata as one array per column, indexed by global SNP index.

    Writers and result builders slice these arrays directly; nothing
    materialises a per-SNP dict. ``pos`` is normalised to int64 and ``chr``
    to str at construction, so downstream formatting needs no coercion.

    Attributes:
        chr: Chromosome identifier per SNP (str).
        rs: SNP identifier / rsID per SNP (str).
        pos: Base-pair position per SNP (int64).
        a1: Minor allele per SNP (str).
        a0: Major allele per SNP (str).
    """

    chr: np.ndarray
    rs: np.ndarray
    pos: np.ndarray
    a1: np.ndarray
    a0: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.rs)
        for name in ("chr", "pos", "a1", "a0"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"SnpMeta columns must share one length; "
                    f"{name} has {len(getattr(self, name))}, rs has {n}"
                )

    def __len__(self) -> int:
        return len(self.rs)

    @classmethod
    def from_plink_meta(cls, meta: PlinkMetadata) -> SnpMeta:
        """Build from get_plink_metadata output without copying string data."""
        return cls(
            chr=np.asarray(meta.chromosome).astype(str),
            rs=np.asarray(meta.sid),
            pos=np.asarray(meta.bp_position, dtype=np.int64),
            a1=np.asarray(meta.allele_1),
            a0=np.asarray(meta.allele_2),
        )

    @classmethod
    def from_dicts(cls, snp_info: list) -> SnpMeta:
        """Parse a caller-supplied list of per-SNP dicts.

        The boundary for the public batch API. Requires the canonical keys
        chr/rs/pos/a1/a0 on every dict; raises KeyError on the first miss.
        """
        return cls(
            chr=np.array([str(s["chr"]) for s in snp_info]),
            rs=np.array([s["rs"] for s in snp_info]),
            pos=np.array([int(s["pos"]) for s in snp_info], dtype=np.int64),
            a1=np.array([s["a1"] for s in snp_info]),
            a0=np.array([s["a0"] for s in snp_info]),
        )
