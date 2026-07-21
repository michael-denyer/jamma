"""Unified output schema for LMM association results.

Single source of truth for the mapping between lmm_mode (int),
test_type (str), array keys, AssocResult field names, TSV column
headers, and format specifiers.  All other dispatch tables in the
LMM subsystem are derived views of MODE_SPECS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict

# LmmMode type alias (kept local to avoid circular imports with compute_numpy)
LmmMode = int


class RunnerTiming(TypedDict, total=False):
    """Timing breakdown from LMM runner execution.

    All keys are optional because not all runners populate all fields.
    For example, ``rotation_exposed_s`` only appears in multi-chunk runs.

    Attributes:
        rotation_s: Total UT@G rotation time (seconds).
        rotation_exposed_s: Rotation time exposed (not overlapped) by compute.
        numpy_compute_s: Total NumPy/C compute time (seconds).
        result_write_s: Total result write time (seconds).
    """

    rotation_s: float
    rotation_exposed_s: float
    numpy_compute_s: float
    result_write_s: float


class PipelineTiming(TypedDict, total=False):
    """Timing breakdown from pipeline execution.

    All keys are optional; keys from the runner are merged at pipeline exit.

    Attributes:
        kinship_s: Kinship load/compute time (seconds).
        load_s: Total data loading time through kinship (seconds).
        lmm_s: LMM association runtime (seconds).
        total_s: Total pipeline wall time (seconds).
        rotation_s: UT@G rotation time from the runner (seconds).
        rotation_exposed_s: Exposed rotation time from the runner (seconds).
    """

    kinship_s: float
    load_s: float
    lmm_s: float
    total_s: float
    rotation_s: float
    rotation_exposed_s: float


class GWASTiming(TypedDict, total=False):
    """Timing breakdown from GWAS API execution.

    Subset of PipelineTiming exposed through the public gwas() API.

    Attributes:
        kinship_s: Kinship load/compute time (seconds).
        lmm_s: LMM association runtime (seconds).
        total_s: Total pipeline wall time (seconds).
    """

    kinship_s: float
    lmm_s: float
    total_s: float


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


@dataclass(frozen=True)
class LmmConfig:
    """Configuration for LMM association runners.

    Groups the common parameters shared by all runner entry points.
    Frozen to prevent accidental mutation — runners clamp values (e.g.,
    n_refine >= 20) on local variables after unpacking via as_kwargs().

    Attributes:
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (clamped to min 20 internally
            for ~1e-5 tolerance).
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
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2, got {self.n_grid}")

    def as_kwargs(self) -> dict:
        """Return config fields as a dict suitable for unpacking into runner kwargs.

        Maps config field names to the parameter names used by runner functions.
        This eliminates the duplicated 10-line unpacking blocks in each runner.

        Returns:
            Dict with keys matching runner function parameters.
        """
        return {
            "maf_threshold": self.maf_threshold,
            "miss_threshold": self.miss_threshold,
            "l_min": self.l_min,
            "l_max": self.l_max,
            "n_grid": self.n_grid,
            "n_refine": self.n_refine,
            "check_memory": self.check_memory,
            "show_progress": self.show_progress,
            "lmm_mode": self.lmm_mode,
        }


@dataclass(frozen=True, slots=True)
class LmmRunResult:
    """Return type for LMM runner functions.

    Bundles per-SNP association results with run-level metadata
    such as heritability estimates.

    Attributes:
        associations: Per-SNP association results.
        pve: PVE (proportion of variance explained) from null model REML.
            None if no SNPs passed filtering (early return).
        pve_se: Standard error of PVE from REML second derivative delta method.
            None if not computed or likelihood surface is flat.
        n_tested: Number of SNPs tested. Populated by batch runners
            (batch and streaming runners) when output_path is set
            (associations list is empty). None when associations list is
            populated (backward compat).
    """

    associations: list  # list[AssocResult] -- avoid circular import with stats.py
    pve: float | None = None
    pve_se: float | None = None
    n_tested: int | None = None

    @property
    def snp_count(self) -> int:
        """Number of SNPs tested, from n_tested or len(associations)."""
        return self.n_tested if self.n_tested is not None else len(self.associations)


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


class LazySnpMeta:
    """Lazy view over PLINK metadata arrays, avoiding per-SNP dict materialization.

    Instead of building a list of n_snps dicts at construction time, this wrapper
    holds references to the underlying metadata arrays and materializes a single
    dict on each __getitem__ access. This saves O(n_snps) dict + string objects.

    Compatible with all snp_info consumers that use integer indexing (snp_info[idx]).

    Items are dicts with keys:
        chr: Chromosome identifier (str).
        rs: SNP identifier / rsID (str).
        pos: Base-pair position (int).
        a1: Minor allele (str).
        a0: Major allele (str).
    """

    __slots__ = ("_a0", "_a1", "_chr", "_pos", "_rs")

    def __init__(self, meta: dict) -> None:
        self._chr = meta["chromosome"]
        self._rs = meta["sid"]
        self._pos = meta["bp_position"]
        self._a1 = meta["allele_1"]
        self._a0 = meta["allele_2"]

    def __len__(self) -> int:
        return len(self._rs)

    def __getitem__(self, i: int | slice) -> dict | list[dict]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return {
            "chr": str(self._chr[i]),
            "rs": self._rs[i],
            "pos": int(self._pos[i]),
            "a1": self._a1[i],
            "a0": self._a0[i],
        }
