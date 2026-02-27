"""Unified output schema for LMM association results.

Single source of truth for the mapping between lmm_mode (int),
test_type (str), JAX array keys, AssocResult field names, TSV column
headers, and format specifiers.  All other dispatch tables in the
LMM subsystem are derived views of MODE_SPECS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class StatColumn:
    """One statistical output column.

    Maps a JAX output array key to an AssocResult field name, a TSV
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


class LazySnpMeta:
    """Lazy view over PLINK metadata arrays, avoiding per-SNP dict materialization.

    Instead of building a list of n_snps dicts at construction time, this wrapper
    holds references to the underlying metadata arrays and materializes a single
    dict on each __getitem__ access. This saves O(n_snps) dict + string objects.

    Compatible with all snp_info consumers that use integer indexing (snp_info[idx]).
    """

    __slots__ = ("_chr", "_rs", "_pos", "_a1", "_a0")

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
